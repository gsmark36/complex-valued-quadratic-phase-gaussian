import gc
import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import torch
import odak
import lpips
import flip_evaluator as flip
import pycvvdp
from argparse import Namespace
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from model_2d_gaussian_rgb import Gaussians2D, Scene2D, make_trainable_2d
from utils import set_seed, GaussianLoss, Adan, propagator, console_only_print, multiplane_loss, visualize_gaussian_positions


log_debug = True  # redirect stdout to <result_dir>/log.txt
logging.getLogger('odak').setLevel(logging.WARNING)  # silence per-image save messages


def load_target_image(image_path, img_size):
    """Load an RGB target image as a [3, H, W] tensor in [0, 1]"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target image not found: {image_path}")
    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size, Image.LANCZOS)
    img_array = np.array(img) / 255.0
    target_tensor = torch.from_numpy(img_array).float()
    target_tensor = target_tensor.permute(2, 0, 1)  # (C, H, W)
    print(f"Loaded target image: {image_path}, size: {img_size}")
    return target_tensor


def load_depth_image(depth_path, img_size):
    """Load a depth map as a [1, H, W] tensor in [0, 1]"""
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth image not found: {depth_path}")
    depth_img = Image.open(depth_path).convert('L')
    depth_img = depth_img.resize(img_size, Image.LANCZOS)
    depth_array = np.array(depth_img) / 255.0
    depth_tensor = torch.from_numpy(depth_array).float().unsqueeze(0)  # (1, H, W)
    print(f"Loaded depth image: {depth_path}, size: {img_size}")
    return depth_tensor


def calculate_psnr(pred, target):
    """PSNR between prediction and target tensors"""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    return peak_signal_noise_ratio(target_np, pred_np)


class MeansCosineScheduler:
    """Cosine-anneal the learning rate of the 'means' parameter group only"""
    def __init__(self, optimizer, num_itrs, eta_min=1e-3):
        self.optimizer = optimizer
        self.group_indices = [i for i, g in enumerate(optimizer.param_groups) if g.get("name") == "means"]
        temp_optimizer = torch.optim.SGD([{'params': optimizer.param_groups[i]['params'],
                                           'lr': optimizer.param_groups[i]['lr']} for i in self.group_indices])
        self.decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(temp_optimizer, T_max=max(1, num_itrs), eta_min=eta_min)

    def step(self):
        self.decay_scheduler.step()
        for i, group_idx in enumerate(self.group_indices):
            self.optimizer.param_groups[group_idx]['lr'] = self.decay_scheduler.get_last_lr()[i]


def setup_optimizer(gaussians, num_itrs, lr=0.01):
    """Adan with per-parameter-group learning rates"""
    param_groups = [
        {'params': [gaussians.means_2d], 'lr': lr, 'name': 'means'},
        {'params': [gaussians.pre_act_scales], 'lr': 0.005, 'name': 'scales'},
        {'params': [gaussians.colours, gaussians.pre_act_phase], 'lr': 0.0025, 'name': 'amplitude_phase'},
        {'params': [gaussians.pre_act_opacities], 'lr': 0.025, 'name': 'opacity'},
        {'params': [gaussians.pre_act_rotation], 'lr': 0.001, 'name': 'rotation'},
        {'params': [gaussians.curvature], 'lr': 0.001, 'name': 'curvature'},
    ]
    optimizer = Adan(param_groups, lr=0, eps=1e-8, foreach=False)
    trainable_params = [p for g in param_groups for p in g['params']]
    scheduler = MeansCosineScheduler(optimizer, num_itrs)
    return optimizer, scheduler, trainable_params


def reconstruct(scene, propagator, img_size):
    """Render the hologram and propagate it to every depth plane, get intensities [planes, C, pH, pW]"""
    hologram_complex = scene.render(img_size)
    phase_map = odak.learn.wave.calculate_phase(hologram_complex) % (2 * odak.pi)
    amplitude = torch.clamp(odak.learn.wave.calculate_amplitude(hologram_complex), min=0.0, max=1.0)
    reconstruction_intensities_sum = propagator.reconstruct(phase_map, amplitude=amplitude, no_grad=False)
    return torch.sum(reconstruction_intensities_sum, dim=0), phase_map, amplitude


def evaluate(recons, targets, H, W, lpips_fn, cvvdp_metric):
    """PSNR / SSIM / LPIPS / FLIP / CVVDP for every supervised plane"""
    metrics = {k: [] for k in ["psnr", "ssim", "lpips", "flip", "cvvdp"]}
    for plane_idx in range(min(len(recons), len(targets))):
        recon = torch.clamp(recons[plane_idx], min=0.0, max=1.0)
        recon = odak.learn.tools.crop_center(recon, size=(H, W))
        target = targets[plane_idx]

        recon_np = recon.permute(1, 2, 0).detach().cpu().numpy()
        target_np = target.permute(1, 2, 0).detach().cpu().numpy()
        metrics["psnr"].append(float(peak_signal_noise_ratio(target_np, recon_np, data_range=1.0)))
        metrics["ssim"].append(float(structural_similarity(target_np, recon_np, data_range=1.0, channel_axis=2)))

        # LPIPS: input range [-1, 1]
        metrics["lpips"].append(lpips_fn(2 * recon.unsqueeze(0) - 1, 2 * target.unsqueeze(0) - 1).item())

        # FLIP: HxWx3 numpy in [0, 1], LDR (reference, test)
        recon_rgb = recon.permute(1, 2, 0).contiguous().detach().cpu().numpy().astype(np.float32)
        target_rgb = target.permute(1, 2, 0).contiguous().detach().cpu().numpy().astype(np.float32)
        _, flip_val, _ = flip.evaluate(target_rgb, recon_rgb, "LDR")
        metrics["flip"].append(float(flip_val))

        # CVVDP: 3-channel image in [0, 1], single frame
        jod, _ = cvvdp_metric.predict(recon, target, dim_order="CHW", frames_per_second=0)
        metrics["cvvdp"].append(float(jod))
    return metrics


def run_training_2d(args, args_prop, propagator, result_dir, checkpoint_dir, device):
    img_size = tuple(args.img_size)

    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    cvvdp_metric = pycvvdp.cvvdp(display_name='standard_4k', heatmap=None, quiet=True)

    target_image = load_target_image(args.target_image_path, img_size).to(device)
    C, H, W = target_image.shape

    if args.depth_path:
        depth_image = load_depth_image(args.depth_path, img_size).to(device)
    else:
        if args_prop.num_planes > 1:
            raise ValueError("A depth map (--depth_path) is required for more than one plane.")
        depth_image = torch.ones((1, H, W), device=device)
        print("No depth map: single plane with default depth (all ones)")

    gaussians = Gaussians2D(num_points=args.num_gaussians, img_size=img_size, device=device, args_prop=args_prop)
    scene = Scene2D(gaussians, args_prop)
    make_trainable_2d(gaussians)

    # Curvature warm-up
    # Warm-up stage: freeze curvature to zero for all primitives (planar Gaussians)
    # Refinement stage: unlock curvature to optimize high-frequency details
    gaussians.curv_mode = 'off' if args.primitive == 'flat' else args.curv_mode
    if args.primitive == 'flat':
        gaussians.curvature.requires_grad_(False)
        curv_warmup_active = False
    else:
        curv_warmup_active = args.warmup_iters > 0
        if curv_warmup_active:
            gaussians.curvature.requires_grad_(False)
    print(f"Primitive={args.primitive} | curv_mode={gaussians.curv_mode} | "
          f"warmup={args.warmup_iters if args.primitive != 'flat' else 'n/a (frozen)'} "
          f"| nyquist_frac={gaussians.curv_nyquist_frac} | phase_iso={bool(args.phase_iso)}")

    optimizer, scheduler, trainable_params = setup_optimizer(gaussians, num_itrs=args.num_itrs, lr=args.lr)

    total_params = sum(p.numel() for p in trainable_params)
    if args.primitive == 'flat':
        total_params -= gaussians.curvature.numel()
    print(f"Total trainable parameters: {total_params:,}")

    # Synthesized multi-plane targets (defocus-blurred by depth)
    targets, loss_function, mask = multiplane_loss(target_image=target_image, target_depth=depth_image, args_prop=args_prop)
    for idx, target in enumerate(targets):
        odak.learn.tools.save_image(f"{result_dir}/target_{idx}.png", target, cmin=0., cmax=1.0)

    running_losses, running_ssim_losses, running_psnrs = [], [], []
    best_psnr = 0.0

    start_time = time.time()
    pbar = tqdm(range(args.num_itrs), desc='Training 2D Gaussians (RGB)', miniters=100)
    for itr in pbar:
        # Unlock curvature once warm-up stage is over
        if curv_warmup_active and itr == args.warmup_iters:
            gaussians.curvature.requires_grad_(True)
            curv_warmup_active = False
            with console_only_print():
                print(f"[warm-up] curvature unfrozen at iter {itr}")

        optimizer.zero_grad()

        reconstruction_intensities, _, _ = reconstruct(scene, propagator, img_size)

        loss = 0.
        for idx, (reconstruction_intensity, target) in enumerate(zip(reconstruction_intensities, targets)):
            pred_cropped = odak.learn.tools.crop_center(reconstruction_intensity, size=(H, W))
            pred_cropped = torch.clamp(pred_cropped, min=0.0, max=1.0)
            loss += loss_function(pred_cropped, target, idx, inject_noise=True)
            ssim_loss = GaussianLoss(pred_cropped, target)
            loss += ssim_loss
            if idx == 0:
                running_psnrs.append(calculate_psnr(pred_cropped, target))

        running_losses.append(loss.item())
        running_ssim_losses.append(ssim_loss.item())
        del reconstruction_intensities

        loss.backward()
        for p in trainable_params:
            if p.grad is not None:
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        if itr % 100 == 0:
            pbar.set_postfix({
                'Loss': f'{sum(running_losses[-50:]) / min(len(running_losses), 50):.6f}',
                'PSNR': f'{sum(running_psnrs[-50:]) / min(len(running_psnrs), 50):.2f}',
                'SSIM_loss': f'{sum(running_ssim_losses[-50:]) / min(len(running_ssim_losses), 50):.6f}',
                'LR': f'{current_lr:.2e}',
                'G': f'{len(gaussians)}',
            })

        # Visualization
        if args.viz_freq > 0 and itr % args.viz_freq == 0:
            with torch.no_grad():
                reconstruction_intensities, phase_map, amplitude = reconstruct(scene, propagator, img_size)
                for plane_idx in range(min(args_prop.num_planes, len(reconstruction_intensities))):
                    recon = torch.clamp(reconstruction_intensities[plane_idx], min=0.0, max=1.0)
                    recon = odak.learn.tools.crop_center(recon, size=(H, W))
                    odak.learn.tools.save_image(f"{result_dir}/recon_{itr:06d}_{plane_idx}.png", recon, cmin=0., cmax=1.0)
                phase_cropped = odak.learn.tools.crop_center(phase_map.squeeze(0), size=(H, W))
                amp_cropped = odak.learn.tools.crop_center(amplitude.squeeze(0), size=(H, W))
                odak.learn.tools.save_image(f"{result_dir}/phase_{itr:06d}.png", phase_cropped, cmin=0., cmax=2 * odak.pi)
                odak.learn.tools.save_image(f"{result_dir}/amp_{itr:06d}.png", amp_cropped, cmin=0., cmax=1.0)
                visualize_gaussian_positions(gaussians, img_size, itr, result_dir)

        # Evaluation
        if (itr % args.eval_freq == 0 and itr > 0) or (itr == args.num_itrs - 1):
            with torch.no_grad():
                reconstruction_intensities, _, _ = reconstruct(scene, propagator, img_size)
                metrics = evaluate(reconstruction_intensities, targets, H, W, lpips_fn, cvvdp_metric)
                means = {k: sum(v) / len(v) for k, v in metrics.items()}

                print(f"\nEvaluation at iteration {itr}:")
                for plane_idx in range(len(metrics["psnr"])):
                    print(f"Plane {plane_idx}: " + ", ".join(f"{k.upper()}: {metrics[k][plane_idx]:.6f}" for k in metrics))
                print("Mean: " + ", ".join(f"{k.upper()}: {means[k]:.6f}" for k in means))
                print(f"Current LR: {current_lr:.2e}")

                # Save model (best by PSNR of primary plane)
                primary_plane_idx = min(1, args_prop.num_planes - 1)
                primary_psnr = metrics["psnr"][primary_plane_idx] if len(metrics["psnr"]) > primary_plane_idx else metrics["psnr"][0]
                if primary_psnr > best_psnr:
                    best_psnr = primary_psnr
                    gaussians.save_gaussians(os.path.join(checkpoint_dir, f"best_gaussians_2d_{itr}.pth"))
                    print(f"Saved BEST model with PSNR {best_psnr:.3f}")
                else:
                    gaussians.save_gaussians(os.path.join(checkpoint_dir, f"latest_gaussians_2d_{itr}.pth"))

                if itr == args.num_itrs - 1:
                    with open(os.path.join(result_dir, "metrics.json"), "w") as f:
                        json.dump({"iteration": itr, "mean": means, "per_plane": metrics}, f, indent=2)
                sys.stdout.flush()
                del reconstruction_intensities
                gc.collect()
                torch.cuda.empty_cache()

    gaussians.save_gaussians(os.path.join(checkpoint_dir, f"final_gaussians_2d_{args.num_itrs}.pth"))
    print(f"Wall time: {(time.time() - start_time) / 60:.2f} min")
    if device.type == "cuda":
        print(f"Peak GPU memory (allocated): {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB")
    print("[*] Training Completed.")


def get_args():
    parser = argparse.ArgumentParser(description="Complex-Valued Quadratic Phase Gaussian, RGB pipeline")
    # Data / output
    parser.add_argument("--target_image_path", default="./data/flower.png", type=str, help="Target image")
    parser.add_argument("--depth_path", default="./data/flower_depth.png", type=str,
                        help="Depth map in [0,255], pass '' for a single-plane target without depth")
    parser.add_argument("--result_base", default="./results", type=str, help="Base output directory")
    parser.add_argument("--tag", default="", type=str, help="Folder name under result_base (default: <image>_rgb_<primitive>)")
    # Representation
    parser.add_argument("--img_size", nargs=2, default=[640, 480], type=int, help="Target resolution W H")
    parser.add_argument("--compression_ratio", default=0.2, type=float,
                        help="Gaussian number N = H*W/2 * ratio (640x480, 0.2 -> 30720)")
    parser.add_argument("--num_gaussians", default=None, type=int, help="Explicit Gaussian count (overrides compression_ratio)")
    parser.add_argument("--primitive", default="curv", type=str, choices=["flat", "curv"],
                        help="'flat' = flat-phase baseline, 'curv' = CVQPG")
    parser.add_argument("--curv_mode", default="scale_aware", type=str, choices=["scale_aware", "constant"],
                        help="Curvature bound: 'scale_aware' (per-Gaussian Nyquist) or 'constant' (200*tanh)")
    parser.add_argument("--curv_nyquist", default=0.9, type=float,
                        help="Fraction of the per-Gaussian 3-sigma Nyquist limit")
    parser.add_argument("--warmup_iters", default=400, type=int,
                        help="Number of warm-up iterations")
    parser.add_argument("--phase_iso", default=0, type=int,
                        help="Enable isotropic/standard quadratic phase factor for ablation study")
    # Optimization
    parser.add_argument("--num_itrs", default=2001, type=int, help="Number of training iterations")
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate of the Gaussian means")
    parser.add_argument("--eval_freq", default=500, type=int, help="Evaluation and checkpoint interval")
    parser.add_argument("--viz_freq", default=1000, type=int, help="Results saving interval (0 = disabled)")
    parser.add_argument("--seed", default=100, type=int, help="Random seed")
    # Optics
    parser.add_argument("--wavelengths", nargs=3, default=[639e-9, 532e-9, 473e-9], type=float, help="R, G, B wavelengths (m)")
    parser.add_argument("--pixel_pitch", default=3.74e-6, type=float, help="SLM pixel pitch (m)")
    parser.add_argument("--num_planes", default=2, type=int, help="Number of supervised depth planes")
    parser.add_argument("--d_val", default=3e-3, type=float, help="Propagation distance to the center of the volume (m)")
    parser.add_argument("--volume_depth", default=4e-3, type=float, help="Depth span of the multi-plane stack (m)")
    parser.add_argument("--split_ratio", default=1.0, type=float, help="Depth exponent used when splitting into 2 planes")
    parser.add_argument("--pad_size", nargs=2, default=None, type=int,
                        help="Padded size pH pW for propagation")
    parser.add_argument("--aperture_size", default=0, type=int,
                        help="Fourier aperture radius in px (0 = disabled, -1 = sum(img_size)/1.4, >0 = explicit radius in px)")
    # Performance
    parser.add_argument("--tile_size", default=64, type=int, help="Renderer tile size in px")
    parser.add_argument("--gauss_batch", default=1024, type=int, help="Gaussians per renderer batch")
    parser.add_argument("--grad_ckpt", default=0, type=int,
                        help="Enable gradient checkpointing in the renderer for memory savings")
    parser.add_argument("--tf32", default=1, type=int, help="Enable TF32 matmul on Ampere+ GPUs (0 = float32)")
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"], help="Training device")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)

    if args.num_gaussians is None:
        args.num_gaussians = int(((args.img_size[0] * args.img_size[1] * 6) / 12) * args.compression_ratio)
    args_prop = Namespace(
        wavelengths=list(args.wavelengths),
        pixel_pitch=args.pixel_pitch,
        volume_depth=args.volume_depth,
        d_val=args.d_val,
        # Guard band >= 144 px per side, rounded up to a multiple of 64
        pad_size=(list(args.pad_size) if args.pad_size is not None
                  else [((args.img_size[1] + 2 * 144 + 63) // 64) * 64,
                        ((args.img_size[0] + 2 * 144 + 63) // 64) * 64]),
        # aperture_size is a radius on the 2x Fourier grid, 0 disables it by masking the whole grid
        aperture_size=(int(sum(args.img_size) / 1.4) if args.aperture_size < 0
                       else (10 * max(args.img_size) if args.aperture_size == 0
                             else args.aperture_size)),
        num_planes=args.num_planes,
        split_ratio=args.split_ratio,
        curv_nyquist=args.curv_nyquist,
        tile_size=args.tile_size,
        gauss_batch=args.gauss_batch,
        phase_iso=bool(args.phase_iso),
        grad_ckpt=bool(args.grad_ckpt),
    )
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    # Plane distances, centered on d_val and spanning volume_depth
    if args_prop.num_planes > 1:
        args_prop.distances = torch.linspace(-args_prop.volume_depth / 2., args_prop.volume_depth / 2., args_prop.num_planes) + args_prop.d_val
    else:
        args_prop.distances = [args_prop.d_val]

    tag = args.tag or f"{os.path.splitext(os.path.basename(args.target_image_path))[0]}_rgb_{args.primitive}"
    result_dir = os.path.join(args.result_base, tag)
    checkpoint_dir = os.path.join(result_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Result directory: {result_dir}")

    if log_debug:
        sys.stdout = open(os.path.join(result_dir, "log.txt"), "w")

    print(f"Args: {vars(args)}")
    print("Distance: ", args_prop.distances)
    print(f"PadConfig: pad_size={args_prop.pad_size} | aperture_radius={args_prop.aperture_size}")

    propagator = propagator(
        resolution=args_prop.pad_size,
        wavelengths=args_prop.wavelengths,
        pixel_pitch=args_prop.pixel_pitch,
        number_of_frames=3,
        distances=args_prop.distances,
        propagation_type='Bandlimited Angular Spectrum',
        laser_channel_power=torch.eye(3),
        aperture_size=args_prop.aperture_size,
        device=device,
    )
    run_training_2d(args, args_prop, propagator, result_dir, checkpoint_dir, device)

    if log_debug:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
