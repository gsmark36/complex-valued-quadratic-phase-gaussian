"""Export phase-only holograms (POH) from a finished training run.

The trained complex field is rendered from a checkpoint and folded into a phase-only
pattern with the double-phase amplitude coding (DPAC). The POH is then reconstructed
through a Fourier aperture (the 4f filter that removes the DPAC checkerboard carrier) and
compared with the targets and with the complex-field reconstruction used in training.

    python export_poh.py results/flower_rgb_curv          # result folder
    python export_poh.py flower_rgb_curv                  # tag under ./results
    python export_poh.py results/flower_rgb_curv/checkpoints/best_gaussians_2d_2000.pth

The run settings are read from <result_dir>/log.txt, the outputs go to <result_dir>/poh/.
"""

import os
import re
import ast
import glob
import json
import argparse
import importlib
import numpy as np
import torch
import odak
import lpips
import pycvvdp
from argparse import Namespace
from PIL import Image


REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_run(target, checkpoint):
    """Result folder, checkpoint path and training args of a run"""
    if target.endswith(".pth"):
        if not os.path.isfile(target):
            raise FileNotFoundError(f"Checkpoint not found: {target}")
        ckpt_path = target
        result_dir = os.path.dirname(os.path.dirname(os.path.abspath(target)))
    else:
        candidates = [target, os.path.join("results", target), os.path.join(REPO_DIR, "results", target)]
        result_dir = next((c for c in candidates if os.path.isdir(c)), None)
        if result_dir is None:
            raise FileNotFoundError(f"Result folder not found: {target} (also looked under ./results)")
        ckpts = glob.glob(os.path.join(result_dir, "checkpoints", f"{checkpoint}_gaussians_2d_*.pth"))
        if not ckpts:
            raise FileNotFoundError(f"No {checkpoint}_gaussians_2d_*.pth in {os.path.join(result_dir, 'checkpoints')}")
        # the highest iteration number (the latest best for 'best', the only file for 'final')
        ckpt_path = max(ckpts, key=lambda p: int(re.findall(r"_(\d+)\.pth$", p)[0]))

    log_path = os.path.join(result_dir, "log.txt")
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"log.txt not found in {result_dir} (needed for the training settings)")
    with open(log_path) as f:
        log = f.read()
    args_line = next((l for l in log.splitlines() if l.startswith("Args: ")), None)
    if args_line is None:
        raise ValueError(f"No 'Args: ...' line in {log_path}")
    args = Namespace(**ast.literal_eval(args_line[len("Args: "):]))
    pad_line = re.search(r"PadConfig: pad_size=\[(\d+), (\d+)\]", log)
    logged_pad = [int(pad_line.group(1)), int(pad_line.group(2))] if pad_line else None
    return result_dir, ckpt_path, args, logged_pad


def build_args_prop(args):
    """The training-time optics settings (same formulas as the train scripts)"""
    args_prop = Namespace(
        wavelengths=list(args.wavelengths) if hasattr(args, "wavelengths") else [args.wavelength],
        pixel_pitch=args.pixel_pitch,
        volume_depth=args.volume_depth,
        d_val=args.d_val,
        pad_size=(list(args.pad_size) if args.pad_size is not None
                  else [((args.img_size[1] + 2 * 144 + 63) // 64) * 64,
                        ((args.img_size[0] + 2 * 144 + 63) // 64) * 64]),
        aperture_size=(int(sum(args.img_size) / 1.4) if args.aperture_size < 0
                       else (10 * max(args.img_size) if args.aperture_size == 0
                             else args.aperture_size)),
        num_planes=args.num_planes,
        split_ratio=args.split_ratio,
        curv_nyquist=args.curv_nyquist,
        tile_size=args.tile_size,
        gauss_batch=args.gauss_batch,
        phase_iso=bool(args.phase_iso),
        grad_ckpt=False,
        renderer=getattr(args, "renderer", "torch"),
    )
    if args_prop.num_planes > 1:
        args_prop.distances = torch.linspace(-args_prop.volume_depth / 2., args_prop.volume_depth / 2., args_prop.num_planes) + args_prop.d_val
    else:
        args_prop.distances = [args_prop.d_val]
    return args_prop


def make_propagator(propagator, args_prop, aperture_size, device):
    C = len(args_prop.wavelengths)
    return propagator(
        resolution=args_prop.pad_size,
        wavelengths=args_prop.wavelengths,
        pixel_pitch=args_prop.pixel_pitch,
        number_of_frames=C,
        distances=args_prop.distances,
        propagation_type='Bandlimited Angular Spectrum',
        laser_channel_power=torch.eye(C),
        aperture_size=aperture_size,
        device=device,
    )


def dpac(phase, amplitude):
    """Double-phase amplitude coding: phase +- arccos(A) interleaved on a checkerboard"""
    offset = torch.arccos(amplitude.clamp(0.0, 1.0))
    lo, hi = phase - offset, phase + offset
    poh = torch.empty_like(phase)
    poh[..., 0::2, 0::2] = lo[..., 0::2, 0::2]
    poh[..., 0::2, 1::2] = hi[..., 0::2, 1::2]
    poh[..., 1::2, 0::2] = hi[..., 1::2, 0::2]
    poh[..., 1::2, 1::2] = lo[..., 1::2, 1::2]
    return poh % (2 * np.pi)


def find_file(path):
    """A data file recorded in log.txt, also tried relative to the repository"""
    if not path:
        return None
    for p in (path, os.path.join(REPO_DIR, path), os.path.join(REPO_DIR, "data", os.path.basename(path))):
        if os.path.isfile(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description="Export phase-only holograms (DPAC) from a training run")
    ap.add_argument("run", type=str, help="Result folder, its tag under ./results, or a checkpoint .pth")
    ap.add_argument("--checkpoint", default="final", choices=["final", "best"],
                    help="Checkpoint used when 'run' is a folder or tag")
    ap.add_argument("--aperture_size", default=None, type=int,
                    help="Fourier aperture radius in px for the POH reconstruction (default: min(img_size))")
    ap.add_argument("--renderer", default=None, choices=["torch", "cuda"],
                    help="Renderer (default: the one used in training)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device")
    cli = ap.parse_args()

    result_dir, ckpt_path, args, logged_pad = resolve_run(cli.run, cli.checkpoint)
    pipeline = "rgb" if hasattr(args, "wavelengths") else "grayscale"
    model = importlib.import_module(f"model_2d_gaussian_{pipeline}")
    train = importlib.import_module(f"train_2d_gaussian_{pipeline}")
    from utils import propagator, multiplane_loss

    device = torch.device(cli.device if (cli.device == "cuda" and torch.cuda.is_available()) else "cpu")
    args_prop = build_args_prop(args)
    if cli.renderer is not None:
        args_prop.renderer = cli.renderer
    if args_prop.renderer == "cuda" and device.type != "cuda":
        print("The CUDA renderer needs a CUDA device: using the torch renderer")
        args_prop.renderer = "torch"
    if logged_pad is not None and logged_pad != args_prop.pad_size:
        raise ValueError(f"pad_size {args_prop.pad_size} does not match log.txt ({logged_pad})")

    img_size = tuple(args.img_size)
    W, H = img_size
    aperture_size = cli.aperture_size if cli.aperture_size is not None else min(img_size)
    out_dir = os.path.join(result_dir, "poh")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Run: {result_dir} ({pipeline}, {args.primitive}) | checkpoint: {ckpt_path}")
    print(f"Renderer: {args_prop.renderer} | pad_size: {args_prop.pad_size} | POH aperture radius: {aperture_size}")

    # Trained Gaussians, with the same curvature mode as in training
    num_points = torch.load(ckpt_path, map_location="cpu")["means_2d"].shape[0]
    gaussians = model.Gaussians2D(num_points=num_points, img_size=img_size, device=device, args_prop=args_prop)
    gaussians.load_gaussians(ckpt_path)
    gaussians.curv_mode = 'off' if args.primitive == 'flat' else args.curv_mode
    scene = model.Scene2D(gaussians, args_prop)

    with torch.no_grad():
        field = scene.render(img_size)                                       # [C, pH, pW]
        phase = odak.learn.wave.calculate_phase(field) % (2 * np.pi)
        amplitude = odak.learn.wave.calculate_amplitude(field).clamp(0.0, 1.0)  # same clamp as training
        levels = torch.round(dpac(phase, amplitude) / (2 * np.pi) * 256).to(torch.int64) % 256
    levels_np = levels.cpu().numpy().astype(np.uint8)
    for c in range(levels_np.shape[0]):
        Image.fromarray(levels_np[c]).save(os.path.join(out_dir, f"poh_{c}.png"))

    # Reconstruct the 8-bit POH through the Fourier aperture, and the complex field as in training
    with torch.no_grad():
        poh_phase = levels.float() * (2 * np.pi / 256)
        prop_poh = make_propagator(propagator, args_prop, aperture_size, device)
        recon_poh = torch.sum(prop_poh.reconstruct(poh_phase), dim=0)          # [planes, C, pH, pW]
        prop_train = make_propagator(propagator, args_prop, args_prop.aperture_size, device)
        recon_complex, _, _ = train.reconstruct(scene, prop_train, img_size)
    for name, recon in (("poh", recon_poh), ("complex", recon_complex)):
        for k in range(recon.shape[0]):
            img = odak.learn.tools.crop_center(recon[k].clamp(0.0, 1.0), size=(H, W))
            odak.learn.tools.save_image(os.path.join(out_dir, f"recon_{name}_{k}.png"), img, cmin=0., cmax=1.0)

    summary = {"checkpoint": os.path.abspath(ckpt_path), "pipeline": pipeline, "primitive": args.primitive,
               "renderer": args_prop.renderer, "img_size": list(img_size), "pad_size": args_prop.pad_size,
               "poh_aperture_size": aperture_size, "encoding": "DPAC, 8-bit, gray level g = phase 2*pi*g/256"}

    # Metrics against the multi-plane targets, if the data files can be found
    image_path, depth_path = find_file(args.target_image_path), find_file(args.depth_path)
    if image_path is None or (args.depth_path and depth_path is None):
        print("Target image or depth map not found: metrics skipped")
    else:
        target_image = train.load_target_image(image_path, img_size).to(device)
        depth_image = (train.load_depth_image(depth_path, img_size).to(device) if depth_path
                       else torch.ones((1, H, W), device=device))
        targets, _, _ = multiplane_loss(target_image=target_image, target_depth=depth_image, args_prop=args_prop)
        lpips_fn = lpips.LPIPS(net='vgg', verbose=False).to(device)
        cvvdp_metric = pycvvdp.cvvdp(display_name='standard_4k', heatmap=None, quiet=True)
        with torch.no_grad():
            for name, recon in (("complex", recon_complex), ("poh", recon_poh)):
                metrics = train.evaluate(recon, targets, H, W, lpips_fn, cvvdp_metric)
                summary[name] = {"mean": {k: sum(v) / len(v) for k, v in metrics.items()}, "per_plane": metrics}
                print(f"{name:>7}: " + ", ".join(f"{k.upper()} {v:.4f}" for k, v in summary[name]["mean"].items()))

    with open(os.path.join(out_dir, "poh_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
