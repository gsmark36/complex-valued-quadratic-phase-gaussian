import os
import sys
import random
import contextlib
import numpy as np
import torch
from pytorch_msssim import SSIM


@contextlib.contextmanager
def console_only_print():
    """Temporarily route print() to the console while stdout is redirected to log.txt."""
    original_stdout = sys.stdout
    if original_stdout != sys.__stdout__:
        sys.stdout = sys.__stdout__
    try:
        yield
    finally:
        sys.stdout = original_stdout


def GaussianLoss(pred, target, lambda_ssim=0.025):
    """
    SSIM loss term for RGB reconstructions.

    Args:
        pred: Predicted image [3, H, W] or [B, 3, H, W]
        target: Target image [3, H, W] or [B, 3, H, W]
        lambda_ssim: Weight for the SSIM loss component

    Returns:
        total_loss: Weighted SSIM loss
    """
    # Add batch dimension if not present
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    ssim = SSIM(data_range=1.0, size_average=True, channel=3)
    ssim_loss = (1 - ssim(pred, target)) * 0.2
    return lambda_ssim * ssim_loss


def GaussianLossGrayscale(pred, target, lambda_ssim=0.025):
    """
    SSIM loss term for grayscale reconstructions.

    Args:
        pred: Predicted image [1, H, W] or [B, 1, H, W]
        target: Target image [1, H, W] or [B, 1, H, W]
        lambda_ssim: Weight for the SSIM loss component

    Returns:
        total_loss: Weighted SSIM loss
    """
    # Add batch dimension if not present
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    ssim = SSIM(data_range=1.0, size_average=True, channel=1)
    ssim_loss = (1 - ssim(pred, target)) * 0.2
    return lambda_ssim * ssim_loss


def multiplane_loss(target_image, target_depth, args_prop):
    """Slice the target into depth planes (with defocus blur) and build the matching loss."""
    from .propagator import multiplane_loss_odak
    loss_function = multiplane_loss_odak(
                        target_image = target_image,
                        target_depth = target_depth,
                        target_blur_size = 20,
                        number_of_planes = args_prop.num_planes,
                        blur_ratio = 8,
                        weights = [1.0, 1.0, 1.0, 0.0],
                        scheme = "defocus",
                        reduction = "mean",
                        split_ratio = args_prop.split_ratio,
                        device = target_image.device
    )
    targets, mask, quantized_depth = loss_function.get_targets()
    return targets, loss_function, mask


def count_param(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print(f"Frozen Parameters: {total_params - trainable_params:,} ({(total_params - trainable_params) / 1e6:.2f}M)")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")
