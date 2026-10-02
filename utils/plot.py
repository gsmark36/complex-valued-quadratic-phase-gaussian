import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def save_with_colormap(tensor, path, vmin, vmax, cmap='viridis'):
    """Save a single-channel tensor as a false-color PNG."""
    arr = tensor.squeeze().detach().cpu().float().numpy()
    arr = np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0., 1.)
    colored = (matplotlib.colormaps[cmap](arr)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(colored).save(path)


def visualize_gaussian_positions(gaussians, img_size, itr, result_dir):
    """
    Scatter the Gaussian centers on the image plane, colored by their (activated) curvature.
    Flat primitives show up as curvature 0 everywhere.
    """
    with torch.no_grad():
        scales, _, _, opacities, means_2d, curvature = gaussians.apply_activations()
        gaussians.invalidate_cache()
        means = means_2d.detach().cpu().numpy()
        curv = curvature.detach().cpu().numpy()

        W, H = img_size
        vmax = max(float(abs(curv).max()), 1e-12)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8 * H / W))
        sc = ax.scatter(means[:, 0], means[:, 1], c=curv, s=1, cmap='coolwarm',
                        vmin=-vmax, vmax=vmax, rasterized=True)
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_title(f'Gaussian centers (N={len(means)}), iteration {itr}')
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='curvature (1/m)')
        plt.tight_layout()
        plt.savefig(f"{result_dir}/gaussian_positions_{itr:06d}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
