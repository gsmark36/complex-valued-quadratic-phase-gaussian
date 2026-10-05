import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from argparse import Namespace
from torch.utils.checkpoint import checkpoint


class Gaussians2D(torch.nn.Module):
    def __init__(self, num_points: int, img_size: Tuple[int, int], device: str, args_prop: Namespace):
        super(Gaussians2D, self).__init__()

        self.device = device
        self.img_size = img_size
        self.args_prop = args_prop

        data = self._initialize_2d(num_points, img_size)

        self.register_parameter('means_2d', torch.nn.Parameter(data["means_2d"], requires_grad=False))
        self.register_parameter('pre_act_scales', torch.nn.Parameter(data["pre_act_scales"], requires_grad=False))
        self.register_parameter('pre_act_rotation', torch.nn.Parameter(data["pre_act_rotation"], requires_grad=False))
        self.register_parameter('colours', torch.nn.Parameter(data["colours"], requires_grad=False))
        self.register_parameter('pre_act_phase', torch.nn.Parameter(data["pre_act_phase"], requires_grad=False))
        self.register_parameter('pre_act_opacities', torch.nn.Parameter(data["pre_act_opacities"], requires_grad=False))
        self.register_parameter('curvature', torch.nn.Parameter(data["curvature"], requires_grad=False))

        # Curvature bound for activation
        # Ablation: Constant curvature bound vs scale-aware curvature bound
        self.curv_nyquist_frac = getattr(args_prop, 'curv_nyquist', 0.9)
        self.curv_mode = 'scale_aware'   # 'off' | 'scale_aware' | 'constant' (ablation)
        self.curv_const_max = 200.0      # 200*tanh bound used by 'constant' mode

        # Optimization caches
        self._activation_cache = None
        self._cache_dirty = True
        self._covariance_cache = None

        self.to(device)

    def _initialize_2d(self, num_points: int, img_size: Tuple[int, int]):
        W, H = img_size
        data = {}

        # Position initialization with tanh constraint (GaussianImage)
        data["means_2d"] = torch.rand((num_points, 2), dtype=torch.float32)
        data["means_2d"][:, 0] = data["means_2d"][:, 0] * W
        data["means_2d"][:, 1] = data["means_2d"][:, 1] * H

        normalized_means = data["means_2d"].clone()
        normalized_means[:, 0] = (normalized_means[:, 0] / W) * 2 - 1  # Convert to [-1, 1]
        normalized_means[:, 1] = (normalized_means[:, 1] / H) * 2 - 1
        normalized_means = torch.clamp(normalized_means, -0.999, 0.999)
        data["means_2d"] = torch.atanh(normalized_means)

        # Initialize colors
        data["colours"] = torch.rand((num_points, 3), dtype=torch.float32)

        # Initialize scales with minimum size to prevent dot artifacts
        min_init_scale = 1.5  # Minimum initial scale in pixels
        max_init_scale = 5.0  # Maximum initial scale in pixels
        scale_range = max_init_scale - min_init_scale
        data["pre_act_scales"] = torch.log(
            torch.rand((num_points, 2), dtype=torch.float32) * scale_range + min_init_scale
        )

        # Initialize rotation / phase / opacities
        data["pre_act_rotation"] = torch.rand((num_points,), dtype=torch.float32) * 2 * np.pi - np.pi
        data["pre_act_phase"] = torch.zeros((num_points, 3), dtype=torch.float32)
        data["pre_act_opacities"] = torch.zeros((num_points,), dtype=torch.float32) - 0.5  # sigmoid -> ~0.38

        # Initialize curvature at 0 (planar Gaussians)
        data["curvature"] = torch.zeros((num_points,), dtype=torch.float32)

        print(f"Initialized {num_points} 2D Gaussians for image size {img_size}")
        print(f"Initial scale range: [{min_init_scale:.2f}, {max_init_scale:.2f}] pixels")
        return data

    def invalidate_cache(self):
        self._cache_dirty = True
        self._covariance_cache = None

    def apply_activations(self):
        means_tanh = torch.tanh(self.means_2d)  # Constrain to (-1, 1)
        means_2d = torch.zeros_like(means_tanh)
        means_2d[:, 0] = (means_tanh[:, 0] + 1) * 0.5 * self.img_size[0]  # Scale to [0, W]
        means_2d[:, 1] = (means_tanh[:, 1] + 1) * 0.5 * self.img_size[1]  # Scale to [0, H]

        scales = torch.exp(self.pre_act_scales) + 0.1  # Add small constant for stability

        rotation = self.pre_act_rotation
        phase = self.pre_act_phase % (2.0 * np.pi)
        opacities = torch.sigmoid(self.pre_act_opacities)

        # Curvature activation (check paper for details)
        wavelength = float(min(self.args_prop.wavelengths))  # shortest wavelength binds the bound
        dxy = float(self.args_prop.pixel_pitch)
        sigma = scales.max(dim=1).values
        if self.curv_mode == 'off':
            curvature = torch.zeros_like(self.curvature)
        elif self.curv_mode == 'constant':
            curvature = self.curv_const_max * torch.tanh(self.curvature)
        elif self.curv_mode == 'scale_aware':
            c_max = self.curv_nyquist_frac * wavelength / (6.0 * sigma * (dxy ** 2) + 1e-20)
            curvature = c_max * torch.tanh(self.curvature)
        else:
            raise ValueError(f"Unknown curv_mode: {self.curv_mode}")

        self._activation_cache = (scales, rotation, phase, opacities, means_2d, curvature)
        self._cache_dirty = False

        return scales, rotation, phase, opacities, means_2d, curvature

    @staticmethod
    def invert_cov_2D(cov_00, cov_01, cov_11):
        """Optimized 2x2 matrix inversion for covariance matrices"""
        det = cov_00 * cov_11 - cov_01 * cov_01
        det = det.clamp(min=1e-10)  # Numerical stability
        inv_det = 1.0 / det

        inv_00 = cov_11 * inv_det
        inv_01 = -cov_01 * inv_det
        inv_11 = cov_00 * inv_det

        return inv_00, inv_01, inv_11

    def compute_2d_covariance_elements(self, scales, rotation):
        """Optimized 2D covariance computation"""
        if (self._covariance_cache is not None and
            not self._cache_dirty and
            self._covariance_cache['scales'].shape == scales.shape):
            return self._covariance_cache['cov_elements']

        cos_r = torch.cos(rotation)
        sin_r = torch.sin(rotation)

        sx, sy = scales[:, 0], scales[:, 1]
        sx2, sy2 = sx * sx, sy * sy

        cos_r2 = cos_r * cos_r
        sin_r2 = sin_r * sin_r
        cos_sin = cos_r * sin_r

        cov_00 = sx2 * cos_r2 + sy2 * sin_r2 + 0.1  # Add regularization
        cov_01 = (sx2 - sy2) * cos_sin
        cov_11 = sx2 * sin_r2 + sy2 * cos_r2 + 0.1  # Add regularization

        self._covariance_cache = {
            'scales': scales.clone(),
            'cov_elements': (cov_00, cov_01, cov_11)
        }

        return cov_00, cov_01, cov_11

    def save_gaussians(self, save_path: str):
        state_dict = {
            'means_2d': self.means_2d.cpu(),
            'pre_act_scales': self.pre_act_scales.cpu(),
            'pre_act_rotation': self.pre_act_rotation.cpu(),
            'colours': self.colours.cpu(),
            'pre_act_phase': self.pre_act_phase.cpu(),
            'pre_act_opacities': self.pre_act_opacities.cpu(),
            'img_size': self.img_size,
            'curvature': self.curvature.cpu()
        }
        torch.save(state_dict, save_path)
        print(f"2D Gaussians saved to {save_path}")

    def load_gaussians(self, load_path: str):
        """Load saved gaussians and invalidate caches"""
        state_dict = torch.load(load_path, map_location=self.device)
        self.means_2d.data.copy_(state_dict['means_2d'].to(self.device))
        self.pre_act_scales.data.copy_(state_dict['pre_act_scales'].to(self.device))
        self.pre_act_rotation.data.copy_(state_dict['pre_act_rotation'].to(self.device))
        self.colours.data.copy_(state_dict['colours'].to(self.device))
        self.pre_act_phase.data.copy_(state_dict['pre_act_phase'].to(self.device))
        self.pre_act_opacities.data.copy_(state_dict['pre_act_opacities'].to(self.device))
        self.curvature.data.copy_(state_dict['curvature'].to(self.device))
        self.invalidate_cache()
        print(f"2D Gaussians loaded from {load_path}")

    def __len__(self):
        return len(self.means_2d)


def soft_window_pad(hologram: torch.Tensor, pad_size: list) -> torch.Tensor:
    """
    This function implements a Tukey window (cosine-tapered window) to pad a field with a
    smooth transition from the content edge to zero, avoiding sharp edges
    Check Odak 0.2.8 (odak.learn.tools.matrix.smooth_pad) for similar implementation
    """
    C, H, W = hologram.shape
    pH, pW  = pad_size[0], pad_size[1]
    device  = hologram.device

    pad_h   = (pH - H) // 2
    pad_h_b = pH - H - pad_h
    pad_w   = (pW - W) // 2
    pad_w_r = pW - W - pad_w

    def _rep_pad(t: torch.Tensor) -> torch.Tensor:
        return F.pad(
            t.unsqueeze(0),
            (pad_w, pad_w_r, pad_h, pad_h_b),
            mode='replicate',
        ).squeeze(0)

    # 1. Replicate padding with no_grad
    with torch.no_grad():
        rep = torch.complex(_rep_pad(hologram.real), _rep_pad(hologram.imag))

    # 2. Keep gradient connections
    def _zero_pad(t: torch.Tensor) -> torch.Tensor:
        return F.pad(t.unsqueeze(0), (pad_w, pad_w_r, pad_h, pad_h_b)).squeeze(0)

    zero = torch.complex(_zero_pad(hologram.real), _zero_pad(hologram.imag))

    # 3. Zero pad with gradients in the center, replicate padding without gradients in the guard band
    holo_mask = torch.zeros(1, pH, pW, device=device)
    holo_mask[:, pad_h:pad_h + H, pad_w:pad_w + W] = 1.0
    padded = zero + rep * (1.0 - holo_mask)

    # 4. Construct window function
    def _taper_1d(total: int, inner_start: int, inner_size: int) -> torch.Tensor:
        w = torch.zeros(total, device=device, dtype=torch.float32)
        w[inner_start: inner_start + inner_size] = 1.0
        if inner_start > 0:
            t = torch.arange(inner_start, device=device, dtype=torch.float32)
            w[:inner_start] = 0.5 * (1.0 - torch.cos(torch.pi * (t + 1.0) / inner_start))
        right_start = inner_start + inner_size
        n_right = total - right_start
        if n_right > 0:
            t = torch.arange(n_right, device=device, dtype=torch.float32)
            w[right_start:] = 0.5 * (1.0 + torch.cos(torch.pi * (t + 1.0) / n_right))
        return w

    wy = _taper_1d(pH, pad_h, H)
    wx = _taper_1d(pW, pad_w, W)
    window = torch.outer(wy, wx).unsqueeze(0)

    return padded * window


class Scene2D:
    def __init__(self, gaussians: Gaussians2D, args_prop):
        self.gaussians = gaussians
        self.args_prop = args_prop
        self.device = gaussians.device
        self.wavelengths = torch.tensor(args_prop.wavelengths, dtype=torch.float32, device=self.device)
        
        # Renderer batching knobs (larger -> fewer Python-loop iterations, more peak memory)
        self.tile_size = getattr(args_prop, 'tile_size', 64)
        self.gauss_batch = getattr(args_prop, 'gauss_batch', 1024)
        
        # Ablation: isotropic/standard QPF vs Mahalanobis QPF
        self.phase_iso = getattr(args_prop, 'phase_iso', False)
        
        # Gradient checkpointing of the per-tile render body
        self.grad_ckpt = getattr(args_prop, 'grad_ckpt', False)

    def evaluate_gaussians_2d_optimized(self, points_2d, means_2d, inv_00, inv_01, inv_11):
        """
        Vectorized evaluation of 2D Gaussians

        Args:
            points_2d: (1, P, 2) pixel coordinates
            means_2d: (N, 2) gaussian centers
            inv_00, inv_01, inv_11: (N,) inverse covariance elements

        Returns:
            power: (N, P) Gaussian exponent at each pixel
            mahal_dist: (N, P) squared Mahalanobis distance
        """
        means_expanded = means_2d.unsqueeze(1)  # (N, 1, 2)
        diff = points_2d - means_expanded  # (N, P, 2)

        dx = diff[..., 0]  # (N, P)
        dy = diff[..., 1]  # (N, P)

        inv_00_exp = inv_00.view(-1, 1)  # (N, 1)
        inv_01_exp = inv_01.view(-1, 1)  # (N, 1)
        inv_11_exp = inv_11.view(-1, 1)  # (N, 1)

        mahal_dist = (dx * dx * inv_00_exp +
                     2 * dx * dy * inv_01_exp +
                     dy * dy * inv_11_exp)

        power = -0.5 * mahal_dist
        power = torch.clamp(power, min=-50)  # Prevent underflow

        return power, mahal_dist

    def _render_tile_body(self, x0, y0, x1, y1, k, dxy2,
                          means_2d, inv_00, inv_01, inv_11, opacities,
                          complex_features, curvature, scales, radius, tile_diag_half):
        """Complex contribution of one tile, flattened to [C, P]"""
        device = self.device
        C = len(self.wavelengths)

        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0

        # Cull Gaussians whose 3-sigma footprint cannot reach this tile
        dist_sq = (means_2d[:, 0] - cx)**2 + (means_2d[:, 1] - cy)**2
        threshold = radius + tile_diag_half
        mask = dist_sq < (threshold**2)

        # Local pixel grid for the tile
        xs = torch.arange(x0, x1, device=device, dtype=torch.float32)
        ys = torch.arange(y0, y1, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        pts = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)  # [1, P, 2]

        tile_hologram_flat = torch.zeros((C, pts.shape[1]), dtype=torch.complex64, device=device)
        if not bool(mask.any()):
            return tile_hologram_flat

        # Extract active Gaussian parameters for the tile
        m_means = means_2d[mask]
        m_inv_00 = inv_00[mask]
        m_inv_01 = inv_01[mask]
        m_inv_11 = inv_11[mask]
        m_opac = opacities[mask].unsqueeze(1)
        m_complex = complex_features[mask]              # [N_active, C]
        m_curvature = curvature[mask].unsqueeze(1)      # [N_active, 1]
        m_scales = scales[mask]                         # [N_active, 2]

        N_active = m_means.shape[0]
        batch_size = self.gauss_batch

        for g0 in range(0, N_active, batch_size):
            g1 = min(g0 + batch_size, N_active)

            b_means = m_means[g0:g1]
            b_inv_00 = m_inv_00[g0:g1]
            b_inv_01 = m_inv_01[g0:g1]
            b_inv_11 = m_inv_11[g0:g1]
            b_opac = m_opac[g0:g1]
            b_complex = m_complex[g0:g1]            # [B, C]
            b_curvature = m_curvature[g0:g1]        # [B, 1]
            b_scales = m_scales[g0:g1]              # [B, 2]

            # Gaussian & Mahalanobis evaluation
            powers, mahal_dist = self.evaluate_gaussians_2d_optimized(pts, b_means, b_inv_00, b_inv_01, b_inv_11)
            gauss_vals = torch.exp(powers)
            alphas = b_opac * gauss_vals  # [B, P]

            if self.phase_iso:
                # Isotropic/standard QPF
                d = pts - b_means.unsqueeze(1)                                          # [B, P, 2]
                base_phase = 0.5 * b_curvature * (d * d).sum(-1) * dxy2                 # [B, P]
            else:
                # Mahalanobis QPF
                b_area = (b_scales[:, 0] * b_scales[:, 1]).abs().unsqueeze(1)           # [B, 1]
                base_phase = 0.5 * b_curvature * mahal_dist * b_area * dxy2             # [B, P]

            # For each color channel, sum all Gaussians: alpha * exp(i * k_c * phase) * complex color
            modulated = torch.polar(
                alphas.unsqueeze(0).expand(C, -1, -1),
                k.view(C, 1, 1) * base_phase.unsqueeze(0)
            )                                                        # [C, B, P]
            tile_hologram_flat = tile_hologram_flat + torch.einsum("cbp, bc -> cp", modulated, b_complex)

        return tile_hologram_flat

    def render_hologram_direct(self, img_size: Tuple[int, int]):
        """Render the complex hologram with tile-based spatial culling, then pad it for propagation"""
        W, H = img_size
        device = self.device
        C = len(self.wavelengths)
        k = 2 * np.pi / self.wavelengths
        dxy = self.args_prop.pixel_pitch
        dxy2 = dxy ** 2

        colours = self.gaussians.colours
        scales, rotation, phase, opacities, means_2d, curvature = self.gaussians.apply_activations()
        cov_00, cov_01, cov_11 = self.gaussians.compute_2d_covariance_elements(scales, rotation)
        inv_00, inv_01, inv_11 = self.gaussians.invert_cov_2D(cov_00, cov_01, cov_11)

        # Complex hologram initialization
        hologram = torch.zeros((C, H, W), dtype=torch.complex64, device=device)

        # Per-Gaussian complex amplitude
        phase_factor = torch.exp(1j * phase)
        complex_features = (colours * phase_factor).to(torch.complex64)  # [N, C]

        # Culling radius (3-sigma rule)
        mid  = 0.5 * (cov_00 + cov_11)
        disc = torch.sqrt(((cov_00 - cov_11) * 0.5) ** 2 + cov_01 ** 2)
        radius = 3.0 * torch.sqrt(mid + disc)

        tile_size = self.tile_size
        tile_diag_half = math.sqrt((tile_size / 2)**2 + (tile_size / 2)**2)
        use_ckpt = self.grad_ckpt and torch.is_grad_enabled()

        for y0 in range(0, H, tile_size):
            y1 = min(y0 + tile_size, H)
            for x0 in range(0, W, tile_size):
                x1 = min(x0 + tile_size, W)
                args = (x0, y0, x1, y1, k, dxy2, means_2d, inv_00, inv_01, inv_11,
                        opacities, complex_features, curvature, scales, radius, tile_diag_half)
                if use_ckpt:
                    tile_hologram_flat = checkpoint(self._render_tile_body, *args, use_reentrant=False)
                else:
                    tile_hologram_flat = self._render_tile_body(*args)
                hologram[:, y0:y1, x0:x1] += tile_hologram_flat.view(C, y1 - y0, x1 - x0)

        hologram_field = soft_window_pad(hologram, self.args_prop.pad_size)
        return hologram_field

    def render(self, img_size: Tuple[int, int]):
        """Render the padded complex hologram [C, pH, pW]"""
        self.gaussians.invalidate_cache()
        return self.render_hologram_direct(img_size)


def make_trainable_2d(gaussians):
    """Make 2D Gaussian parameters trainable"""
    params = [gaussians.means_2d,
              gaussians.pre_act_scales,
              gaussians.pre_act_rotation,
              gaussians.colours,
              gaussians.pre_act_phase,
              gaussians.pre_act_opacities,
              gaussians.curvature]
    for p in params:
        p.requires_grad_()

    # Hook to invalidate cache when parameters are updated
    def invalidate_hook(grad):
        gaussians.invalidate_cache()
        return grad

    for p in params:
        p.register_hook(invalidate_hook)
