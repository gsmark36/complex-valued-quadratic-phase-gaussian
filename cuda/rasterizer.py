"""Autograd wrapper for the QPF CUDA rasterizer, plus a pure-torch reference.

The extension is compiled on first use (torch.utils.cpp_extension.load, needs nvcc and
ninja) into cuda/build/. Covariance construction and inversion stay in torch; this
Function takes inv_cov and |sx*sy| directly and returns their gradients, so the
(scales, rotation) -> cov -> inv_cov chain is handled by autograd exactly as in the torch
renderer.
"""

import os
import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load


_DIR = os.path.dirname(os.path.abspath(__file__))
_ext = None


def ext():
    """Compile (first call only) and return the extension module"""
    global _ext
    if _ext is None:
        build_dir = os.path.join(_DIR, "build")
        os.makedirs(build_dir, exist_ok=True)
        _ext = load(name="qpf_2d_cuda",
                    sources=[os.path.join(_DIR, f) for f in
                             ("binding.cpp", "forward.cu", "backward.cu", "raster.cu")],
                    extra_cflags=["-O3"],
                    extra_cuda_cflags=["-O3", "--threads=8", "-diag-suppress=20012",
                                       "--expt-relaxed-constexpr"],
                    build_directory=build_dir, verbose=False)
    return _ext


class QPFRender(Function):
    @staticmethod
    def forward(ctx, means_2d, inv_cov, colours, phase, opacities, curvature, area,
                radii, k, dxy2, width, height):
        C = colours.shape[1]
        args = [t.contiguous() for t in (means_2d, inv_cov, colours, phase,
                                         opacities, curvature, area)]
        out_r, out_i, point_list, ranges = ext().qpf_render_forward(
            *args, radii.contiguous().int(), k.contiguous(),
            float(dxy2), int(width), int(height), int(C))
        ctx.save_for_backward(*args, k.contiguous(), point_list, ranges)
        ctx.cfg = (float(dxy2), int(width), int(height), int(C))
        return torch.complex(out_r, out_i)

    @staticmethod
    def backward(ctx, grad_out):
        (means, inv_cov, colours, phase, opac, curv, area, k, point_list, ranges) = ctx.saved_tensors
        dxy2, W, H, C = ctx.cfg
        gr = grad_out.real.contiguous()
        gi = grad_out.imag.contiguous()
        d = ext().qpf_render_backward(gr, gi, means, inv_cov, colours, phase, opac, curv, area,
                                      k, point_list, ranges, dxy2, W, H, C)
        # means, inv_cov, colours, phase, opacities, curvature, area, then non-tensor args
        return d[0], d[1], d[2], d[3], d[4], d[5], d[6], None, None, None, None, None


def qpf_render(means_2d, inv_cov, colours, phase, opacities, curvature, area,
               radii, k, dxy2, width, height):
    """Complex field [C, H, W] of N Gaussians (linear coherent sum, quadratic phase)

    means_2d [N,2] pixel coords, inv_cov [N,3] = (inv00, inv01, inv11), colours/phase [N,C],
    opacities/curvature/area [N] (area = |sx*sy|), radii [N] culling radius in px,
    k [C] wavenumbers, dxy2 = pixel_pitch**2.
    """
    return QPFRender.apply(means_2d, inv_cov, colours, phase, opacities, curvature, area,
                           radii, k, dxy2, width, height)


def qpf_render_reference(means_2d, inv_cov, colours, phase, opacities, curvature, area,
                         k, dxy2, width, height):
    """Pure-torch reference without culling (all Gaussians x all pixels), for testing"""
    dev = means_2d.device
    C = colours.shape[1]
    ys, xs = torch.meshgrid(torch.arange(height, device=dev, dtype=torch.float32),
                            torch.arange(width, device=dev, dtype=torch.float32), indexing="ij")
    pts = torch.stack([xs, ys], -1).reshape(1, -1, 2)                 # [1, P, 2]
    d = means_2d.unsqueeze(1) - pts                                    # [N, P, 2]
    dx, dy = d[..., 0], d[..., 1]
    mahal = (inv_cov[:, 0:1] * dx * dx + 2 * inv_cov[:, 1:2] * dx * dy + inv_cov[:, 2:3] * dy * dy)
    G = torch.exp(torch.clamp(-0.5 * mahal, min=-50.0))
    alpha = opacities.unsqueeze(1) * G                                 # [N, P]
    base = 0.5 * curvature.unsqueeze(1) * mahal * area.unsqueeze(1) * dxy2
    ph = phase.t().unsqueeze(-1) + k.view(C, 1, 1) * base.unsqueeze(0)  # [C, N, P]
    amp = colours.t().unsqueeze(-1) * alpha.unsqueeze(0)                # [C, N, P]
    out = torch.polar(amp, ph).sum(1)                                   # [C, P]
    return out.view(C, height, width)
