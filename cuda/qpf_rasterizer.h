#pragma once
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <tuple>

// QPF (quadratic-phase-factor) 2D gaussian rasterizer, the CUDA counterpart of the
// torch tile renderer in model_2d_gaussian_{rgb,grayscale}.py.
//
//  1. LINEAR COHERENT SUM, not alpha blending. Contributions are summed as complex
//     fields with no transmittance and no occlusion ordering, exactly as the torch
//     renderer does and as coherent superposition means physically.
//
//  2. Per-pixel QUADRATIC PHASE. Each gaussian carries a curvature c; the phase at a
//     pixel is phase[g,ch] + k[ch] * 0.5 * c * mahal * area * dxy2. The Mahalanobis
//     distance is already computed for the amplitude, so the footprint-following phase is
//     nearly free. c = 0 gives the flat (planar) Gaussian baseline.
//
// Covariance construction and inversion stay in torch: they are O(N) and keeping them
// out of CUDA makes the (scales, rotation) -> cov -> inv_cov chain bit-identical to the
// reference implementation. This kernel takes inv_cov directly and returns its gradient.

namespace qpf_2d_cuda {

// Scratch buffers (tile counts, prefix sums, sort pairs) are plain torch tensors so the
// caching allocator handles reuse; no hand-rolled chunk allocator.

// forward: complex field [C,H,W] (as real+imag) and the tile binning needed by backward
std::vector<torch::Tensor> qpf_render_forward(
    const torch::Tensor& means_2d,      // [N,2] float32, pixel coords
    const torch::Tensor& inv_cov,       // [N,3] float32, [inv00, inv01, inv11]
    const torch::Tensor& colours,       // [N,C] float32
    const torch::Tensor& phase,         // [N,C] float32
    const torch::Tensor& opacities,     // [N]   float32
    const torch::Tensor& curvature,     // [N]   float32
    const torch::Tensor& area,          // [N]   float32, |sx*sy|
    const torch::Tensor& radii,         // [N]   int32, culling radius in pixels
    const torch::Tensor& k,             // [C]   float32, wavenumber per channel
    float dxy2,
    int width, int height, int num_channels);

std::vector<torch::Tensor> qpf_render_backward(
    const torch::Tensor& grad_real,     // [C,H,W]
    const torch::Tensor& grad_imag,     // [C,H,W]
    const torch::Tensor& means_2d,
    const torch::Tensor& inv_cov,
    const torch::Tensor& colours,
    const torch::Tensor& phase,
    const torch::Tensor& opacities,
    const torch::Tensor& curvature,
    const torch::Tensor& area,
    const torch::Tensor& k,
    const torch::Tensor& point_list,    // [P] uint32, sorted gaussian ids
    const torch::Tensor& ranges,        // [num_tiles,2] uint32
    float dxy2,
    int width, int height, int num_channels);

} // namespace qpf_2d_cuda
