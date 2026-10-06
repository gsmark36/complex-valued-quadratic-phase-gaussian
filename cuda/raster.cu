#include "qpf_rasterizer.h"
#include "config.h"
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

namespace qpf_2d_cuda {

void launch_preprocess(int N, const float2*, const int*, uint32_t*, dim3);
void launch_duplicate(int N, const float2*, const int*, const uint32_t*, uint64_t*, uint32_t*, dim3);
void launch_ranges(int P, const uint64_t*, uint2*);
void launch_render(const dim3, const dim3, int, const uint2*, const uint32_t*, const float2*,
                   const float*, const float*, const float*, const float*, const float*,
                   const float*, const float*, float, float*, float*, int, int);
void launch_render_backward(const dim3, const dim3, int, const uint2*, const uint32_t*, const float2*,
                            const float*, const float*, const float*, const float*, const float*,
                            const float*, const float*, float, const float*, const float*,
                            float*, float*, float*, float*, float*, float*, float*, int, int);

static inline void check_inputs(const torch::Tensor& t, const char* name, int64_t n) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32 || t.scalar_type() == torch::kInt32,
                name, " must be float32/int32");
    TORCH_CHECK(t.size(0) == n, name, " has ", t.size(0), " rows, expected ", n);
}

std::vector<torch::Tensor> qpf_render_forward(
    const torch::Tensor& means_2d, const torch::Tensor& inv_cov,
    const torch::Tensor& colours, const torch::Tensor& phase,
    const torch::Tensor& opacities, const torch::Tensor& curvature,
    const torch::Tensor& area, const torch::Tensor& radii, const torch::Tensor& k,
    float dxy2, int width, int height, int num_channels)
{
    const at::cuda::CUDAGuard guard(means_2d.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int N = means_2d.size(0);
    for (auto pr : {std::make_pair(&means_2d,"means_2d"), std::make_pair(&inv_cov,"inv_cov"),
                    std::make_pair(&colours,"colours"), std::make_pair(&phase,"phase"),
                    std::make_pair(&opacities,"opacities"), std::make_pair(&curvature,"curvature"),
                    std::make_pair(&area,"area"), std::make_pair(&radii,"radii")})
        check_inputs(*pr.first, pr.second, N);
    TORCH_CHECK(k.numel() == num_channels, "k must have num_channels entries");

    auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(means_2d.device());
    auto iopt = torch::TensorOptions().dtype(torch::kInt32).device(means_2d.device());

    const dim3 tile_grid((width + BLOCK_X - 1)/BLOCK_X, (height + BLOCK_Y - 1)/BLOCK_Y, 1);
    const dim3 block(BLOCK_X, BLOCK_Y, 1);
    const int num_tiles = tile_grid.x * tile_grid.y;

    auto out_real = torch::zeros({num_channels, height, width}, fopt);
    auto out_imag = torch::zeros({num_channels, height, width}, fopt);
    auto ranges   = torch::zeros({num_tiles, 2}, iopt);

    auto tiles_touched = torch::zeros({N}, iopt);
    auto point_offsets = torch::zeros({N}, iopt);
    launch_preprocess(N, (const float2*)means_2d.data_ptr<float>(), radii.data_ptr<int>(),
                      (uint32_t*)tiles_touched.data_ptr<int>(), tile_grid);

    size_t scan_bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, scan_bytes, (uint32_t*)tiles_touched.data_ptr<int>(),
                                  (uint32_t*)point_offsets.data_ptr<int>(), N, stream);
    auto scan_buf = torch::empty({(int64_t)scan_bytes},
                                 torch::TensorOptions().dtype(torch::kUInt8).device(means_2d.device()));
    cub::DeviceScan::InclusiveSum(scan_buf.data_ptr(), scan_bytes, (uint32_t*)tiles_touched.data_ptr<int>(),
                                  (uint32_t*)point_offsets.data_ptr<int>(), N, stream);

    int total_pairs = point_offsets[N-1].item<int>();
    if (total_pairs <= 0) {
        auto empty_list = torch::zeros({1}, iopt);
        return {out_real, out_imag, empty_list, ranges};
    }

    auto keys_u = torch::empty({total_pairs}, torch::TensorOptions().dtype(torch::kInt64).device(means_2d.device()));
    auto keys_s = torch::empty_like(keys_u);
    auto vals_u = torch::empty({total_pairs}, iopt);
    auto vals_s = torch::empty_like(vals_u);
    launch_duplicate(N, (const float2*)means_2d.data_ptr<float>(), radii.data_ptr<int>(),
                     (const uint32_t*)point_offsets.data_ptr<int>(),
                     (uint64_t*)keys_u.data_ptr<int64_t>(), (uint32_t*)vals_u.data_ptr<int>(), tile_grid);

    size_t sort_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes,
        (uint64_t*)keys_u.data_ptr<int64_t>(), (uint64_t*)keys_s.data_ptr<int64_t>(),
        (uint32_t*)vals_u.data_ptr<int>(), (uint32_t*)vals_s.data_ptr<int>(), total_pairs, 0, 64, stream);
    auto sort_buf = torch::empty({(int64_t)sort_bytes},
                                 torch::TensorOptions().dtype(torch::kUInt8).device(means_2d.device()));
    cub::DeviceRadixSort::SortPairs(sort_buf.data_ptr(), sort_bytes,
        (uint64_t*)keys_u.data_ptr<int64_t>(), (uint64_t*)keys_s.data_ptr<int64_t>(),
        (uint32_t*)vals_u.data_ptr<int>(), (uint32_t*)vals_s.data_ptr<int>(), total_pairs, 0, 64, stream);

    launch_ranges(total_pairs, (const uint64_t*)keys_s.data_ptr<int64_t>(), (uint2*)ranges.data_ptr<int>());

    launch_render(tile_grid, block, num_channels,
                  (const uint2*)ranges.data_ptr<int>(), (const uint32_t*)vals_s.data_ptr<int>(),
                  (const float2*)means_2d.data_ptr<float>(), inv_cov.data_ptr<float>(),
                  colours.data_ptr<float>(), phase.data_ptr<float>(), opacities.data_ptr<float>(),
                  curvature.data_ptr<float>(), area.data_ptr<float>(), k.data_ptr<float>(),
                  dxy2, out_real.data_ptr<float>(), out_imag.data_ptr<float>(), width, height);

    return {out_real, out_imag, vals_s, ranges};
}

std::vector<torch::Tensor> qpf_render_backward(
    const torch::Tensor& grad_real, const torch::Tensor& grad_imag,
    const torch::Tensor& means_2d, const torch::Tensor& inv_cov,
    const torch::Tensor& colours, const torch::Tensor& phase,
    const torch::Tensor& opacities, const torch::Tensor& curvature,
    const torch::Tensor& area, const torch::Tensor& k,
    const torch::Tensor& point_list, const torch::Tensor& ranges,
    float dxy2, int width, int height, int num_channels)
{
    const at::cuda::CUDAGuard guard(means_2d.device());
    const int N = means_2d.size(0);
    auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(means_2d.device());

    auto d_means = torch::zeros({N, 2}, fopt);
    auto d_inv   = torch::zeros({N, 3}, fopt);
    auto d_col   = torch::zeros({N, num_channels}, fopt);
    auto d_ph    = torch::zeros({N, num_channels}, fopt);
    auto d_op    = torch::zeros({N}, fopt);
    auto d_cv    = torch::zeros({N}, fopt);
    auto d_ar    = torch::zeros({N}, fopt);

    const dim3 tile_grid((width + BLOCK_X - 1)/BLOCK_X, (height + BLOCK_Y - 1)/BLOCK_Y, 1);
    const dim3 block(BLOCK_X, BLOCK_Y, 1);

    launch_render_backward(tile_grid, block, num_channels,
        (const uint2*)ranges.data_ptr<int>(), (const uint32_t*)point_list.data_ptr<int>(),
        (const float2*)means_2d.data_ptr<float>(), inv_cov.data_ptr<float>(),
        colours.data_ptr<float>(), phase.data_ptr<float>(), opacities.data_ptr<float>(),
        curvature.data_ptr<float>(), area.data_ptr<float>(), k.data_ptr<float>(),
        dxy2,
        grad_real.contiguous().data_ptr<float>(), grad_imag.contiguous().data_ptr<float>(),
        d_means.data_ptr<float>(), d_inv.data_ptr<float>(), d_col.data_ptr<float>(),
        d_ph.data_ptr<float>(), d_op.data_ptr<float>(), d_cv.data_ptr<float>(), d_ar.data_ptr<float>(),
        width, height);

    return {d_means, d_inv, d_col, d_ph, d_op, d_cv, d_ar};
}

} // namespace qpf_2d_cuda
