#include "qpf_rasterizer.h"
#include "config.h"
#include <cub/cub.cuh>
#include <cmath>

namespace qpf_2d_cuda {

__device__ inline void getRect(const float2 pos, int radius, uint2& rect_min, uint2& rect_max, dim3 grid) {
    rect_min.x = min(grid.x, max(0, (int)((pos.x - radius) / BLOCK_X)));
    rect_min.y = min(grid.y, max(0, (int)((pos.y - radius) / BLOCK_Y)));
    rect_max.x = min(grid.x, max(0, (int)((pos.x + radius + BLOCK_X - 1) / BLOCK_X)));
    rect_max.y = min(grid.y, max(0, (int)((pos.y + radius + BLOCK_Y - 1) / BLOCK_Y)));
}

__global__ void preprocessKernel(int N, const float2* __restrict__ means_2d,
                                 const int* __restrict__ radii,
                                 uint32_t* __restrict__ tiles_touched, dim3 grid) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    tiles_touched[idx] = 0;
    const int r = radii[idx];
    if (r <= 0) return;
    uint2 lo, hi; getRect(means_2d[idx], r, lo, hi, grid);
    if (hi.x <= lo.x || hi.y <= lo.y) return;
    tiles_touched[idx] = (hi.y - lo.y) * (hi.x - lo.x);
}

__global__ void duplicateWithKeysKernel(int N, const float2* __restrict__ means_2d,
                                        const int* __restrict__ radii,
                                        const uint32_t* __restrict__ point_offsets,
                                        uint64_t* __restrict__ keys, uint32_t* __restrict__ vals,
                                        dim3 grid) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const int r = radii[idx];
    if (r <= 0) return;
    uint32_t off = (idx == 0) ? 0 : point_offsets[idx - 1];
    uint2 lo, hi; getRect(means_2d[idx], r, lo, hi, grid);
    for (uint32_t y = lo.y; y < hi.y; y++)
        for (uint32_t x = lo.x; x < hi.x; x++) {
            // Only the tile id matters: a linear coherent sum is order-independent, so the
            // low 32 bits are free (kept as the gaussian id purely for deterministic order).
            keys[off] = ((uint64_t)(y * grid.x + x) << 32) | (uint64_t)idx;
            vals[off] = idx;
            off++;
        }
}

__global__ void identifyTileRangesKernel(int num_pairs, const uint64_t* __restrict__ keys,
                                         uint2* __restrict__ ranges) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;
    const uint32_t tile = (uint32_t)(keys[idx] >> 32);
    if (idx == 0) ranges[tile].x = 0;
    else {
        const uint32_t prev = (uint32_t)(keys[idx - 1] >> 32);
        if (prev != tile) { ranges[prev].y = idx; ranges[tile].x = idx; }
    }
    if (idx == num_pairs - 1) ranges[tile].y = num_pairs;
}

// the render kernel: linear coherent sum with per-pixel quadratic phase
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderTileKernel(const uint2* __restrict__ ranges,
                 const uint32_t* __restrict__ point_list,
                 const float2* __restrict__ means_2d,
                 const float* __restrict__ inv_cov,
                 const float* __restrict__ colours,
                 const float* __restrict__ phase,
                 const float* __restrict__ opacities,
                 const float* __restrict__ curvature,
                 const float* __restrict__ area,
                 const float* __restrict__ k,
                 float dxy2,
                 float* __restrict__ out_real, float* __restrict__ out_imag,
                 int W, int H) {
    const uint32_t tile_x = blockIdx.x, tile_y = blockIdx.y;
    const uint32_t horiz = (W + BLOCK_X - 1) / BLOCK_X;
    const uint32_t tile_id = tile_y * horiz + tile_x;

    const uint32_t px = tile_x * BLOCK_X + threadIdx.x;
    const uint32_t py = tile_y * BLOCK_Y + threadIdx.y;
    const bool inside = (px < W) && (py < H);
    const float2 pixf = { (float)px, (float)py };

    const uint2 range = ranges[tile_id];
    const int toDo = (int)range.y - (int)range.x;
    if (toDo <= 0) return;

    float kc[CHANNELS];
    #pragma unroll
    for (int c = 0; c < CHANNELS; c++) kc[c] = k[c];

    float acc_r[CHANNELS] = {0.0f};
    float acc_i[CHANNELS] = {0.0f};

    __shared__ float2 s_mean[BLOCK_SIZE];
    __shared__ float  s_inv[BLOCK_SIZE * 3];
    __shared__ float  s_opac[BLOCK_SIZE];
    __shared__ float  s_curv[BLOCK_SIZE];
    __shared__ float  s_area[BLOCK_SIZE];
    __shared__ float  s_col[BLOCK_SIZE * CHANNELS];
    __shared__ float  s_ph[BLOCK_SIZE * CHANNELS];

    const int tr = threadIdx.y * blockDim.x + threadIdx.x;
    const int rounds = (toDo + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int i = 0; i < rounds; i++) {
        __syncthreads();
        const int prog = i * BLOCK_SIZE + tr;
        if (range.x + prog < range.y) {
            const uint32_t g = point_list[range.x + prog];
            s_mean[tr] = means_2d[g];
            s_inv[tr*3+0] = inv_cov[g*3+0];
            s_inv[tr*3+1] = inv_cov[g*3+1];
            s_inv[tr*3+2] = inv_cov[g*3+2];
            s_opac[tr] = opacities[g];
            s_curv[tr] = curvature[g];
            s_area[tr] = area[g];
            #pragma unroll
            for (int c = 0; c < CHANNELS; c++) {
                s_col[c*BLOCK_SIZE+tr] = colours[g*CHANNELS+c];
                s_ph [c*BLOCK_SIZE+tr] = phase  [g*CHANNELS+c];
            }
        }
        __syncthreads();
        if (!inside) continue;

        const int batch = min(BLOCK_SIZE, toDo - i * BLOCK_SIZE);
        for (int j = 0; j < batch; j++) {
            const float2 xy = s_mean[j];
            const float dx = xy.x - pixf.x, dy = xy.y - pixf.y;
            const float i00 = s_inv[j*3+0], i01 = s_inv[j*3+1], i11 = s_inv[j*3+2];
            const float mahal = dx*(i00*dx + i01*dy) + dy*(i01*dx + i11*dy);
            // matches the torch reference: power = clamp(-0.5*mahal, min=-50)
            const float power = fmaxf(-0.5f * mahal, -50.0f);
            const float G = __expf(power);
            const float alpha = s_opac[j] * G;

            const float base = 0.5f * s_curv[j] * mahal * s_area[j] * dxy2;

            #pragma unroll
            for (int c = 0; c < CHANNELS; c++) {
                float s, co;
                __sincosf(s_ph[c*BLOCK_SIZE+j] + kc[c] * base, &s, &co);
                const float amp = s_col[c*BLOCK_SIZE+j] * alpha;
                acc_r[c] += amp * co;
                acc_i[c] += amp * s;
            }
        }
    }

    if (inside) {
        const int pid = py * W + px;
        #pragma unroll
        for (int c = 0; c < CHANNELS; c++) {
            atomicAdd(&out_real[c*H*W + pid], acc_r[c]);
            atomicAdd(&out_imag[c*H*W + pid], acc_i[c]);
        }
    }
}

void launch_render(const dim3 grid, const dim3 block, int channels,
                   const uint2* ranges, const uint32_t* point_list,
                   const float2* means, const float* inv_cov, const float* colours,
                   const float* phase, const float* opac, const float* curv,
                   const float* area, const float* k, float dxy2,
                   float* out_r, float* out_i, int W, int H) {
    #define LAUNCH(C) renderTileKernel<C><<<grid, block>>>(ranges, point_list, means, inv_cov, \
        colours, phase, opac, curv, area, k, dxy2, out_r, out_i, W, H)
    switch (channels) { case 1: LAUNCH(1); break; case 3: LAUNCH(3); break;
        default: TORCH_CHECK(false, "qpf rasterizer supports 1 or 3 channels, got ", channels); }
    #undef LAUNCH
}

void launch_preprocess(int N, const float2* means, const int* radii, uint32_t* tt, dim3 grid) {
    preprocessKernel<<<(N + 255)/256, 256>>>(N, means, radii, tt, grid);
}
void launch_duplicate(int N, const float2* means, const int* radii, const uint32_t* offs,
                      uint64_t* keys, uint32_t* vals, dim3 grid) {
    duplicateWithKeysKernel<<<(N + 255)/256, 256>>>(N, means, radii, offs, keys, vals, grid);
}
void launch_ranges(int P, const uint64_t* keys, uint2* ranges) {
    identifyTileRangesKernel<<<(P + 255)/256, 256>>>(P, keys, ranges);
}

} // namespace qpf_2d_cuda
