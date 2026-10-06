#include "qpf_rasterizer.h"
#include "config.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cmath>

namespace cg = cooperative_groups;

namespace qpf_2d_cuda {

// Gradients of the linear coherent sum.
//
//   out_real[c,p] += A_c * alpha * cos(ph_c),   out_imag[c,p] += A_c * alpha * sin(ph_c)
//   A_c = colours[g,c],  alpha = opac*G,  G = exp(clamp(-0.5*mahal, min=-50))
//   ph_c = phase[g,c] + k_c * base
//   base = 0.5*curv*mahal*area*dxy2        (mahalanobis / footprint-following)
//
// The subtle part: `mahal` feeds BOTH the amplitude (through G) and the phase
// (through base), so dL/dmahal has two terms. Missing the phase term silently
// biases every geometric gradient (means, inv_cov) whenever curvature != 0.

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderBackwardKernel(const uint2* __restrict__ ranges,
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
                     const float* __restrict__ grad_real,
                     const float* __restrict__ grad_imag,
                     float* __restrict__ d_means, float* __restrict__ d_inv_cov,
                     float* __restrict__ d_colours, float* __restrict__ d_phase,
                     float* __restrict__ d_opac, float* __restrict__ d_curv,
                     float* __restrict__ d_area,
                     int W, int H) {
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    const uint32_t tile_x = blockIdx.x, tile_y = blockIdx.y;
    const uint32_t horiz = (W + BLOCK_X - 1) / BLOCK_X;
    const uint32_t tile_id = tile_y * horiz + tile_x;

    const uint32_t px = tile_x * BLOCK_X + threadIdx.x;
    const uint32_t py = tile_y * BLOCK_Y + threadIdx.y;
    const bool inside = (px < W) && (py < H);
    const float2 pixf = { (float)px, (float)py };
    const int pid = inside ? (int)(py * W + px) : 0;

    const uint2 range = ranges[tile_id];
    const int toDo = (int)range.y - (int)range.x;
    if (toDo <= 0) return;

    float kc[CHANNELS], gr[CHANNELS], gi[CHANNELS];
    #pragma unroll
    for (int c = 0; c < CHANNELS; c++) {
        kc[c] = k[c];
        gr[c] = inside ? grad_real[c*H*W + pid] : 0.0f;
        gi[c] = inside ? grad_imag[c*H*W + pid] : 0.0f;
    }

    __shared__ uint32_t s_id[BLOCK_SIZE];
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
        block.sync();
        const int prog = i * BLOCK_SIZE + tr;
        if (range.x + prog < range.y) {
            const uint32_t g = point_list[range.x + prog];
            s_id[tr] = g;
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
        block.sync();

        const int batch = min(BLOCK_SIZE, toDo - i * BLOCK_SIZE);
        for (int j = 0; j < batch; j++) {
            const float2 xy = s_mean[j];
            const float dx = xy.x - pixf.x, dy = xy.y - pixf.y;
            const float i00 = s_inv[j*3+0], i01 = s_inv[j*3+1], i11 = s_inv[j*3+2];
            const float mahal = dx*(i00*dx + i01*dy) + dy*(i01*dx + i11*dy);
            const float raw_power = -0.5f * mahal;
            const float power = fmaxf(raw_power, -50.0f);
            const float G = __expf(power);
            const float opac = s_opac[j];
            const float alpha = opac * G;
            const float curv = s_curv[j], ar = s_area[j];

            const float base = 0.5f * curv * mahal * ar * dxy2;

            float S_alpha = 0.0f;   // dL/dalpha
            float S_base  = 0.0f;   // dL/dbase
            float dcol[CHANNELS], dph[CHANNELS];
            #pragma unroll
            for (int c = 0; c < CHANNELS; c++) {
                float s, co;
                __sincosf(s_ph[c*BLOCK_SIZE+j] + kc[c]*base, &s, &co);
                const float A = s_col[c*BLOCK_SIZE+j];
                const float proj_c = gr[c]*co + gi[c]*s;     // d/d(amplitude)
                const float proj_s = -gr[c]*s  + gi[c]*co;   // d/d(phase)
                dcol[c] = alpha * proj_c;
                dph[c]  = A * alpha * proj_s;
                S_alpha += A * proj_c;
                S_base  += kc[c] * dph[c];
            }

            // amplitude path: mahal -> power -> G -> alpha   (clamp kills it past -50)
            const float dpower_dmahal = (raw_power >= -50.0f) ? -0.5f : 0.0f;
            float dL_dmahal = S_alpha * opac * G * dpower_dmahal;

            const float dcurv = S_base * 0.5f * mahal * ar * dxy2;
            const float darea = S_base * 0.5f * curv * mahal * dxy2;
            // phase path: mahal -> base
            dL_dmahal += S_base * 0.5f * curv * ar * dxy2;
            float dmx = 0.0f, dmy = 0.0f;

            // mahal -> inv_cov and mahal -> means
            const float d_i00 = dL_dmahal * dx * dx;
            const float d_i01 = dL_dmahal * 2.0f * dx * dy;
            const float d_i11 = dL_dmahal * dy * dy;
            dmx += dL_dmahal * 2.0f * (i00*dx + i01*dy);
            dmy += dL_dmahal * 2.0f * (i11*dy + i01*dx);

            const float dopac = S_alpha * G;

            // one atomic per warp instead of one per pixel
            const uint32_t g = s_id[j];
            float r;
            r = cg::reduce(warp, dmx,   cg::plus<float>()); if (warp.thread_rank()==0 && r!=0.f) atomicAdd(&d_means[g*2+0], r);
            r = cg::reduce(warp, dmy,   cg::plus<float>()); if (warp.thread_rank()==0 && r!=0.f) atomicAdd(&d_means[g*2+1], r);
            r = cg::reduce(warp, d_i00, cg::plus<float>()); if (warp.thread_rank()==0 && r!=0.f) atomicAdd(&d_inv_cov[g*3+0], r);
            r = cg::reduce(warp, d_i01, cg::plus<float>()); if (warp.thread_rank()==0 && r!=0.f) atomicAdd(&d_inv_cov[g*3+1], r);
            r = cg::reduce(warp, d_i11, cg::plus<float>()); if (warp.thread_rank()==0 && r!=0.f) atomicAdd(&d_inv_cov[g*3+2], r);
            r = cg::reduce(warp, dopac, cg::plus<float>()); if (warp.thread_rank()==0 && r!=0.f) atomicAdd(&d_opac[g], r);
            r = cg::reduce(warp, dcurv, cg::plus<float>()); if (warp.thread_rank()==0 && r!=0.f) atomicAdd(&d_curv[g], r);
            r = cg::reduce(warp, darea, cg::plus<float>()); if (warp.thread_rank()==0 && r!=0.f) atomicAdd(&d_area[g], r);
            #pragma unroll
            for (int c = 0; c < CHANNELS; c++) {
                r = cg::reduce(warp, dcol[c], cg::plus<float>()); if (warp.thread_rank()==0 && r!=0.f) atomicAdd(&d_colours[g*CHANNELS+c], r);
                r = cg::reduce(warp, dph[c],  cg::plus<float>()); if (warp.thread_rank()==0 && r!=0.f) atomicAdd(&d_phase  [g*CHANNELS+c], r);
            }
        }
    }
}

void launch_render_backward(const dim3 grid, const dim3 block, int channels,
    const uint2* ranges, const uint32_t* point_list, const float2* means, const float* inv_cov,
    const float* colours, const float* phase, const float* opac, const float* curv,
    const float* area, const float* k, float dxy2,
    const float* gr, const float* gi,
    float* d_means, float* d_inv, float* d_col, float* d_ph, float* d_op, float* d_cv, float* d_ar,
    int W, int H) {
    #define LAUNCHB(C) renderBackwardKernel<C><<<grid, block>>>(ranges, point_list, means, inv_cov, \
        colours, phase, opac, curv, area, k, dxy2, gr, gi, d_means, d_inv, d_col, d_ph, d_op, d_cv, d_ar, W, H)
    switch (channels) { case 1: LAUNCHB(1); break; case 3: LAUNCHB(3); break;
        default: TORCH_CHECK(false, "qpf rasterizer supports 1 or 3 channels, got ", channels); }
    #undef LAUNCHB
}

} // namespace qpf_2d_cuda
