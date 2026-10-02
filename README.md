# CVQPG: Complex-Valued Quadratic Phase Gaussians for Computer-Generated Holography

Official PyTorch implementation of **CVQPG**, a hologram representation built from complex-valued 2D Gaussians
that each carry a learnable **quadratic phase** (a per-primitive virtual lens), instead of the flat phase used by
prior Gaussian hologram representations.

At an equal number of primitives, CVQPG improves multi-plane hologram reconstruction over the flat-phase baseline
on every tested scene and metric: **+0.19 dB PSNR (RGB)** and **+0.33 dB PSNR (grayscale)** on average over 10 scenes.

The code is pure PyTorch, with no custom CUDA kernels to compile, and includes both the RGB and the grayscale pipelines.

---

## Method in one paragraph

Following Zhan et al. [1], a full-complex hologram is represented directly on the SLM plane as a sum of anisotropic
2D Gaussians with complex amplitude. CVQPG multiplies each primitive by a quadratic phase that follows its footprint,

$$\phi_i(\mathbf{p}) = k \cdot \tfrac{1}{2}\,\gamma_i\,\det(\mathbf{S}_i)\,(\mathbf{p}-\mathbf{x}_i)^\top\Sigma_i^{-1}(\mathbf{p}-\mathbf{x}_i)\,\Delta^2 ,$$

where $\gamma_i = 1/f_i$ is a learnable curvature (optical power) and $\Delta$ the pixel pitch. Setting $\gamma_i = 0$
recovers the flat primitive exactly. Two additions make the curvature trainable:

* **Scale-adaptive curvature bound.** $\gamma_i$ is bounded per primitive by the SLM's Nyquist limit at its
  $3\sigma$ edge, $\gamma_{\max,i} = \eta\,\lambda / (6\,\sigma_i\,\Delta^2)$, so small, detail-carrying Gaussians
  may bend their wavefront much more than a single global bound would allow, without aliasing.
* **Warm-up.** Curvature is frozen at 0 for the first 400 iterations, so the model first converges as a flat
  representation and then refines curvature as a residual.

The hologram is rendered by a tile-based rasterizer, padded with a soft (tapered) window, and propagated to the
target planes with the band-limited angular spectrum method.

## Repository structure

```
├── model_2d_gaussian_rgb.py          # CVQPG primitives + tile renderer, RGB (3 wavelengths)
├── model_2d_gaussian_grayscale.py    # same, grayscale (1 wavelength)
├── train_2d_gaussian_rgb.py          # training / evaluation, RGB
├── train_2d_gaussian_grayscale.py    # training / evaluation, grayscale
├── utils/
│   ├── data_utils.py                 # SSIM loss, multi-plane targets, seeding
│   ├── propagator.py                 # band-limited ASM propagation, multi-plane defocus loss
│   ├── optimizer.py                  # Adan optimizer
│   └── plot.py                       # visualization helpers
├── scripts/
│   ├── run_rgb.sh                    # one image: flat vs. CVQPG (RGB)
│   ├── run_grayscale.sh              # one image: flat vs. CVQPG (grayscale)
│   ├── run_benchmark.sh              # the paper's 10-scene benchmark
│   └── summarize.py                  # results -> Markdown table
├── data/                             # the 10 benchmark scenes (image + depth map)
└── results/                          # outputs are written here
```

## Installation

Tested with Python 3.11, PyTorch 2.8.0 + CUDA 12.8, on Linux with an NVIDIA TITAN RTX (24 GB).

```bash
conda create -n cvqpg python=3.11 -y      # or: python3 -m venv .venv && source .venv/bin/activate
conda activate cvqpg
# 1) PyTorch for your CUDA version, see https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# 2) everything else
pip install -r requirements.txt
```

On first use, LPIPS downloads the torchvision VGG16 weights (~530 MB), so an internet connection is required.

### Note on the odak version

The propagation utilities depend on [odak](https://github.com/kaanaksit/odak). Two versions are relevant:

| version | how to install | use it when |
|---|---|---|
| **0.2.7** (PyPI, default) | installed by `requirements.txt` | you want a plain `pip install`. Results match the paper to within floating-point noise (≈0.002 dB PSNR after 200 iterations), but not bit for bit. |
| **0.2.8** (GitHub commit `4057dc2`) | `pip install "odak @ git+https://github.com/kaanaksit/odak@4057dc2bde1c9ad1bce22fc091cf8c2a1a9f010f"` (requires `git`) | you want to reproduce the paper's numbers **bit-exactly**. This is the version used for all results in the paper. |

Run the second command after `pip install -r requirements.txt` to replace 0.2.7 with 0.2.8. Both versions run the same
code without changes.

## Quick start

```bash
# CVQPG, RGB, on the default scene (data/flower.png + data/flower_depth.png)
python train_2d_gaussian_rgb.py

# the flat-phase baseline through the identical pipeline
python train_2d_gaussian_rgb.py --primitive flat

# grayscale
python train_2d_gaussian_grayscale.py

# both primitives on one scene, then print a comparison table
bash scripts/run_rgb.sh flower
bash scripts/run_grayscale.sh dragon
```

All scripts are run from the repository root. Each run writes to `results/<tag>/`:

| file | content |
|---|---|
| `log.txt` | configuration and evaluation log (PSNR, SSIM, LPIPS, FLIP, CVVDP per plane) |
| `metrics.json` | final-iteration metrics (mean and per plane) |
| `recon_<iter>_<plane>.png` | reconstructed intensity at each depth plane |
| `target_<plane>.png` | the defocus-aware target of each plane |
| `phase_<iter>.png`, `amp_<iter>.png` | phase and amplitude of the optimized hologram |
| `gaussian_positions_<iter>.png` | Gaussian centers colored by their curvature |
| `checkpoints/*.pth` | Gaussian parameters (best / latest / final) |

### Not enough GPU memory?

The default RGB setting (640×480, 30,720 Gaussians) needs about 19 GB. Add `--grad_ckpt 1` to enable gradient
checkpointing in the renderer. It gives identical results with much less memory, at the cost of a slower backward pass
(see [Runtime](#runtime-and-memory)). Lowering `--compression_ratio` or `--img_size` also helps.

## Main options

Both training scripts share the same options, and the defaults are the paper's configuration.

| option | default | meaning |
|---|---|---|
| `--target_image_path`, `--depth_path` | `data/flower.png`, `data/flower_depth.png` | target and its depth map (`--depth_path ''` + `--num_planes 1` for a flat target) |
| `--primitive` | `curv` | `curv` = CVQPG, `flat` = flat-phase baseline (curvature frozen at 0) |
| `--img_size W H` | `640 480` | target resolution |
| `--compression_ratio` | `0.2` | number of Gaussians: RGB `W·H/2·r` (30,720), grayscale `W·H/4·r` (15,360) |
| `--num_gaussians` | – | explicit Gaussian count (overrides the ratio) |
| `--num_itrs` | `2001` | training iterations |
| `--warmup_iters` | `400` | iterations with curvature frozen before it is refined |
| `--curv_mode` | `scale_aware` | `scale_aware` (ours) or `global` (one constant bound, 200 m⁻¹) |
| `--curv_nyquist` | `0.9` | fraction η of the per-Gaussian Nyquist limit |
| `--phase_iso` | `0` | `1` = isotropic phase ½γr² instead of the footprint-following form (ablation) |
| `--num_planes` | `2` | number of supervised depth planes |
| `--d_val`, `--volume_depth` | `3e-3`, `4e-3` | center distance and depth span of the plane stack (m) |
| `--wavelengths` / `--wavelength` | `639e-9 532e-9 473e-9` / `639e-9` | RGB / grayscale wavelengths (m) |
| `--pixel_pitch` | `3.74e-6` | SLM pixel pitch (m) |
| `--pad_size pH pW` | auto (`768 960`) | propagation canvas; default is a ≥144 px guard band per side, rounded to /64 |
| `--aperture_size` | `0` | Fourier aperture radius; 0 = off |
| `--grad_ckpt` | `0` | gradient checkpointing in the renderer (low memory) |
| `--tile_size`, `--gauss_batch` | `64`, `1024` | renderer tiling (speed/memory trade-off only) |
| `--eval_freq`, `--viz_freq` | `500`, `1000` | evaluation / image-saving interval |
| `--result_base`, `--tag` | `./results`, auto | output folder `result_base/tag` |

Run `python train_2d_gaussian_rgb.py -h` for the full list.

### Using your own images

Put an image and a grayscale depth map of the same aspect ratio into `data/` and pass them with
`--target_image_path` / `--depth_path`. Both are resized to `--img_size`. With 2 planes, depth values below
mid-gray go to the first plane (`d_val − volume_depth/2`) and the rest to the second (`d_val + volume_depth/2`).
Without a depth map, use `--depth_path '' --num_planes 1`.

## Reproducing the paper

```bash
bash scripts/run_benchmark.sh rgb 0.2         # RGB, 30,720 Gaussians
bash scripts/run_benchmark.sh rgb 0.1         # RGB, 15,360 Gaussians
bash scripts/run_benchmark.sh grayscale 0.2   # grayscale, 15,360 Gaussians
bash scripts/run_benchmark.sh grayscale 0.1   # grayscale, 7,680 Gaussians
```

Each call trains flat and CVQPG on the 10 scenes in `data/`, skips runs that already finished, and writes
`results/benchmark_<pipeline>_<ratio>/summary.md`. Expected means over the 10 scenes (640×480, 2 planes, iteration 2000):

| setting | N | primitive | PSNR↑ | SSIM↑ | LPIPS↓ | FLIP↓ | CVVDP↑ |
|---|---|---|---|---|---|---|---|
| RGB · 0.2 | 30,720 | flat | 29.265 | 0.8307 | 0.2895 | 0.1144 | 9.168 |
| | | **CVQPG** | **29.450** | **0.8364** | **0.2828** | **0.1125** | **9.203** |
| RGB · 0.1 | 15,360 | flat | 28.034 | 0.7992 | 0.3292 | 0.1292 | 8.917 |
| | | **CVQPG** | **28.225** | **0.8053** | **0.3217** | **0.1265** | **8.966** |
| Gray · 0.2 | 15,360 | flat | 29.504 | 0.8340 | 0.3813 | 0.0753 | 8.935 |
| | | **CVQPG** | **29.834** | **0.8443** | **0.3694** | **0.0725** | **9.019** |
| Gray · 0.1 | 7,680 | flat | 28.076 | 0.7964 | 0.4204 | 0.0888 | 8.548 |
| | | **CVQPG** | **28.398** | **0.8079** | **0.4069** | **0.0850** | **8.658** |

CVQPG is better than flat on all 10 scenes × 5 metrics in each of the four settings.

Component ablations (RGB · 0.1) can be run with the same scripts by appending options, e.g.
`--warmup_iters 0` (no warm-up), `--curv_mode global` (no scale-adaptive bound) or `--phase_iso 1`
(isotropic phase).

### Runtime and memory

NVIDIA TITAN RTX, 640×480, 2 planes, 2001 iterations, mean over the 10 scenes. The flat baseline takes the same time
within ±3.5%.

| setting | N | wall time | peak GPU memory (allocated) |
|---|---|---|---|
| RGB · 0.2 | 30,720 | 26.7 min | 18.9 GB |
| RGB · 0.1 | 15,360 | 17.1 min | 10.9 GB |
| Gray · 0.2 | 15,360 | 13.5 min | 9.5 GB |
| Gray · 0.1 | 7,680 | 8.7 min | 5.5 GB |

With `--grad_ckpt 1`, peak memory for RGB · 0.2 drops from 18.3 GB to **1.0 GB**, and training is about 20% slower
(1.29 → 1.06 it/s on a TITAN RTX). The results are bit-identical to training without checkpointing.

## Acknowledgements

* The flat-phase complex-valued Gaussian hologram representation that CVQPG builds on is from Zhan et al. [1].
* `utils/propagator.py` is adapted from [odak](https://github.com/kaanaksit/odak) (MPL-2.0), including the
  multi-plane defocus loss of Kavaklı et al., *Realistic Defocus Blur for Multiplane Computer-Generated Holography*.
* `utils/optimizer.py` is the Adan optimizer from [sail-sg/Adan](https://github.com/sail-sg/Adan) (Apache-2.0).
* The position parameterization follows GaussianImage (Zhang et al., 2024).
* Metrics: [LPIPS](https://github.com/richzhang/PerceptualSimilarity), [FLIP](https://github.com/NVlabs/flip),
  [ColorVideoVDP](https://github.com/gfxdisp/ColorVideoVDP).

[1] Zhan et al., *Complex-Valued 2D Gaussian Representation for Computer-Generated Holography*, 2025.

## Citation

```bibtex
@misc{cvqpg,
  title  = {Complex-Valued Quadratic Phase Gaussians for Computer-Generated Holography},
  author = {TODO},
  year   = {2026},
  note   = {TODO: venue}
}
```

## License

TODO
