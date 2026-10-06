<div align="center">

# Hologram Representation via Quadratic Phase Gaussian Splatting

[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3829339.3847858-orange?logo=acm)](https://doi.org/10.1145/3829339.3847858)
[![arXiv](https://img.shields.io/badge/arXiv-2609.11434-b31b1b?logo=arXiv)](https://arxiv.org/abs/2609.11434)
[![GitHub](https://img.shields.io/badge/GitHub-Code-black?logo=github)](https://github.com/gsmark36/complex-valued-quadratic-phase-gaussian)
[![Website](https://img.shields.io/badge/Project%20Website-155ECB)]()

**SIGGRAPH Asia 2026 Technical Communications**

[Haolong Wang](https://scholar.google.com/citations?user=_tV0YKkAAAAJ&hl)<sup>1</sup> &emsp; [Yicheng Zhan](https://albertgary.github.io/)<sup>2</sup> &emsp; [Kaan Akşit](https://www.kaanaksit.com/)<sup>2</sup> &emsp; [Simeng Qiu](https://qsimeng.github.io/)<sup>1</sup>

<sup>1</sup> Swansea University &emsp; <sup>2</sup> University College London (UCL)

<!-- TODO: Update personal website link and project page link -->

</div>

## Overview

This repository contains the implementation of **Complex-Valued Quadratic Phase Gaussian (CVQPG)**, a novel hologram representation method that augments each 2D Gaussian primitive with a quadratic phase profile controlled by a learnable curvature parameter. 

The code builds on the work [Complex-Valued 2D Gaussian Representation for Computer-Generated Holography](https://github.com/complight/Complex-Valued_2D_Gaussian_Representation) (ECCV 2026), which is also used as the baseline model for comparison. 

This project depends on the scientific computing toolkit [Odak](https://github.com/kaanaksit/odak) from [Computational Light Laboratory](https://complightlab.com/). 

### Repository structure

```
├── model_2d_gaussian_rgb.py          # Gaussian primitives + tile renderer, RGB
├── model_2d_gaussian_grayscale.py    # Gaussian primitives + tile renderer, grayscale
├── train_2d_gaussian_rgb.py          # training / evaluation, RGB
├── train_2d_gaussian_grayscale.py    # training / evaluation, grayscale
├── export_poh.py                     # phase-only hologram export
├── cuda/                             # optional CUDA rasterizer
│   ├── rasterizer.py                 # autograd wrapper (compiled on first use)
│   └── *.cu, *.cpp, *.h              # forward / backward kernels
├── utils/
│   ├── data_utils.py                 # SSIM loss, multi-plane targets, seeding
│   ├── propagator.py                 # BLASM propagation, multi-plane defocus loss
│   ├── optimizer.py                  # Adan optimizer
│   └── plot.py                       # visualization helpers
├── scripts/
│   ├── run_rgb.sh                    # one scene: flat vs CVQPG, RGB
│   ├── run_grayscale.sh              # one scene: flat vs CVQPG, grayscale
│   ├── run_benchmark.sh              # all scenes: flat vs CVQPG
│   └── summarize.py                  # conversion: result folders -> Markdown table
├── data/                             # 10 scenes (image + depth map)
├── results/                          # result outputs
└── requirements.txt
```

## Installation

Tested on Linux with Python 3.11, PyTorch 2.8.0 + CUDA 12.8, GPU NVIDIA TITAN RTX (24 GB). 

```bash
conda create -n cvqpg python=3.11 -y
conda activate cvqpg
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # pick your CUDA version
pip install -r requirements.txt
```

### Odak version
The propagation utilities depend on [Odak](https://github.com/kaanaksit/odak). 

`requirements.txt` installs **Odak 0.2.7** from PyPI, which may introduce floating-point noise to the results. 

Install **Odak 0.2.8** to reproduce bit-exact results from the paper. 

```bash
pip install "odak @ git+https://github.com/kaanaksit/odak@4057dc2bde1c9ad1bce22fc091cf8c2a1a9f010f"
```

### CUDA renderer (optional)

`--renderer cuda` replaces the PyTorch tile renderer with the CUDA rasterizer in `cuda/`. It is compiled automatically on first use (cached in `cuda/build/`) and needs the CUDA compiler `nvcc` with the same major version as the CUDA build of PyTorch. 

```bash
conda install -c nvidia cuda-nvcc=12.8
```

The CUDA rasterizer reduces the runtime and GPU memory to **1.6 min** and **1 GB** for the default RGB setting. The results are **not bit-identical** to the PyTorch renderer. 

## Quick start

```bash
# CVQPG on the default scene (flower)
python train_2d_gaussian_rgb.py
python train_2d_gaussian_grayscale.py

# flat-phase (planar Gaussian) baseline
python train_2d_gaussian_rgb.py --primitive flat
python train_2d_gaussian_grayscale.py --primitive flat

# CUDA renderer (optional)
python train_2d_gaussian_rgb.py --renderer cuda

# flat vs CVQPG on one scene, then print a comparison table
bash scripts/run_rgb.sh flower
bash scripts/run_grayscale.sh dragon

# all 10 scenes (RGB, compression ratio 0.2)
bash scripts/run_benchmark.sh rgb 0.2

# export phase-only holograms of a finished run (flower_rgb_curv)
python export_poh.py results/flower_rgb_curv
```

Each run writes to `<result_base>/<tag>/` (default `results/<image>_<rgb|gray>_<primitive>/`). 

| output file | content |
|---|---|
| `log.txt` | arguments, plane distances, and per-plane PSNR / SSIM / LPIPS / FLIP / CVVDP at every evaluation |
| `metrics.json` | metrics of the final evaluation |
| `target_<k>.png` | synthesized target of plane k (k = 0, 1, …) |
| `recon_<iter>_<k>.png` | reconstructed intensity of plane k (k = 0, 1, …) |
| `phase_<iter>.png`, `amp_<iter>.png` | phase and amplitude of the complex hologram (one color channel per wavelength for RGB, false color for grayscale) |
| `gaussian_positions_<iter>.png` | Gaussian center positions, colored by curvature |
| `checkpoints/` | Gaussian parameters (.pth) saved at each evaluation. Optimizer state is not saved. |
| `poh/` | phase-only holograms, written by `export_poh.py` (see [POH export](#poh-export)) |

## Usage

### Dataset

`data/` contains the 10 scenes used in the paper. Each scene consists of a target image and a depth map. 

To use your own data, pass an image with `--target_image_path` and a depth map with `--depth_path`. 

* **Image:** any format Pillow can read (e.g. `.jpg`, `.png`). 
* **Depth map:** an 8-bit grayscale image of the same scene. Darker pixels are closer to the SLM. 
* **Without depth:** use `--depth_path '' --num_planes 1` for single plane with default depth (all ones). 
* **Resizing:** both files are resized to `--img_size` without cropping. 
* **Scripts:** scenes are looked up by name as `data/<name>.jpg` or `data/<name>.png`, and `data/<name>_depth.png`. 

### Default configuration

| setting | RGB | grayscale |
|---|---|---|
| resolution | 640×480 | 640×480 |
| Gaussians (compression ratio 20%) | 30,720 | 15,360 |
| wavelengths | 639 / 532 / 473 nm | 639 nm |
| pixel pitch | 3.74 µm | 3.74 µm |
| planes | 1 mm and 5 mm | 1 mm and 5 mm |
| propagation canvas | 768×960, no Fourier aperture | 768×960, no Fourier aperture |
| iterations / warm-up | 2001 / 400 | 2001 / 400 |
| curvature bound | scale-aware, η = 0.9 | scale-aware, η = 0.9 |

Run `python train_2d_gaussian_rgb.py -h` for the list of training options. 

### POH export

`export_poh.py` converts the complex hologram of a finished run into the phase-only hologram (POH) with double-phase amplitude coding (DPAC). No retraining is needed. RGB and grayscale runs are detected automatically. 

The result folder must contain `log.txt` (for training settings) and `checkpoints/`. The target image and depth map recorded in `log.txt` are optional (only used for metrics). 

```bash
python export_poh.py results/flower_rgb_curv                     # result folder
python export_poh.py flower_rgb_curv                             # tag under ./results
python export_poh.py results/flower_rgb_curv/checkpoints/best_gaussians_2d_2000.pth   # checkpoint file
```

| output file | content |
|---|---|
| `poh_<c>.png` | 8-bit padded POH of wavelength c (c = 0, 1, 2 for RGB, 0 for grayscale) |
| `recon_poh_<k>.png` | reconstruction of plane k from the 8-bit POH, through the Fourier aperture |
| `recon_complex_<k>.png` | reconstruction of plane k from the complex hologram |
| `poh_metrics.json` | export settings, and PSNR / SSIM / LPIPS / FLIP / CVVDP of both reconstructions |

Run `python export_poh.py -h` for export options. 

## Citation

```bibtex
@inproceedings{wang2026hologram,
  author = {Wang, Haolong and Zhan, Yicheng and Ak{\c{s}}it, Kaan and Qiu, Simeng},
  title = {Hologram Representation via Quadratic Phase Gaussian Splatting},
  booktitle = {SIGGRAPH Asia 2026 Technical Communications (SA Technical Communications '26)},
  year = {2026},
  month = {December 01--04},
  publisher = {Association for Computing Machinery},
  location = {Kuala Lumpur, Malaysia},
  pages = {4},
  isbn = {979-8-4007-2841-9/2026/12},
  doi = {10.1145/3829339.3847858},
  url = {https://arxiv.org/abs/2609.11434}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details. 
