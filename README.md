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

This repository contains the PyTorch implementation of **Complex-Valued Quadratic Phase Gaussian (CVQPG)**, a novel hologram representation method that augments each 2D Gaussian primitive with a quadratic phase profile controlled by a learnable curvature parameter. 

The code builds on the work [Complex-Valued 2D Gaussian Representation for Computer-Generated Holography](https://github.com/complight/Complex-Valued_2D_Gaussian_Representation) (ECCV 2026), which is also used as the baseline model for comparison. 

This project depends on the scientific computing toolkit [Odak](https://github.com/kaanaksit/odak) from [Computational Light Laboratory](https://complightlab.com/). 

### Repository structure

```
├── model_2d_gaussian_rgb.py          # Gaussian primitives + tile renderer, RGB
├── model_2d_gaussian_grayscale.py    # Gaussian primitives + tile renderer, grayscale
├── train_2d_gaussian_rgb.py          # training / evaluation, RGB
├── train_2d_gaussian_grayscale.py    # training / evaluation, grayscale
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

`requirements.txt` installs **Odak 0.2.7** from PyPI, which may introduce floating-point noises to the results. 

Install **Odak 0.2.8** to reproduce bit-exactly results from the paper. 

```bash
pip install "odak @ git+https://github.com/kaanaksit/odak@4057dc2bde1c9ad1bce22fc091cf8c2a1a9f010f"
```

## Quick start

```bash
# CVQPG on the default scene (flower)
python train_2d_gaussian_rgb.py
python train_2d_gaussian_grayscale.py

# flat-phase (planar Gaussian) baseline
python train_2d_gaussian_rgb.py --primitive flat
python train_2d_gaussian_grayscale.py --primitive flat

# flat vs CVQPG on one scene, then print a comparison table
bash scripts/run_rgb.sh flower
bash scripts/run_grayscale.sh dragon

# all 10 scenes (RGB, compression ratio 0.2)
bash scripts/run_benchmark.sh rgb 0.2
```

Each run writes to `<result_base>/<tag>/` (default `results/<image>_<rgb|gray>_<primitive>/`). 

| file | content |
|---|---|
| `log.txt` | arguments, plane distances, and per-plane PSNR / SSIM / LPIPS / FLIP / CVVDP at every evaluation |
| `metrics.json` | metrics of the final evaluation |
| `target_<k>.png` | synthesized target of plane k (k = 0, 1, …) |
| `recon_<iter>_<k>.png` | reconstructed intensity of plane k (k = 0, 1, …) |
| `phase_<iter>.png`, `amp_<iter>.png` | phase and amplitude of the complex hologram (one color channel per wavelength for RGB, false color for grayscale) |
| `gaussian_positions_<iter>.png` | Gaussian center positions, colored by curvature |
| `checkpoints/` | Gaussian parameters (.pth) saved at each evaluation. Optimizer state is not saved. |

## Usage

### Dataset

`data/` contains the 10 scenes used in the paper (5.7 MB in total). Each scene consists of a target image and a depth map. 

| scene | image | depth map |
|---|---|---|
| `bento` | `bento.jpg`, 1024×746 | `bento_depth.png`, 1024×746 |
| `burger` | `burger.jpg`, 2048×1366 | `burger_depth.png`, 2560×1536 |
| `dragon` | `dragon.jpg`, 1024×680 | `dragon_depth.png`, 1024×680 |
| `flower` | `flower.png`, 1008×756 | `flower_depth.png`, 1008×756 |
| `pencil` | `pencil.jpg`, 1024×768 | `pencil_depth.png`, 1024×768 |
| `redcar` | `redcar.jpg`, 1024×768 | `redcar_depth.png`, 1024×768 |
| `statue` | `statue.jpg`, 1024×683 | `statue_depth.png`, 1024×683 |
| `straw` | `straw.jpg`, 1024×768 | `straw_depth.png`, 1024×768 |
| `tiger` | `tiger.jpg`, 1024×683 | `tiger_depth.png`, 1024×683 |
| `windmill` | `windmill.jpg`, 1023×670 | `windmill_depth.png`, 1023×670 |

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

### Training options

**Data / output**

| option | default | description |
|---|---|---|
| `--target_image_path` | `./data/flower.png` | Target image |
| `--depth_path` | `./data/flower_depth.png` | Depth map in [0,255], pass `''` for a single-plane target without depth |
| `--result_base` | `./results` | Base output directory |
| `--tag` | `<image>_rgb_<primitive>` / `<image>_gray_<primitive>` | Folder name under `--result_base` |

**Representation**

| option | default | description |
|---|---|---|
| `--img_size W H` | `640 480` | Target resolution |
| `--compression_ratio` | `0.2` | Gaussian number N = W·H/2·ratio (RGB) or W·H/4·ratio (grayscale) |
| `--num_gaussians` | – | Explicit Gaussian count (overrides `--compression_ratio`) |
| `--primitive` | `curv` | `curv` = CVQPG, `flat` = flat-phase baseline |
| `--curv_mode` | `scale_aware` | Curvature bound: `scale_aware` (per-Gaussian Nyquist) or `constant` (200*tanh) |
| `--curv_nyquist` | `0.9` | Fraction η of the per-Gaussian 3σ Nyquist limit |
| `--warmup_iters` | `400` | Number of warm-up iterations |
| `--phase_iso` | `0` | Enable isotropic/standard quadratic phase factor for ablation study |

**Optimization**

| option | default | description |
|---|---|---|
| `--num_itrs` | `2001` | Number of training iterations |
| `--lr` | `0.01` | Learning rate of the Gaussian means |
| `--eval_freq` | `500` | Evaluation and checkpoint interval |
| `--viz_freq` | `1000` | Results saving interval (`0` = disabled) |
| `--seed` | `100` | Random seed |

**Optics**

| option | default | description |
|---|---|---|
| `--wavelengths` (RGB) | `639e-9 532e-9 473e-9` | R, G, B wavelengths (m) |
| `--wavelength` (grayscale) | `639e-9` | Wavelength (m) |
| `--pixel_pitch` | `3.74e-6` | SLM pixel pitch (m) |
| `--num_planes` | `2` | Number of supervised depth planes |
| `--d_val` | `3e-3` | Propagation distance to the center of the volume (m) |
| `--volume_depth` | `4e-3` | Depth span of the multi-plane stack (m) |
| `--split_ratio` | `1.0` | Depth exponent used when splitting into 2 planes |
| `--pad_size pH pW` | auto | Padded size for propagation |
| `--aperture_size` | `0` | Fourier aperture radius in px (`0` = off, `-1` = sum(img_size)/1.4, `>0` = explicit) |

**Performance**

| option | default | description |
|---|---|---|
| `--grad_ckpt` | `0` | Enable gradient checkpointing in the renderer for memory savings |
| `--tile_size` | `64` | Renderer tile size in px |
| `--gauss_batch` | `1024` | Gaussians per renderer batch |
| `--tf32` | `1` | Enable TF32 matmul on Ampere+ GPUs (`0` = float32) |
| `--device` | `cuda` | Training device |

Run `python train_2d_gaussian_rgb.py -h` for the list in the terminal. 

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
