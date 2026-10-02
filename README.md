# Hologram Representation via Quadratic Phase Gaussian Splatting

**SIGGRAPH Asia 2026 Technical Communications**

TODO: README.md to be updated with latest information.

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

### Note on the odak version

The propagation utilities depend on [odak](https://github.com/kaanaksit/odak). Two versions are relevant:

| version | how to install | use it when |
|---|---|---|
| **0.2.7** (PyPI, default) | installed by `requirements.txt` | you want a plain `pip install`. Results match the paper to within floating-point noise (≈0.002 dB PSNR after 200 iterations), but not bit for bit. |
| **0.2.8** (GitHub commit `4057dc2`) | `pip install "odak @ git+https://github.com/kaanaksit/odak@4057dc2bde1c9ad1bce22fc091cf8c2a1a9f010f"` (requires `git`) | you want to reproduce the paper's numbers **bit-exactly**. This is the version used for all results in the paper. |

Run the second command after `pip install -r requirements.txt` to replace 0.2.7 with 0.2.8. Both versions run the same code without changes.

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

## Citation

```bibtex
@misc{cvqpg,
  title  = {Complex-Valued Quadratic Phase Gaussians for Computer-Generated Holography},
  author = {TODO},
  year   = {2026},
  note   = {TODO: venue}
}
```
