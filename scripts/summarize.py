"""Summarize flat-vs-CVQPG runs into a Markdown table.

Usage: python scripts/summarize.py <results_folder>
Expects <results_folder>/<image>_flat/metrics.json and <results_folder>/<image>_cvqpg/metrics.json
(written by the training scripts at the final iteration).
"""
import os
import sys
import json

METRICS = [("psnr", "PSNR↑"), ("ssim", "SSIM↑"), ("lpips", "LPIPS↓"), ("flip", "FLIP↓"), ("cvvdp", "CVVDP↑")]


def load(folder, tag):
    path = os.path.join(folder, tag, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["mean"]


def main(folder):
    names = sorted({d.rsplit("_", 1)[0] for d in os.listdir(folder)
                    if os.path.isdir(os.path.join(folder, d)) and d.rsplit("_", 1)[-1] in ("flat", "cvqpg")})
    rows, sums, n = [], {"flat": {k: 0.0 for k, _ in METRICS}, "cvqpg": {k: 0.0 for k, _ in METRICS}}, 0
    for name in names:
        flat, curv = load(folder, f"{name}_flat"), load(folder, f"{name}_cvqpg")
        if flat is None or curv is None:
            continue
        n += 1
        for label, m in (("flat", flat), ("cvqpg", curv)):
            rows.append(f"| {name} | {label} | " + " | ".join(f"{m[k]:.4f}" for k, _ in METRICS) +
                        (f" | {curv['psnr'] - flat['psnr']:+.3f} |" if label == "cvqpg" else " | |"))
            for k, _ in METRICS:
                sums[label][k] += m[k]

    print(f"## {folder}\n")
    print("| image | primitive | " + " | ".join(h for _, h in METRICS) + " | ΔPSNR |")
    print("|---|---|" + "---|" * (len(METRICS) + 1))
    print("\n".join(rows))
    if n:
        mean = {label: {k: v / n for k, v in s.items()} for label, s in sums.items()}
        for label in ("flat", "cvqpg"):
            print(f"| **mean ({n})** | {label} | " + " | ".join(f"{mean[label][k]:.4f}" for k, _ in METRICS) +
                  (f" | **{mean['cvqpg']['psnr'] - mean['flat']['psnr']:+.3f}** |" if label == "cvqpg" else " | |"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
