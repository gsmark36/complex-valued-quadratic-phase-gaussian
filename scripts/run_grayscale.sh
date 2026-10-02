#!/usr/bin/env bash
# Grayscale pipeline on one image: flat baseline vs. CVQPG (ours), identical settings.
# Run from the repository root:
#   bash scripts/run_grayscale.sh [image_name] [extra train args...]
#   bash scripts/run_grayscale.sh flower --grad_ckpt 1
# Results: results/grayscale/<image_name>_{flat,cvqpg}/
set -e
PY=${PYTHON:-python}
NAME=${1:-flower}
shift || true

IMG=$(ls data/${NAME}.jpg data/${NAME}.png 2>/dev/null | head -n 1)
DEP=data/${NAME}_depth.png
[ -n "$IMG" ] || { echo "No data/${NAME}.jpg or data/${NAME}.png"; exit 1; }

for prim in flat curv; do
  tag=${NAME}_$([ "$prim" = flat ] && echo flat || echo cvqpg)
  echo ">>> $tag"
  $PY train_2d_gaussian_grayscale.py --primitive $prim --target_image_path "$IMG" --depth_path "$DEP" \
      --result_base results/grayscale --tag "$tag" "$@"
done
$PY scripts/summarize.py results/grayscale
