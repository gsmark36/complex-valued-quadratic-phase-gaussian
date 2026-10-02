#!/usr/bin/env bash
# Full benchmark of the paper: 10 images x {flat, CVQPG}, for the RGB and/or grayscale pipeline.
# Run from the repository root:
#   bash scripts/run_benchmark.sh [rgb|grayscale|all] [compression_ratio] [extra train args...]
#   bash scripts/run_benchmark.sh rgb 0.2
#   bash scripts/run_benchmark.sh all 0.1 --grad_ckpt 1
# Results: results/benchmark_<pipeline>_<ratio>/<image>_{flat,cvqpg}/ plus a summary table.
# One RGB run at 640x480 / 0.2 takes ~27 min on a TITAN RTX (20 runs per pipeline).
set -e
PY=${PYTHON:-python}
PIPE=${1:-all}
RATIO=${2:-0.2}
shift 2 || shift $#

IMAGES=(pencils redcar flower straw dragon tiger windmill statue burger bento)

run_pipeline() {  # $1 = rgb | grayscale
  local pipe=$1 out=results/benchmark_${1}_${RATIO}
  mkdir -p "$out"
  for name in "${IMAGES[@]}"; do
    img=$(ls data/${name}.jpg data/${name}.png 2>/dev/null | head -n 1)
    for prim in flat curv; do
      tag=${name}_$([ "$prim" = flat ] && echo flat || echo cvqpg)
      if [ -f "$out/$tag/metrics.json" ]; then echo "skip $out/$tag (done)"; continue; fi
      echo ">>> [$(date +%H:%M:%S)] $pipe $tag"
      $PY train_2d_gaussian_${pipe}.py --primitive $prim --compression_ratio "$RATIO" \
          --target_image_path "$img" --depth_path "data/${name}_depth.png" \
          --result_base "$out" --tag "$tag" "$@"
    done
  done
  $PY scripts/summarize.py "$out" | tee "$out/summary.md"
}

case $PIPE in
  rgb|grayscale) run_pipeline "$PIPE" "$@" ;;
  all) run_pipeline rgb "$@"; run_pipeline grayscale "$@" ;;
  *) echo "pipeline must be rgb, grayscale or all"; exit 1 ;;
esac
