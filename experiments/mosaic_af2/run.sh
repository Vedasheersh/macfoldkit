#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || ( "$1" != mps && "$1" != cpu ) ]]; then
  echo 'Usage: bash experiments/mosaic_af2/run.sh mps|cpu NEW_OUTPUT [options]' >&2
  exit 2
fi
PLATFORM="$1"
OUTPUT="$2"
shift 2
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL_HOME="${MACFOLDKIT_HOME:-$HOME/Library/Caches/macfoldkit}"
export PYTHONPATH="$MODEL_HOME/sources/mosaic/src"
export JAX_PLATFORMS="$PLATFORM" MLX_ENABLE_TF32=0 JAX_MPS_ASYNC_DISPATCH=0
unset JAX_MPS_LIBRARY_PATH MLX_METAL_GPU_ARCH PYTHONOPTIMIZE
exec "$MODEL_HOME/runtimes/mosaic/bin/python" -u "$HERE/run_design.py" \
  --weights "$MODEL_HOME/weights/colabfold" --platform "$PLATFORM" --output "$OUTPUT" "$@"
