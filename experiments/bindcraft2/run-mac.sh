#!/usr/bin/env bash
# Run a BindCraft2 driver script on Apple Silicon, bypassing its CUDA-oriented launcher.
#
# BindCraft2 is source-available under a hosting-restricted, non-OSI licence and is NOT
# redistributed here. Point BINDCRAFT2_SOURCE at your own checkout; this script only sets up
# an environment and hands off to a python script you name.
#
# Why these settings, all verified on an M5 Pro / macOS 26.6.2:
#  - jax-mps 0.11.2 and dm-haiku come from MacFoldKit's mosaic runtime. BindCraft2 declares
#    jax>=0.11,<0.12, which that satisfies.
#  - optax and biotite come from BINDCRAFT2_DEPS; matplotlib and its own dependencies from
#    BINDCRAFT2_MAC_DEPS. numpy is deliberately absent from both so the runtime's wins.
#  - MPLBACKEND=Agg because trajectory_output.py imports matplotlib at module level and
#    there is no display.
#  - MLX_ENABLE_TF32=0 and JAX_MPS_ASYNC_DISPATCH=0 match every other MacFoldKit runner.
#    TF32 measured as making no difference; async dispatch likewise.
#  - JAX_MPS_LIBRARY_PATH / MLX_METAL_GPU_ARCH / PYTHONOPTIMIZE are cleared so a stale
#    override cannot select a different backend build.
#
# Importing bindcraft.trajectory this way pulls in NONE of cli.py, design_workers.py,
# selfcheck.py or preflight.py, and issues no nvidia-smi call -- verified, not assumed.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: BINDCRAFT2_SOURCE=/path/to/BindCraft2 bash run-mac.sh SCRIPT.py [args...]" >&2
  exit 2
fi

SCRIPT="$1"; shift

MODEL_HOME="${MACFOLDKIT_HOME:-$HOME/Library/Caches/macfoldkit}"
PLATFORM="${BINDCRAFT2_PLATFORM:-mps}"
: "${BINDCRAFT2_SOURCE:?set BINDCRAFT2_SOURCE to your BindCraft2 checkout}"

for required in \
  "$BINDCRAFT2_SOURCE/bindcraft/trajectory.py" \
  "$MODEL_HOME/runtimes/mosaic/bin/python" \
  "$MODEL_HOME/weights/colabfold/params"
do
  [[ -e "$required" ]] || { echo "missing: $required" >&2; exit 3; }
done

DEPS="${BINDCRAFT2_DEPS:-}"
MAC_DEPS="${BINDCRAFT2_MAC_DEPS:-}"
PATHS="$BINDCRAFT2_SOURCE"
[[ -n "$DEPS" ]] && PATHS="$PATHS:$DEPS"
[[ -n "$MAC_DEPS" ]] && PATHS="$PATHS:$MAC_DEPS"

export PYTHONPATH="$PATHS"
export JAX_PLATFORMS="$PLATFORM"
export MLX_ENABLE_TF32=0
export JAX_MPS_ASYNC_DISPATCH=0
export MPLBACKEND=Agg
unset JAX_MPS_LIBRARY_PATH MLX_METAL_GPU_ARCH PYTHONOPTIMIZE

exec "$MODEL_HOME/runtimes/mosaic/bin/python" -u "$SCRIPT" \
  --data-dir "$MODEL_HOME/weights/colabfold" "$@"
