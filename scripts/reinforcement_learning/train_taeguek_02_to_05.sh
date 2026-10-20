#!/usr/bin/env bash
# Run inside the cyclo_mjlab container. Each chapter starts a separate policy.
# NUM_ENVS=4096 MAX_ITERATIONS=20000 GPU_ID=0 bash "$0"
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_DIR"

DRY_RUN=0
case "${1:-}" in
  --dry-run) DRY_RUN=1; shift ;;
  --help|-h)
    cat <<'HELP'
Train Taegeuk 02 -> 03 -> 04 -> 05 as four independent policies.
Usage: bash scripts/reinforcement_learning/train_taeguek_02_to_05.sh [--dry-run]
Environment: NUM_ENVS=4096 MAX_ITERATIONS=20000 GPU_ID=0 PYTHON_BIN=python
             RUN_TAG=<timestamp> OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
--dry-run validates all task references/rewards and prints the commands.
Training stops at the first failure. This script does not resume old checkpoints
or transfer weights between chapters. Results are written under logs/rsl_rl/k1_mimic.
HELP
    exit 0 ;;
  "") ;;
  *) echo "Unknown argument: $1 (use --help)" >&2; exit 2 ;;
esac
(( $# == 0 )) || { echo "Unexpected arguments: $*" >&2; exit 2; }

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_$$}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MUJOCO_GL=egl

for name in NUM_ENVS MAX_ITERATIONS OMP_NUM_THREADS MKL_NUM_THREADS; do
  [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || {
    echo "$name must be a positive integer." >&2; exit 2;
  }
done
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "GPU_ID must be a nonnegative integer." >&2; exit 2; }
[[ "$RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "RUN_TAG must use letters, numbers, _ or -." >&2; exit 2; }
command -v "$PYTHON_BIN" >/dev/null

if (( ! DRY_RUN )); then
  mkdir -p logs/sequences
  sequence_log="logs/sequences/taeguek_02_to_05_${RUN_TAG}.log"
  exec > >(tee -a "$sequence_log") 2>&1
  echo "Sequence log: $PROJECT_DIR/$sequence_log"
fi

# Validate every chapter before starting the first long-running job.
"$PYTHON_BIN" - "$GPU_ID" <<'PY'
from pathlib import Path
import sys
import numpy as np
import torch
import source.tasks
from mjlab.tasks.registry import load_env_cfg
from source.tasks.mimic.mdp.pelvis_xy_rewards import reference_pelvis_xy

gpu = int(sys.argv[1])
if not torch.cuda.is_available() or gpu >= torch.cuda.device_count():
    raise RuntimeError(f"CUDA GPU {gpu} is not available")
for chapter in range(2, 6):
    task = f"Cyclo-Mimic-K1-Rev1-Taeguek{chapter:02d}"
    cfg = load_env_cfg(task)
    path = Path(cfg.commands['reference_trajectory'].trajectory_file)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as motion:
        fps = float(np.asarray(motion['fps']).item())
        frames = len(motion['joint_pos'])
    if fps != 50 or frames < 2:
        raise RuntimeError(f"Invalid reference: {path}, fps={fps}, frames={frames}")
    term = cfg.rewards['reference_pelvis_xy']
    if term.func is not reference_pelvis_xy or term.weight != 1.0 or term.params.get('std') != 0.5:
        raise RuntimeError(f"Unexpected XY reward configuration: {task}")
    print(f"[CHECK] {task}: {frames} frames, {fps:g} Hz; pelvis XY weight=1, std=0.5 m; {path}", flush=True)
PY

trap 'echo "Sequence interrupted; remaining chapters will not start." >&2; exit 130' INT
trap 'echo "Sequence terminated; remaining chapters will not start." >&2; exit 143' TERM

for chapter in 02 03 04 05; do
  task="Cyclo-Mimic-K1-Rev1-Taeguek${chapter}"
  run_name="taeguek${chapter}_xy_${RUN_TAG}"
  cmd=("$PYTHON_BIN" -u scripts/reinforcement_learning/train.py "$task"
    --env.scene.num-envs "$NUM_ENVS"
    --agent.max-iterations "$MAX_ITERATIONS"
    --agent.run-name "$run_name"
    --agent.logger tensorboard
    --agent.resume False
    --agent.upload-model False
    --gpu-ids "[$GPU_ID]")
  printf '[%s] %s\n' "$(date -Is)" "$task"
  printf ' %q' "${cmd[@]}"; printf '\n'
  if (( DRY_RUN )); then continue; fi
  status=0
  "${cmd[@]}" || status=$?
  if (( status != 0 )); then
    echo "[$(date -Is)] $task failed (exit $status); stopping sequence." >&2
    exit "$status"
  fi
  echo "[$(date -Is)] Finished $task"
done

if (( DRY_RUN )); then
  echo "Validation complete; training was not started."
else
  echo "[$(date -Is)] Taegeuk 02, 03, 04 and 05 training completed."
fi
