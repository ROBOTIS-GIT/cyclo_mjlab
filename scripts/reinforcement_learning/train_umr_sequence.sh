#!/usr/bin/env bash
# Train the two UMR motions in order, starting a fresh policy for each task.
# Run inside the container:
#   bash scripts/reinforcement_learning/train_umr_sequence.sh
# Optional overrides: NUM_ENVS=2048 MAX_ITERATIONS=30000 PYTHON_BIN=python

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_DIR"

NUM_ENVS="${NUM_ENVS:-2048}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30000}"
PYTHON_BIN="${PYTHON_BIN:-python}"

for value in "$NUM_ENVS" "$MAX_ITERATIONS"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_ENVS and MAX_ITERATIONS must be positive integers." >&2
    exit 2
  fi
done
command -v "$PYTHON_BIN" >/dev/null

# Check both inputs before spending time training the first policy.
MOTION_DIR="source/assets/motions/K1_rev1"
for motion in \
  "$MOTION_DIR/umr_lie_down/k1_0008_lie_down_locomotion_start_footlock_converted.npz" \
  "$MOTION_DIR/umr_stand_up/k1_0008_stand_up_soft_landing_converted.npz"; do
  if [[ ! -r "$motion" ]]; then
    echo "Missing motion: $motion" >&2
    exit 1
  fi
done

trap 'echo "Training sequence interrupted; remaining tasks will not start." >&2; exit 130' INT
trap 'echo "Training sequence terminated; remaining tasks will not start." >&2; exit 143' TERM

train_task() {
  local task="$1"
  local run_name="$2"
  echo "[$(date -Is)] Starting $task ($MAX_ITERATIONS iterations, $NUM_ENVS environments)"
  local status=0
  "$PYTHON_BIN" -u scripts/reinforcement_learning/train.py "$task" \
    --env.scene.num-envs "$NUM_ENVS" \
    --agent.max-iterations "$MAX_ITERATIONS" \
    --agent.run-name "$run_name" || status=$?
  if (( status != 0 )); then
    echo "[$(date -Is)] $task failed (exit $status); stopping sequence." >&2
    exit "$status"
  fi
  echo "[$(date -Is)] Finished $task"
}

train_task Cyclo-Mimic-K1-Rev1-UMR-LieDown umr_0008_lie_down
train_task Cyclo-Mimic-K1-Rev1-UMR-StandUp-SoftLanding umr_stand_up_soft_landing
echo "[$(date -Is)] Both training runs completed."
