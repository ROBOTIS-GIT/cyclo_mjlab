#!/usr/bin/env bash
set -uo pipefail
cd /workspace/cyclo_mjlab
state=setup/taeguek_support_v1_20261016
mkdir "$state/launch_lock" || exit 77
printf '%s\n' "$$" > "$state/driver.pid"
export NUM_ENVS=4096 MAX_ITERATIONS=30000 GPU_ID=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export RUN_TAG=20261016_support_v1
bash scripts/reinforcement_learning/train_taeguek_02_to_05_support.sh
status=$?
printf '%s\n' "$status" > "$state/driver.exit"
exit "$status"
