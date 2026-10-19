#!/usr/bin/env bash
set -euo pipefail
cd /workspace/cyclo_mjlab
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MUJOCO_GL=egl
for chapter in 04 05; do
  if [[ "$chapter" == 04 ]]; then
    source_run=2026-10-17_02-37-59_taeguek04_support_v1_scratch_20261016_support_v1
    source_iteration=29999
  else
    source_run=2026-10-17_11-15-10_taeguek05_support_v1_scratch_20261016_support_v1
    source_iteration=15000
  fi
  python -u scripts/reinforcement_learning/train.py "Cyclo-Mimic-K1-Rev1-Taeguek${chapter}-SupportV2" \
    --env.scene.num-envs 32 --agent.max-iterations 2 \
    --agent.run-name "taeguek${chapter}_support_v2_smoke_20261018" \
    --agent.logger tensorboard --agent.resume False --agent.upload-model False \
    --checkpoint "logs/rsl_rl/k1_mimic/${source_run}/model_${source_iteration}.pt" --gpu-ids '[0]'
done
