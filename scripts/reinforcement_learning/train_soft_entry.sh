#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
SOFT_ENTRY_PYTHON="${SOFT_ENTRY_PYTHON:-/home/robotis-insu/miniconda3/envs/k1_rl_mjlab/bin/python}"
SOFT_ENTRY_CHECKPOINT="${SOFT_ENTRY_CHECKPOINT:-logs/rsl_rl/k1_mimic/2026-10-19_10-15-20_umr_stand_up_ai_sapiens/model_9999.pt}"
exec "$SOFT_ENTRY_PYTHON" -u scripts/reinforcement_learning/train.py \
  Cyclo-Mimic-K1-Rev1-UMR-StandUp-AiSapiens-SoftEntry \
  --checkpoint "$SOFT_ENTRY_CHECKPOINT" --env.scene.num-envs 512 \
  --agent.max-iterations 10000 --agent.run-name umr_stand_up_ai_sapiens_soft_entry "$@"
