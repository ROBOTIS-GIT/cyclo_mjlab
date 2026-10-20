#!/usr/bin/env bash
# Run from any directory; pass extra train.py options to override defaults.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN="${AI_SAPIENS_TRAIN_PYTHON:-/home/robotis-insu/miniconda3/envs/k1_rl_mjlab/bin/python}"
CHECKPOINT="${AI_SAPIENS_TRAIN_CHECKPOINT:-logs/rsl_rl/k1_mimic/2026-10-16_09-29-40_umr_stand_up_ready_handoff/model_29999.pt}"
exec "$PYTHON_BIN" -u scripts/reinforcement_learning/train.py \
  Cyclo-Mimic-K1-Rev1-UMR-StandUp-AiSapiens \
  --checkpoint "$CHECKPOINT" --env.scene.num-envs 512 \
  --agent.max-iterations 10000 --agent.run-name umr_stand_up_ai_sapiens "$@"
