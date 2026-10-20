#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
SOFT_ENTRY_DR_PYTHON="${SOFT_ENTRY_DR_PYTHON:-/home/robotis-insu/miniconda3/envs/k1_rl_mjlab/bin/python}"
SOFT_ENTRY_DR_CHECKPOINT="${SOFT_ENTRY_DR_CHECKPOINT:-logs/rsl_rl/k1_mimic/2026-10-19_15-03-14_umr_stand_up_ai_sapiens_soft_entry/model_9999.pt}"
if [[ ! -f "$SOFT_ENTRY_DR_CHECKPOINT" ]]; then
  echo "Missing SoftEntry checkpoint: $SOFT_ENTRY_DR_CHECKPOINT" >&2
  exit 1
fi
exec "$SOFT_ENTRY_DR_PYTHON" -u scripts/reinforcement_learning/train.py \
  Cyclo-Mimic-K1-Rev1-UMR-StandUp-AiSapiens-SoftEntry-DR \
  --checkpoint "$SOFT_ENTRY_DR_CHECKPOINT" --env.scene.num-envs 512 \
  --agent.max-iterations 10000 --agent.run-name umr_stand_up_ai_sapiens_soft_entry_dr "$@"
