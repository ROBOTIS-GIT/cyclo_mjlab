#!/usr/bin/env bash
set -euo pipefail
cd /workspace/cyclo_mjlab
python -u setup/taebaek_arm_refined_20261020/train_taebaek.py \
  --gpu-ids '[0]' \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 50000 \
  --agent.run-name taebaek_arm_refined_manual \
  --agent.logger tensorboard \
  --agent.resume False \
  --agent.upload-model False
