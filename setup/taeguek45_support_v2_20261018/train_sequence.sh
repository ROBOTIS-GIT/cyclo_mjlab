#!/usr/bin/env bash
# Isolated 4 -> 5 fine-tuning; initial V1 models and other training stay intact.
set -euo pipefail
cd /workspace/cyclo_mjlab
setup=setup/taeguek45_support_v2_20261018
exec 9>"$setup/driver.lock"
flock -n 9 || { echo 'SupportV2 driver already running'; exit 3; }
[[ ! -e "$setup/driver.started" ]] || { echo 'Already launched; inspect status before restarting'; exit 4; }
touch "$setup/driver.started"
echo "$$" > "$setup/driver.pid"
trap 'status=$?; echo "$status" > "$setup/driver.exit"' EXIT
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MUJOCO_GL=egl
run_tag=20261018_support_v2
python "$setup/validate_launch.py"
for chapter in 04 05; do
  if [[ "$chapter" == 04 ]]; then
    source_run=2026-10-17_02-37-59_taeguek04_support_v1_scratch_20261016_support_v1
    source_iteration=29999
  else
    source_run=2026-10-17_11-15-10_taeguek05_support_v1_scratch_20261016_support_v1
    source_iteration=15000
  fi
  echo "[$(date -Is)] START chapter=$chapter source=$source_iteration additional_updates=10000"
  python -u scripts/reinforcement_learning/train.py "Cyclo-Mimic-K1-Rev1-Taeguek${chapter}-SupportV2" \
    --env.scene.num-envs 4096 --agent.max-iterations 10000 \
    --agent.run-name "taeguek${chapter}_support_v2_finetune_${run_tag}" \
    --agent.logger tensorboard --agent.resume False --agent.upload-model False \
    --checkpoint "logs/rsl_rl/k1_mimic/${source_run}/model_${source_iteration}.pt" --gpu-ids '[0]'
  echo "[$(date -Is)] FINISHED chapter=$chapter"
  python "$setup/export_final.py" "$chapter"
done
echo "[$(date -Is)] BOTH_CHAPTERS_COMPLETE"
