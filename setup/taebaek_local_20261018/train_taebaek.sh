#!/usr/bin/env bash
# Host entry point. Run --check for configuration/CUDA validation without learning.
set -euo pipefail
project_root=/home/hc/robotis/cyclo_mjlab
setup_dir="$project_root/setup/taebaek_local_20261018"
mkdir -p "$setup_dir/warp_cache"
exec docker run --rm --init --name cyclo_taebaek_training \
  --gpus 'device=0' --shm-size=16g \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$project_root,target=/workspace/cyclo_mjlab" \
  --workdir /workspace/cyclo_mjlab \
  --env HOME=/tmp --env MUJOCO_GL=egl \
  --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
  --env WARP_CACHE_PATH=/workspace/cyclo_mjlab/setup/taebaek_local_20261018/warp_cache \
  --env PYTHONPATH=/workspace/cyclo_mjlab \
  --entrypoint python cyclo-mjlab:latest \
  -u setup/taebaek_local_20261018/train_taebaek.py "$@"
