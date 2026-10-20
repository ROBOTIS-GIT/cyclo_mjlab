#!/usr/bin/env bash
# Run on robotis-insu@192.168.60.194 (host terminal).
set -euo pipefail
container=cyclo_mjlab_taeguek05_v4
container_root=/workspace/cyclo_mjlab
setup_rel=setup/taeguek07_soft_contact_v1_20261020
host_root=/home/robotis-insu/cyclo_mjlab/work/taeguek05_arm_v4_20261019

available_bytes=$(df -PB1 "$host_root" | awk 'NR==2 {print $4}')
if (( available_bytes < 1073741824 )); then
  echo 'Available disk space is below 1 GiB. Free space before training; no files were removed.' >&2
  exit 1
fi
docker start "$container" >/dev/null
exec docker exec -it --user 1000:1000 \
  -e HOME=/tmp \
  -e MPLCONFIGDIR=/tmp/t7-soft-mpl \
  -e WARP_CACHE_PATH="$container_root/$setup_rel/warp_cache" \
  -w "$container_root" "$container" \
  python -u "$setup_rel/launch.py" 0 --num-envs 4096 "$@"
