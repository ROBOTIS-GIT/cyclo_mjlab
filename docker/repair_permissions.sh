#!/usr/bin/env bash
set -euo pipefail

container="${CYCLO_MJLAB_CONTAINER:-cyclo_mjlab}"
docker exec -i --user 0 --workdir /workspace/cyclo_mjlab "$container" bash -se <<'REPAIR'
set -u
owner="$(id -u cyclo):$(id -g cyclo)"

# New runs need writable directories. Preserve existing logs and checkpoints.
for path in logs wandb; do
  if [[ -d "$path" && ! -L "$path" ]]; then
    find "$path" -xdev -type d -uid 0 -exec chown --no-dereference "$owner" {} +
  fi
done

for path in .blinkmimic-data .supermimic-data /home/cyclo/.cache/{warp,pip,wandb} \
    /home/cyclo/.triton /home/cyclo/.nv/ComputeCache /home/cyclo/.local/share/wandb; do
  if [[ -d "$path" && "$(realpath -s "$path")" == "$(realpath "$path")" ]]; then
    find "$path" -xdev -uid 0 \( -type d -o \( -type f -links 1 \) \) \
      -exec chown --no-dereference "$owner" {} +
  fi
done

for path in /home/cyclo/{.cache,.nv,.local,.local/share}; do
  if [[ -d "$path" && "$path" == "$(realpath "$path")" ]]; then
    chown --no-dereference --from=0 "$owner" "$path"
  fi
done
REPAIR
echo "Permission repair complete for ${container}."
