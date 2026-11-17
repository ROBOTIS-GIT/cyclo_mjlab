#!/usr/bin/env bash

set -euo pipefail

PROJECT_PATH="${PROJECT_PATH:-/workspace/cyclo_mjlab}"

# Cache ownership is prepared in the image before switching to the runtime user.
mkdir -p "${HOME}/.cache/pip" "${HOME}/.cache/warp"

cd "${PROJECT_PATH}"
exec "$@"
