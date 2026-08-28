#!/usr/bin/env bash

set -euo pipefail

PROJECT_PATH="${PROJECT_PATH:-/workspace/cyclo_mjlab}"
CONTAINER_USER="cyclo"
CONTAINER_GROUP="$(id -gn "${CONTAINER_USER}")"
CONTAINER_HOME="/home/${CONTAINER_USER}"

# Named volumes are created as root. Hand their cache directories to the
# unprivileged development user before starting the long-running process.
install -d -o "${CONTAINER_USER}" -g "${CONTAINER_GROUP}" \
  "${CONTAINER_HOME}/.cache/pip" \
  "${CONTAINER_HOME}/.cache/warp"
chown -R "${CONTAINER_USER}:${CONTAINER_GROUP}" "${CONTAINER_HOME}/.cache"

git config --system --replace-all safe.directory "${PROJECT_PATH}"

cd "${PROJECT_PATH}"
exec gosu "${CONTAINER_USER}" "$@"
