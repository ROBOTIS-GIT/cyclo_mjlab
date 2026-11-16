#!/usr/bin/env bash

set -euo pipefail

PROJECT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_DIR="${PROJECT_PATH}/docker"
COMPOSE_FILE="${DOCKER_DIR}/compose.yaml"

export HOST_UID="${HOST_UID:-$(id -u)}"
export HOST_GID="${HOST_GID:-$(id -g)}"
export CYCLO_MJLAB_IMAGE="${CYCLO_MJLAB_IMAGE:-cyclo-mjlab:latest}"
export CYCLO_MJLAB_CONTAINER="${CYCLO_MJLAB_CONTAINER:-cyclo_mjlab}"

COMPOSE=()

print_help() {
  cat <<EOF
Usage: ./docker/container.sh <command>

Commands:
  build       Build or update the Docker image using the cache
  rebuild     Pull the base image and rebuild without the Docker cache
  start       Build the image when missing, then start the container
  enter       Open an interactive shell in the running container
  stop        Stop the container
  status      Show the container status
  logs        Follow the container logs
  clean       Remove the container and image (cache volumes are preserved)
  help        Show this help
EOF
}

detect_compose() {
  if docker compose version >/dev/null 2>&1; then
    COMPOSE=(docker compose)
  elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE=(docker-compose)
  else
    echo "[ERROR] Docker Compose is not installed." >&2
    exit 1
  fi
}

check_host() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] Docker is not installed." >&2
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "[ERROR] Cannot connect to the Docker daemon." >&2
    exit 1
  fi
  detect_compose
}

check_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] nvidia-smi was not found. Install the NVIDIA driver." >&2
    exit 1
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] The NVIDIA driver is not available." >&2
    exit 1
  fi
  if ! docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"'; then
    echo "[ERROR] The NVIDIA Container Toolkit runtime is not configured." >&2
    exit 1
  fi
}

initialize_submodules() {
  if ! git -C "${PROJECT_PATH}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "[WARN] Not a Git checkout; skipping submodule initialization."
    return
  fi

  if git -C "${PROJECT_PATH}" submodule status --recursive | grep -q '^-'; then
    echo "[INFO] Initializing Git submodules..."
    git -C "${PROJECT_PATH}" submodule update --init --recursive
  fi
}

prepare_host_directories() {
  mkdir -p "${PROJECT_PATH}/logs"
}

compose() {
  (
    cd "${DOCKER_DIR}"
    "${COMPOSE[@]}" -f "${COMPOSE_FILE}" "$@"
  )
}

build_image() {
  echo "[INFO] Building ${CYCLO_MJLAB_IMAGE}..."
  compose build cyclo_mjlab
}

rebuild_image() {
  echo "[INFO] Rebuilding ${CYCLO_MJLAB_IMAGE} without cache..."
  compose build --pull --no-cache cyclo_mjlab
}

container_is_running() {
  [[ "$(docker inspect -f '{{.State.Running}}' \
    "${CYCLO_MJLAB_CONTAINER}" 2>/dev/null || true)" == "true" ]]
}

verify_environment() {
  echo "[INFO] Verifying the Python and GPU environment..."
  docker exec --user cyclo \
    --workdir /workspace/cyclo_mjlab \
    "${CYCLO_MJLAB_CONTAINER}" \
    python -c \
      'import importlib.metadata as md; import mujoco, torch, warp; assert torch.cuda.is_available(); mjlab_version = md.version("mjlab"); print(f"[INFO] Ready: torch={torch.__version__}, CUDA={torch.version.cuda}, GPU={torch.cuda.get_device_name(0)}, mjlab={mjlab_version}, mujoco={mujoco.__version__}, warp={warp.__version__}")'
}

start_container() {
  initialize_submodules
  prepare_host_directories

  if ! docker image inspect "${CYCLO_MJLAB_IMAGE}" >/dev/null 2>&1; then
    build_image
  fi

  echo "[INFO] Starting ${CYCLO_MJLAB_CONTAINER}..."
  compose up -d --no-build cyclo_mjlab
  verify_environment
  echo "[INFO] Enter with: ./docker/container.sh enter"
}

enter_container() {
  if ! container_is_running; then
    echo "[ERROR] Container is not running. Run './docker/container.sh start' first." >&2
    exit 1
  fi

  exec docker exec -it \
    --user cyclo \
    --workdir /workspace/cyclo_mjlab \
    --env HOME=/home/cyclo \
    --env DISPLAY="${DISPLAY:-}" \
    "${CYCLO_MJLAB_CONTAINER}" \
    bash
}

stop_container() {
  compose stop cyclo_mjlab
}

show_status() {
  compose ps cyclo_mjlab
}

show_logs() {
  compose logs --follow cyclo_mjlab
}

clean_resources() {
  read -r -p \
    "Remove container '${CYCLO_MJLAB_CONTAINER}' and image '${CYCLO_MJLAB_IMAGE}'? [y/N] " \
    reply
  if [[ ! "${reply}" =~ ^[Yy]$ ]]; then
    echo "[INFO] Clean cancelled."
    return
  fi

  compose down --remove-orphans
  docker image rm "${CYCLO_MJLAB_IMAGE}" 2>/dev/null || true
  echo "[INFO] Named pip/Warp cache volumes were preserved."
}

main() {
  local command="${1:-help}"

  case "${command}" in
    help|-h|--help)
      print_help
      return
      ;;
  esac

  check_host

  case "${command}" in
    build)
      check_gpu
      initialize_submodules
      build_image
      ;;
    rebuild)
      check_gpu
      initialize_submodules
      rebuild_image
      ;;
    start)
      check_gpu
      start_container
      ;;
    enter)
      enter_container
      ;;
    stop)
      stop_container
      ;;
    status)
      show_status
      ;;
    logs)
      show_logs
      ;;
    clean)
      clean_resources
      ;;
    *)
      echo "[ERROR] Unknown command: ${command}" >&2
      print_help
      exit 1
      ;;
  esac
}

main "$@"
