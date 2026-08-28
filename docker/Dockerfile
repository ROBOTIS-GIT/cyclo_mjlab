# syntax=docker/dockerfile:1.7

ARG PYTHON_BASE_IMAGE=python:3.11-slim-bookworm
FROM ${PYTHON_BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG HOST_UID=1000
ARG HOST_GID=1000
ARG PROJECT_PATH=/workspace/cyclo_mjlab

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MUJOCO_GL=egl \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    PROJECT_PATH=${PROJECT_PATH} \
    HOME=/home/cyclo \
    WARP_CACHE_PATH=/home/cyclo/.cache/warp

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      bash-completion \
      build-essential \
      ca-certificates \
      ffmpeg \
      git \
      gosu \
      less \
      libegl1 \
      libgl1 \
      libgles2 \
      libglib2.0-0 \
      libglfw3 \
      libgomp1 \
      libosmesa6 \
      libx11-6 \
      libxcursor1 \
      libxext6 \
      libxi6 \
      libxinerama1 \
      libxrandr2 \
      procps \
      tmux \
      vim \
    && rm -rf /var/lib/apt/lists/*

RUN existing_group="$(getent group "${HOST_GID}" | cut -d: -f1 || true)" \
    && if [[ -z "${existing_group}" ]]; then \
         groupadd --gid "${HOST_GID}" cyclo; \
         existing_group=cyclo; \
       fi \
    && useradd --create-home --uid "${HOST_UID}" --gid "${existing_group}" \
         --shell /bin/bash cyclo \
    && install -d -o cyclo -g "${existing_group}" \
         "${PROJECT_PATH}" \
         /home/cyclo/.cache/pip \
         /home/cyclo/.cache/warp

# Install third-party Python dependencies in a layer that only changes when
# pyproject.toml changes. The source tree is copied and registered as editable below.
WORKDIR /tmp/cyclo_mjlab_deps
COPY pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/pip \
    PIP_CACHE_DIR=/root/.cache/pip \
      python -m pip install --upgrade pip setuptools wheel \
    && PIP_CACHE_DIR=/root/.cache/pip python -m pip install .

WORKDIR ${PROJECT_PATH}
COPY . .
RUN --mount=type=cache,target=/root/.cache/pip \
    PIP_CACHE_DIR=/root/.cache/pip python -m pip install --no-deps -e . \
    && chown -R cyclo:0 "${PROJECT_PATH}" \
    && chmod -R g=u "${PROJECT_PATH}"

COPY docker/entrypoint.sh /usr/local/bin/cyclo-entrypoint
RUN chmod +x /usr/local/bin/cyclo-entrypoint \
    && printf '%s\n' \
      'export PROJECT_PATH=/workspace/cyclo_mjlab' \
      'export MUJOCO_GL=${MUJOCO_GL:-egl}' \
      'cd "${PROJECT_PATH}"' \
      >> /home/cyclo/.bashrc \
    && chown cyclo:0 /home/cyclo/.bashrc

ENTRYPOINT ["/usr/local/bin/cyclo-entrypoint"]
CMD ["sleep", "infinity"]
