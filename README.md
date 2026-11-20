# cyclo_mjlab

[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.11.0-silver.svg)](https://mujoco.readthedocs.io/en/3.11.0/)
[![MJLab](https://img.shields.io/badge/MJLab-1.6.1-silver.svg)](https://github.com/mujocolab/mjlab)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://docs.python.org/3/whatsnew/3.12.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Overview

`cyclo_mjlab` is a research-oriented repository built on
[MJLab](https://github.com/mujocolab/mjlab) and
[MuJoCo](https://mujoco.org/). It provides reinforcement learning environments,
task configurations, and motion-processing tools for developing locomotion and
motion-imitation policies for the ROBOTIS K1 humanoid robot.

The repository currently provides:

- **Velocity**: flat-ground velocity-tracking locomotion
- **Mimic (BeyondMimic/PPO)**: Dance1 and Dance2 reference-motion tracking
- Automatic generation of `params/sim2real.yaml` when Velocity or Mimic training starts
- Automatic export of `exported/policy.onnx` and `exported/policy.pt` during Velocity or Mimic play
- **SuperMimic**: differentiable world-model motion tracking with native MuJoCo/mjlab tasks
- **BlinkMimic**: isolated released recipe with a K1 Rev.1 exact-state reference adapter;
  see [recipe, commands and validation scope](docs/blinkmimic.md)

> [!IMPORTANT]
> This repository currently uses MJLab 1.6.1, MuJoCo 3.11.0, and Python 3.12.
> See the [SuperMimic guide](docs/supermimic.md) for its algorithm-specific setup.

### Locomotion

![Locomotion demo](docs/videos/k1_rl_locomotion_mjlab.webp)

### Motion imitation

![Motion-imitation demo](docs/videos/k1_mimic_dance_mjlab.webp)

## Installation (Docker)

Docker provides a consistent environment with the required Python packages,
MuJoCo libraries, and GPU configuration pre-installed. Conda and a host Python
environment are not required.

### Prerequisites

- Docker Engine with Docker Compose
- NVIDIA GPU with an appropriate NVIDIA driver
- NVIDIA Container Toolkit
- Git and access to this repository

### Steps

1. Clone the repository with its submodules:

   ```bash
   git clone --recurse-submodules git@github.com:ROBOTIS-GIT/cyclo_mjlab.git
   cd cyclo_mjlab
   ```

   If the repository was cloned without submodules, initialize them separately:

   ```bash
   git submodule update --init --recursive
   ```

2. Build the Docker image when needed and start the container:

   ```bash
   ./docker/container.sh start
   ```

3. Enter the running container:

   ```bash
   ./docker/container.sh enter
   ```

After entering the container, the Velocity and Mimic commands below can be
run directly from `/workspace/cyclo_mjlab`.

### Docker commands

| Command | Description |
| --- | --- |
| `./docker/container.sh build` | Build or update the image using the build cache |
| `./docker/container.sh start` | Build if missing, check image compatibility, and start and verify the container |
| `./docker/container.sh recreate` | Check the current image, recreate the container, and verify its runtime and CUDA |
| `./docker/container.sh enter` | Open an interactive shell in the running container |
| `./docker/container.sh stop` | Stop the container |
| `./docker/container.sh logs` | Follow the container logs |
| `./docker/container.sh clean` | Remove the container and image while preserving cache volumes |

The Docker image includes:

- Python 3.12
- MJLab 1.6.1
- MuJoCo and MuJoCo Warp 3.11.0
- Warp 1.14.0
- Runtime dependencies pinned with hashes in `requirements.lock`

The default image/container names are `cyclo-mjlab:latest` and `cyclo_mjlab`.
The image carries `LABEL version="1.0.0"` to identify its release. This label is
metadata; compatibility checks use the actual installed runtime versions.

The project source and `logs/` directory are shared with the host, so training
results remain available after the container is removed.

### Upgrading an existing Docker environment

Release 1.0.0 requires Python 3.12 and MJLab 1.6.1. Images built for the previous
Python 3.11/MJLab 1.2 environment are incompatible. Updating the checkout does not
update an existing image, even when its tag is `latest`.

After updating the checkout, finish or stop any training or playback before
replacing the container, then run on the host:

```bash
git submodule update --init --recursive
./docker/container.sh build
./docker/container.sh recreate
./docker/container.sh enter
```

`build` updates the image using the build cache; `recreate` replaces the container using that
image. Recreating preserves the source bind mount, training logs, and named cache
volumes, but discards changes stored only in the old container's writable layer.
Keep the same `CYCLO_MJLAB_IMAGE` and `CYCLO_MJLAB_CONTAINER` overrides, if used.

Both `start` and `recreate` check the image against this checkout's
`pyproject.toml` before changing containers. An incompatible image causes the
command to fail with rebuild instructions. After startup, the container's
runtime and CUDA are checked again; failures are reported without declaring the
environment ready. Driver or GPU-access failures require fixing the host's
NVIDIA setup rather than simply rebuilding the image.

To run checks manually inside the container:

```bash
python docker/check_environment.py        # Runtime versions and imports
python docker/check_environment.py --gpu  # Also execute a CUDA tensor operation
```

These checks verify the runtime, not training convergence or individual tasks.

## Try Examples

Both motion-tracking algorithms use `scripts/reinforcement_learning/train.py`
and `play.py`. Task IDs select the algorithm; optional
`--algorithm beyondmimic` or `--algorithm supermimic` makes the choice explicit.

### Velocity

#### Train

```bash
python scripts/reinforcement_learning/train.py Cyclo-Velocity-Flat-K1-Rev1-v0 \
  --env.scene.num-envs 4096
```

#### Play

```bash
python scripts/reinforcement_learning/play.py Cyclo-Velocity-Flat-K1-Rev1-v0 \
  --checkpoint-file logs/rsl_rl/k1_velocity/<run>/model_<iteration>.pt \
  --num-envs 1
```

### Mimic

These commands use BeyondMimic/PPO. Tasks are registered directly as
`Cyclo-BeyondMimic-K1-Rev1-Dance1` and `Dance2`.
See the [BeyondMimic guide](docs/beyondmimic.md) for details.

#### Train Dance1

```bash
python scripts/reinforcement_learning/train.py Cyclo-BeyondMimic-K1-Rev1-Dance1 \
  --env.scene.num-envs 4096
```

#### Train Dance2

```bash
python scripts/reinforcement_learning/train.py Cyclo-BeyondMimic-K1-Rev1-Dance2 \
  --env.scene.num-envs 4096
```

#### Play Dance1

```bash
python scripts/reinforcement_learning/play.py Cyclo-BeyondMimic-K1-Rev1-Dance1 \
  --checkpoint-file logs/rsl_rl/k1_mimic/<run>/model_<iteration>.pt \
  --num-envs 1
```

To display the reference trajectory next to the trained policy during playback:

```bash
python scripts/reinforcement_learning/play.py Cyclo-BeyondMimic-K1-Rev1-Dance1 \
  --checkpoint-file logs/rsl_rl/k1_mimic/<run>/model_<iteration>.pt \
  --num-envs 1 \
  --reference-ghost-offset 1.4,0,0
```

#### Play Dance2

```bash
python scripts/reinforcement_learning/play.py Cyclo-BeyondMimic-K1-Rev1-Dance2 \
  --checkpoint-file logs/rsl_rl/k1_mimic/<run>/model_<iteration>.pt \
  --num-envs 1
```

Replace `<run>` and `<iteration>` with the timestamped run directory and model
iteration to use.

### SuperMimic

Both algorithms use the same Python 3.12 / MJLab 1.6.1 environment. Follow the
[SuperMimic guide](docs/supermimic.md) to initialize its shared source submodule.
Use the same supported 50 Hz K1 reference NPZ; the internal adapter preserves
the original file:

```bash
WANDB_MODE=offline python \
  scripts/reinforcement_learning/train.py --algorithm supermimic \
  Cyclo-SuperMimic-GenCoord-K1-Rev1-001 \
  --motion-file source/assets/motions/K1_rev1/dance1/dance1.npz --num-envs 64

python scripts/reinforcement_learning/play.py \
  --algorithm supermimic Cyclo-SuperMimic-GenCoord-K1-Rev1-001 \
  --motion-file source/assets/motions/K1_rev1/dance1/dance1.npz \
  --checkpoint-dir logs/supermimic/Cyclo-SuperMimic-GenCoord-K1-Rev1-001/<timestamp> --viewer viser
```

Replace `<timestamp>` with the training run directory. Select
`Cyclo-SuperMimic-MaxCoord-K1-Rev1-001` for maximal-coordinate tracking. Local inference
bundles use `--checkpoint-dir` without requiring W&B sync; they are not PPO
checkpoint files. Export/support limits and [validation scope](docs/validation.md)
are documented in the guide.

## Motion Utilities

The motion tools convert K1 CSV motion data into the MuJoCo-ordered NPZ format
and validate or replay the converted trajectory:

```bash
python scripts/tools/motion/csv_to_npz.py --help
python scripts/tools/motion/replay_npz.py --help
```

## License

Software in this repository is distributed under the
[Apache License 2.0](LICENSE), subject to the component notices below. Asset,
reference-data and external-weight rights are described separately in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

### Third-party components

- **MJLab**: Apache License 2.0; see
  [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)
- **HybridRobotics/whole_body_tracking**: MIT License; see
  [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)
- **ROBOTIS AI Sapiens K1 Rev.1 assets**: Apache License 2.0; see
  [LICENSE.ai_sapiens](source/assets/robots/ai_sapiens/LICENSE.ai_sapiens)
- **HybridRobotics/SuperMimic**: Apache License 2.0 with BSD-3-Clause math notices;
  original license, acknowledgements and modification record in
  [third_party/supermimic/PATCHES.md](third_party/supermimic/PATCHES.md)
- **fuyichen0725/blinkmimic**: [MIT](third_party/blinkmimic/LICENSE), with original
  [InstantTrack / supermimic_vibed notices](third_party/blinkmimic/THIRD_PARTY_NOTICES.md).
