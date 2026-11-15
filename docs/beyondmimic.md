# BeyondMimic motion tracking (existing Mimic/PPO)

This is Cyclo's K1/mjlab adaptation of the motion-tracking stack from
HybridRobotics/whole_body_tracking (BeyondMimic). See
[attribution](../THIRD_PARTY_LICENSES.md). The source remains at
`source/tasks/mimic` to preserve imports and downstream extensions.

Explicit names `Cyclo-BeyondMimic-K1-Rev1-Dance1` and `Dance2` reuse exactly
the old `Cyclo-Mimic-K1-Rev1-Dance1` and `Dance2` environment/PPO settings.
Both spellings work with the same native train/play scripts. Existing task IDs,
checkpoints, outputs and flags remain supported; Velocity remains unchanged.

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

After entering the container, the training and playback commands below can be
run directly from `/workspace/cyclo_mjlab`.

### Docker commands

| Command | Description |
| --- | --- |
| `./docker/container.sh start` | Build the image if needed, initialize submodules, and start the container |
| `./docker/container.sh enter` | Open an interactive shell in the running container |
| `./docker/container.sh stop` | Stop the container |
| `./docker/container.sh logs` | Follow the container logs |
| `./docker/container.sh clean` | Remove the container and image while preserving cache volumes |

The Docker image includes:

- Python 3.12
- MJLab 1.6.1
- MuJoCo and MuJoCo Warp 3.11.0
- Warp 1.14.0
- All dependencies declared in `pyproject.toml`

The current Docker build/start has not completed validation; host CPU/GPU evidence
and installation alternatives are in [runtime-migration.md](runtime-migration.md).

The project source and `logs/` directory are shared with the host, so training
results remain available after the container is removed.

## Choose and train

```bash
python scripts/reinforcement_learning/train.py --algorithm beyondmimic \
  Cyclo-BeyondMimic-K1-Rev1-Dance1 --env.scene.num-envs 4096
```

Task IDs also infer the algorithm, so existing commands below remain valid.

### Mimic

#### Train Dance1

```bash
python scripts/reinforcement_learning/train.py Cyclo-Mimic-K1-Rev1-Dance1 \
  --env.scene.num-envs 4096
```

#### Train Dance2

```bash
python scripts/reinforcement_learning/train.py Cyclo-Mimic-K1-Rev1-Dance2 \
  --env.scene.num-envs 4096
```

#### Play Dance1

```bash
python scripts/reinforcement_learning/play.py Cyclo-Mimic-K1-Rev1-Dance1 \
  --checkpoint-file logs/rsl_rl/k1_mimic/<run>/model_<iteration>.pt \
  --num-envs 1
```

To display the reference trajectory next to the trained policy during playback:

```bash
python scripts/reinforcement_learning/play.py Cyclo-Mimic-K1-Rev1-Dance1 \
  --checkpoint-file logs/rsl_rl/k1_mimic/<run>/model_<iteration>.pt \
  --num-envs 1 \
  --reference-ghost-offset 1.4,0,0
```

#### Play Dance2

```bash
python scripts/reinforcement_learning/play.py Cyclo-Mimic-K1-Rev1-Dance2 \
  --checkpoint-file logs/rsl_rl/k1_mimic/<run>/model_<iteration>.pt \
  --num-envs 1
```

Replace `<run>` and `<iteration>` with the timestamped run directory and model
iteration to use.

## Data and outputs

K1 CSV motion conversion uses `scripts/tools/motion/csv_to_npz.py` in this
Python 3.12 environment. The NPZ carries named joints, named body states and FPS;
task configs select dance1/dance2 by default, with `--motion-file` override.

PPO writes `logs/rsl_rl/k1_mimic/<run>/model_<iteration>.pt`, environment/agent
YAML and `params/sim2real.yaml`. Play loads a local checkpoint and exports
`exported/policy.onnx` and `exported/policy.pt`; `--export-only True` performs
export without opening a viewer. Keep using the established Cyclo policy
interface; this manual does not authorize hardware execution.

The same 50 Hz Cyclo NPZ can now be passed directly to SuperMimic; its internal
adapter handles the compact-reference layout without changing this
PPO reference or requiring a separate conversion command.

SuperMimic checkpoints cannot be loaded by this PPO runner. To train the other
algorithm, use its [task manual in the same runtime](supermimic.md).
