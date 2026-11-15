# MJLab 1.6.1 runtime

Cyclo uses Python 3.12, MJLab 1.6.1, MuJoCo/MuJoCo Warp 3.11.0,
Warp 1.14.0 and Torch 2.14.0. The complete runtime dependency graph, including
RSL-RL 5.4.2 and mjviser 0.0.16, is pinned with hashes in
[`requirements.lock`](../requirements.lock).

[Official MJLab metadata](https://pypi.org/project/mjlab/1.6.1/) supports
Python 3.10 through 3.13. Python 3.12 is Cyclo's selected and tested baseline;
the upgrade does not require Python 3.13. The lock targets Linux x86-64.
Do not combine the previous MJLab 1.2/MuJoCo 3.5 environment with this lock.

## Installation and existing workflows

The README's Docker commands and task IDs remain valid. The new defaults are
image `cyclo-mjlab:1.6.1-py312`, container `cyclo_mjlab_161`, and Compose project
`cyclo_mjlab_161`; the previous image and running containers are not replaced.
`CYCLO_MJLAB_IMAGE` and `CYCLO_MJLAB_CONTAINER` can override the image and
container names as before. Docker installs the hashed runtime lock without
retaining a second copy of downloaded wheels in a pip cache.

Use a repository checkout with both submodules initialized (`git submodule update
--init --recursive`);
the wheel alone does not include robot assets or the CLI scripts. For an isolated
local Python 3.12 environment:

```bash
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python --require-hashes requirements.lock
uv pip install --python .venv/bin/python --no-deps -e third_party/supermimic -e .
```

Regenerate the lock only when deliberately updating dependencies, then rerun
native regressions:

```bash
uv pip compile pyproject.toml --python-version 3.12 --generate-hashes \
  --no-emit-package supermimic -o requirements.lock
```

The SuperMimic source dependency is resolved with the same graph but installed
from its pinned submodule, not an unrelated package-index release. Both algorithms
share this MJLab runtime; the optional research backend in the fork is not needed.

The existing K1 Velocity and BeyondMimic/PPO task IDs, training/play arguments,
23-action order, and original 50 Hz Dance1/Dance2 NPZ files are preserved.
Actor/critic observation widths remain 80/95 for Velocity and 124/256 for Mimic.
The reward names and weights are checked against the previous baseline.

## API adaptations

- K1 collision masks and priorities are now explicit, as required by MJLab 1.6.
  Foot/non-foot contact dimensions, friction and PD actuator configuration are
  preserved.
- The upright reward uses upstream `upright` without terrain sensors, retaining
  the exponential projected-gravity XY objective and torso selection.
  `variable_posture` remains upstream code.
- Small attributed `foot_height` and `feet_clearance` adapters preserve the old
  selected-site world-Z observation and horizontal-speed-weighted clearance
  cost. They intentionally do not adopt terrain-height sensors or change the
  strict command-speed threshold.
- Reference commands support scoped reset dispatch. Explicit and automatic
  resets both expose the next target after the sampled physical frame. Partial
  resets preserve other environments' frames and aligned targets. Failure counts
  accumulate across resets and enter the global EMA once per simulation step.
  Timer expiry and clip wrap forward the simulator before recomputing alignment.

## Regression checks

Install test tools separately from runtime dependencies:

```bash
uv pip install --python .venv/bin/python pytest ruff
.venv/bin/python -m pytest -q tests
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 CYCLO_RUN_NATIVE_TESTS=1 \
  .venv/bin/python -m pytest -q tests
```

The opt-in native suite uses three CPU environments, four rollout steps and one
PPO update per task. It checks finite observations/actions/rewards, changed actor
parameters, final checkpoint equality, fresh-model reload, and short policy
playback. Both Dance references also exercise actual environment/manager dispatch
for partial resets, clip wrap and automatic resets. The default suite skips
simulation.

A fresh-process checkpoint probe runs three CPU policy steps and checks JIT/ONNX
export agreement with the live policy (including normalization), 23 finite actions
and joint-order metadata against sim2real YAML:

```bash
.venv/bin/python tests/checkpoint_probe.py \
  --task Cyclo-Mimic-K1-Rev1-Dance1 \
  --checkpoint /path/to/model_0.pt --output-dir /tmp/k1-checkpoint-check
```

The normal CLI remains available, including headless export:

```bash
python scripts/reinforcement_learning/train.py Cyclo-Mimic-K1-Rev1-Dance1 \
  --gpu-ids None --env.scene.num-envs 2 --agent.max-iterations 1 \
  --agent.num-steps-per-env 4 --agent.algorithm.num-mini-batches 1 \
  --agent.algorithm.num-learning-epochs 1
python scripts/reinforcement_learning/play.py Cyclo-Mimic-K1-Rev1-Dance1 \
  --checkpoint-file /path/to/model_0.pt --device cpu --num-envs 1 --export-only True
```

Migration validation passed 17 CPU checks and five bounded GPU checks on the
host Python 3.12 environment. Velocity, Dance1 and Dance2 each performed a real
PPO update, saved the final parameters and reloaded for short playback. GPU checks
exercised native MuJoCo Warp CUDA simulation; the PPO models used eager Torch
execution, not `torch.compile`. Fresh-process CPU playback/export checks cover
all three tasks. Checkpoints generated from the previous main
baseline with MJLab 1.2 also loaded for Velocity and Mimic, preserving inference
on saved observation tensors. These fixtures were initialized but untrained:
they establish structural loading/inference compatibility,
not a claim that every historical trained checkpoint or long-term behavior is
unchanged. TorchScript emits an upstream deprecation warning; existing JIT export
is retained. RSL-RL normalization can reject loading weights back into an object
that has already trained under inference mode; use a fresh runner, as the existing
play and resume entry points do.

These bounded checks do not establish convergence, hardware deployment safety,
interactive viewer behavior or optimizer continuation. Sim2real export was checked as a
file/metadata contract; no robot was operated.

Docker build and isolated runtime acceptance subsequently passed for tested source
revision [`6e09bf7`](https://github.com/ROBOTIS-GIT/cyclo_mjlab/commit/6e09bf7de88a9f87692fd7d39f8c4213017f8a92)
using image `cyclo-mjlab:accept-baseline-6e09bf7-task3`. The cached build completed,
and the runtime verified 61 committed source-file hashes. It ran with networking
disabled and no host source or virtual-environment mounts; only the external test
tools, harness and evidence directory were mounted.

The image passed:

- All 17 tests, including native CUDA PPO updates, reset dispatch and checkpoint checks.
- Fresh-process CPU playback/export for Velocity, Dance1 and Dance2, with JIT and
  ONNX outputs matching the live policy on identical observations, including normalization.
- The public export-only CLI and one CPU PPO update through the public training CLI
  (two environments, four rollout steps each).
- One nonconstant 160-by-120 RGB frame rendered with CUDA simulation and EGL.

An initial missing `py.py` dependency in the external pytest-tool overlay was
corrected outside the production source; the acceptance rerun passed without a
production change. These bounded results do not verify interactive Compose/X11
operation, convergence, robot deployment or full optimizer continuation.
