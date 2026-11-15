# Validation

Run from the checkout in the common Python 3.12 / stable MJLab 1.6.1 runtime
installed as described in [runtime-migration.md](runtime-migration.md). Tests use
local K1 references; no external dataset, W&B upload, hardware or long training
is needed. Install pytest and ruff as development tools.

## Unit and bounded native checks

```bash
python -m pytest tests -q
python -m pytest third_party/supermimic/tests -q
ruff check source/tasks/supermimic tests/supermimic --select E4,E7,E9,F,I,B
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 CYCLO_RUN_NATIVE_TESTS=1 \
  TORCHDYNAMO_DISABLE=1 WANDB_MODE=disabled python -m pytest tests -q
```

The default suite skips native simulation. The opt-in suite adds bounded CPU
Velocity/BeyondMimic PPO updates, reset/checkpoint tests, and SuperMimic native
motion/replay checks. The latter exercise original Dance1/2 together, deterministic
multi-clip loading, partial-reset isolation, one-time failure EMA consumption,
real clip-end and episode-timeout truncation, and replay/history boundaries.
Nonzero raw actions also produce exactly the same native processed targets as
the dynamics model; unsupported clipping and joint-order contracts are rejected. K1 numerical checks use
nonidentity root rotation and nonzero root angular velocity, compare native FK
poses/link-origin twists with MuJoCo, recover all 23 joint positions/velocities
with IK in world and root-local frames, and verify finite nonzero gradients.

Input tests cover original source hashes, every body's FK and COM/link velocity
convention, named-joint/body permutations, finite float32 values, rejected metadata,
50 Hz timing, duplicate motion options and task/representation selection. The
shared-core suite covers replay, representations and actual learner behavior.

The lightweight smoke commands remain available:

```bash
cyclo-supermimic smoke --device cpu
cyclo-supermimic smoke --device cpu --representation maxcoord
```

They fit a one-step dynamics loss and check finite actor gradients. They do not
perform the complete official policy learner update; use the acceptance harness
below for that contract.

## Genuine learners and local inference bundles

```bash
CUDA_VISIBLE_DEVICES='' WANDB_MODE=disabled TORCHDYNAMO_DISABLE=1 \
  python tests/supermimic/validate_native.py GenCoord dance1 /tmp/cyclo-supermimic-gen
CUDA_VISIBLE_DEVICES='' WANDB_MODE=disabled TORCHDYNAMO_DISABLE=1 \
  python tests/supermimic/validate_native.py MaxCoord dance2 /tmp/cyclo-supermimic-max
```

The harness enforces `WANDB_MODE=disabled` in its own process, verifies the active
run mode, and puts W&B scratch files beneath the supplied output directory. It
does not change global user logging preferences. Preserve that output directory
for its result JSON, final bundles and adapter evidence. Normal train commands
still honor the caller's W&B configuration; the local-bundle runner name alone
is not an offline guarantee.

The harness collects real native transitions and invokes the official multi-step
world-model and actor-critic BPTT learners. It asserts updates to Muon-managed
hidden matrices in both networks, frozen/unchanged world-model parameters during
the actor step, and exact equality of final in-memory and saved tensors. It then
reloads the local bundle and takes four finite-action native playback steps.
The nondefault 0.0025 s simulation timestep and decimation 8 must survive reload;
checking only their 50 Hz product would miss a representation-integration change.

These runs use privileged `rep_state`, history length 1 and eager CPU execution.
They do not establish full `proprio`/multi-frame history acceptance, CUDA graph
capture or compiled learner performance. `WANDB_MODE=disabled` needs no local W&B
service; offline mode is also supported but starts that service. Regression tests
assert that offline/disabled/dryrun artifact handling never calls the remote API.

## Existing BeyondMimic behavior

```bash
python tests/beyondmimic/test_task_aliases.py
python scripts/reinforcement_learning/train.py Cyclo-Mimic-K1-Rev1-Dance1 \
  --gpu-ids None --motion-file source/assets/motions/K1_rev1/dance1/dance1.npz \
  --env.scene.num-envs 2 --agent.max-iterations 1 --agent.num-steps-per-env 4 \
  --agent.algorithm.num-mini-batches 1 --agent.algorithm.num-learning-epochs 1
```

Historical `Cyclo-Mimic-*` and explicit `Cyclo-BeyondMimic-*` names select identical
train/play/PPO configurations. Both remain separate from SuperMimic's runner and
checkpoint format. See [beyondmimic.md](beyondmimic.md) for normal commands and
[runtime-migration.md](runtime-migration.md) for checkpoint/export regression scope.

## Recorded scope

The stable-runtime integration passed all 82 repository tests with native CPU
checks enabled. The pinned shared suite separately passed 232 tests and 13
subtests in the same stable runtime. Both GenCoord/Dance1 and MaxCoord/Dance2 acceptance runs completed
five world-model and five actor updates, saved final weights, reloaded and played
four native steps. Source NPZ bytes and 50 Hz frame sequences remained unchanged.
The common dependency resolver retained all binary package versions from the
validated baseline lock. Generated logs/checkpoints remain outside Git.

The baseline update separately passed 17 CPU and five host GPU checks, plus
fresh-process playback/export probes. Those GPU checks cover native MuJoCo Warp
simulation with eager PPO; they are not SuperMimic GPU acceptance. Its old MJLab
1.2 checkpoint fixtures were initialized but untrained, so loading them establishes
structural compatibility, not historical trained-policy quality. Older Python
3.13/research-backend SuperMimic and Python 3.11/MJLab 1.2 checks are historical;
they do not validate this stable-runtime combination.

Docker build/start remains unverified after an earlier storage-limited unpack
was canceled. No Docker retry or GPU run was used for this integration's CPU
results. Interactive Native/Viser viewing, authenticated W&B retrieval,
SuperMimic's conditional ONNX deployment exporter and real-robot behavior remain
unvalidated. These bounded checks do not demonstrate convergence, throughput or
hardware readiness. PPO sim2real exports do not automatically apply to SuperMimic.

Source provenance and modifications are recorded in
[third-party notices](../THIRD_PARTY_LICENSES.md) and the shared fork's
[PATCHES.md](../third_party/supermimic/PATCHES.md).
