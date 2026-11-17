# Validation

Run from the checkout in the common Python 3.12 / stable MJLab 1.6.1 runtime
installed through [Docker](beyondmimic.md#installation-docker) or the
[shared local environment](supermimic.md#shared-cyclo-mjlab-environment). Tests use
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
SuperMimic native motion/replay checks. These exercise original Dance1/2 together, deterministic
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

# Run sequentially on a free GPU. First compilation can take several minutes.
python tests/supermimic/validate_native.py GenCoord dance1 /tmp/cyclo-supermimic-gen-cuda \
  --device cuda:0 --learner-execution eager
python tests/supermimic/validate_native.py MaxCoord dance2 /tmp/cyclo-supermimic-max-compiled \
  --device cuda:0 --learner-execution compiled
# Optional small sensor/history check; repeat with either representation.
python tests/supermimic/validate_native.py GenCoord dance1 /tmp/cyclo-supermimic-history \
  --device cuda:0 --actor-obs proprio --history-length 2
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

The default is privileged `rep_state`, history length 1 and eager CPU execution.
The CUDA eager option disables Torch compilation and extends only the test's
capture warmup. The compiled option keeps the production warmup/capture schedule
and asserts that both learner CUDA graphs were captured and Torch compiled graphs
were generated. Small networks, two environments and five updates per learner
keep these checks bounded; they are not performance benchmarks or full training
acceptance. `proprio` with history length 2 is a separate eager smoke, not evidence
for every history/noise/observation configuration.

`WANDB_MODE=disabled` needs no local W&B service; offline mode is also supported
but starts that service. Regression tests assert that offline/disabled/dryrun
artifact handling never calls the remote API.

## Existing BeyondMimic behavior

```bash
python tests/beyondmimic/test_task_registration.py
python scripts/reinforcement_learning/train.py Cyclo-BeyondMimic-K1-Rev1-Dance1 \
  --gpu-ids None --motion-file source/assets/motions/K1_rev1/dance1/dance1.npz \
  --env.scene.num-envs 2 --agent.max-iterations 1 --agent.num-steps-per-env 4 \
  --agent.algorithm.num-mini-batches 1 --agent.algorithm.num-learning-epochs 1
```

`Cyclo-BeyondMimic-*` task IDs preserve the existing train/play/PPO
configurations and remain separate from SuperMimic's runner and checkpoint format. See [beyondmimic.md](beyondmimic.md) for normal commands and
the [recorded scope](#recorded-scope) below for historical checkpoint/export checks.

## Recorded scope

The following results describe the earlier runtime integration and Docker
acceptance at the revisions identified below. They are not reruns of the 0.0.4
baseline synchronization. Baseline commit `59f650a` removed the migration-specific
PPO/reset tests, old checkpoint fixtures and standalone export probe; those files
are not part of the current test commands. The remaining alias, SuperMimic and
shared-core tests and the explicit learner harness above are still available.

The stable-runtime integration passed all 82 repository tests with native CPU
checks enabled. The pinned shared suite separately passed 232 tests and 13
subtests in the same stable runtime. Both GenCoord/Dance1 and MaxCoord/Dance2 CPU
acceptance runs completed five world-model and five actor updates, saved final
weights, reloaded and played four native steps.

The Docker image built from source revision `305a380` subsequently passed all 82
repository tests, with the baseline native PPO checks using CUDA. Its installed
versions and 155 native/shared-source hashes matched the checkout; runtime tests
had networking disabled, enforced W&B disabled mode, and no host source or Python
environment mounted. The native SuperMimic matrix passed as follows:

| Mode | GenCoord / Dance1 | MaxCoord / Dance2 |
| --- | --- | --- |
| CUDA eager, `rep_state`, history 1 | Passed | Passed |
| CUDA compiled/captured, `rep_state`, history 1 | Passed; 11 compiled graphs | Passed; 10 compiled graphs |
| CUDA eager, `proprio`, history 2 | Bounded smoke passed | Bounded smoke passed |

Every cell completed five updates per learner, changed Muon-managed matrices,
kept world-model parameters frozen during actor updates, saved exact final
parameters, and reloaded for four finite-action native playback steps. Both
compiled cases captured the world-model and actor CUDA graphs. Their first actor
updates took about 221 and 188 seconds including compilation; these are startup
observations, not throughput claims. The history-mode smoke does not cover
compiled history-mode training or every observation/randomization setting.
Fresh container processes also passed the installed `cyclo-supermimic --help`
entry point and the common `play.py --algorithm supermimic` CLI for both local
CUDA-trained bundles, with two headless CUDA steps each and no W&B sync.

Source NPZ bytes and 50 Hz frame sequences remained unchanged.
The common dependency resolver retained all binary package versions from the
validated baseline lock. Generated logs/checkpoints remain outside Git.

The baseline update separately passed 17 CPU and five host GPU checks, plus
fresh-process playback/export probes. Those GPU checks cover native MuJoCo Warp
simulation with eager PPO; they are not SuperMimic GPU acceptance. Its old MJLab
1.2 checkpoint fixtures were initialized but untrained, so loading them establishes
structural compatibility, not historical trained-policy quality. Older Python
3.13/research-backend SuperMimic and Python 3.11/MJLab 1.2 checks are historical;
they do not validate this stable-runtime combination.

The baseline Docker checks at source revision `6e09bf7` separately passed native
CUDA PPO, fresh-process playback and live-policy/JIT/ONNX parity (including
normalization and 23-action joint order), public train/export commands and EGL
rendering. Full optimizer continuation and historical trained-policy compatibility
were not established. An earlier canceled unpack did
not establish Docker acceptance; the later completed builds and container runs
do. Logs, final bundles, commands, source/image identities and resource records
are preserved outside Git. The later 0.0.4 baseline synchronization preserves
these historical results without claiming fresh GPU or Docker acceptance.

Interactive Compose/X11 and Native/Viser viewing, authenticated W&B retrieval,
SuperMimic's conditional ONNX deployment exporter and real-robot behavior remain
unvalidated. These bounded checks do not demonstrate convergence, throughput or
hardware readiness. PPO sim2real exports do not automatically apply to SuperMimic.

Source provenance and modifications are recorded in
[third-party notices](../THIRD_PARTY_LICENSES.md) and the shared fork's
[PATCHES.md](../third_party/supermimic/PATCHES.md).
