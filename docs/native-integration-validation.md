# Native algorithm selection validation (2026-11-15)

Repository `/home/hc/Documents/Codex/k1-walking/cyclo_mjlab`, isolated worktree
`/home/hc/Documents/Codex/2026-11-15/task-3/feature-supermimic`, branch
`feature-supermimic`, fetched base `e659f69af227d244a4edab0bb9228e7aeaadef45`.
This follows the initial integration and GPU checks in
[supermimic-validation.md](supermimic-validation.md).

## Implementation and compatibility

- SuperMimic implementation now lives in `source/tasks/supermimic` (configuration,
  K1 research robot adapter, motion conversion, runner CLI and bounded smoke).
  `integrations/supermimic` retains only its locked runtime and compatibility
  package/commands; the algorithms remain upstream and unchanged.
- Native `scripts/reinforcement_learning/train.py` and `play.py` dispatch before
  simulator imports. `--algorithm beyondmimic|supermimic` is explicit; task IDs
  infer the same choice when the flag is omitted. Wrong algorithm/task pairs or
  the wrong Python/mjlab runtime fail early with setup guidance. No automatic
  installation, environment replacement or subprocess relaunch occurs.
- Existing `Cyclo-Mimic-K1-Rev1-Dance1`/`Dance2` retain the original PPO settings,
  observation/reward definitions and output names. Explicit
  `Cyclo-BeyondMimic-K1-Rev1-Dance1`/`Dance2` registrations deep-copy those exact
  registered configs. The legacy `source/tasks/mimic` import path stays valid.
- Existing Python 3.11/mjlab 1.2 task auto-registration is preserved and excludes
  the incompatible research family. The research runtime explicitly registers
  native SuperMimic tasks without importing old PPO assets/configurations.
- K1 asset source, existing motion converter, PPO learner, observation and reward
  code, Velocity implementation, root dependencies and existing Docker setup
  were not modified. Only the native launcher, task-family registration and
  explicit task aliases changed around the legacy execution path.

## Actual verification

GPU occupancy was checked before each sequential GPU phase: no other compute
jobs were present. No existing process/container was stopped. The existing
`cyclo-mjlab:latest` image (`868a9812c8da`) was run in **new disposable** containers
with `--network none`, the isolated worktree mounted at `/workspace/cyclo_mjlab`,
and a separate `/tmp/cyclo-warp-regression` cache. Installed runtime was checked:
Python 3.11.16, mjlab 1.2.0, MuJoCo 3.5.0, Warp 1.12.0, torch 2.14.0.

| Check | Result |
| --- | --- |
| Research runtime `python -m pytest tests/supermimic third_party/supermimic/tests -q` | 232 passed in 6.45 s (final run); one upstream torch.jit deprecation warning |
| Existing runtime `python tests/beyondmimic/test_task_aliases.py` | Passed: both Dance1/2 aliases match legacy train/play/PPO configs; SuperMimic not imported |
| Native `train.py --list-tasks` | Both algorithm families displayed without simulator import |
| Native SuperMimic train/play `<task> --help` | Correct original runner/viewer options displayed |
| Mismatched selector/task and wrong installed runtime | Rejected before simulator-specific imports; covered by unit tests and manual wrong-runtime invocation |
| `scripts/tools/motion/to_supermimic.py` | Converted actual committed dance1 NPZ to `.supermimic-data/native-dance1.npz`; original preserved |
| Ruff on new native modules/compatibility packages/tests, compileall, `git diff --check` | Passed |

The first alias test compared callable object identities from separately created
configs; that test failed without indicating a behavior difference. Comparison
now normalizes callable definitions/partial arguments, and aliases copy the
registered historical configs directly. The corrected test passed in the actual
legacy environment.

### Native BeyondMimic CUDA PPO regression

Inside the disposable existing-runtime container, with `OMP_NUM_THREADS=2`:

```bash
python scripts/reinforcement_learning/train.py --algorithm beyondmimic \
  Cyclo-BeyondMimic-K1-Rev1-Dance1 --env.scene.num-envs 2 \
  --agent.max-iterations 2 --agent.num-steps-per-env 8 \
  --agent.algorithm.num-mini-batches 1 --agent.algorithm.num-learning-epochs 1 \
  --agent.actor.hidden-dims '(32,32)' --agent.critic.hidden-dims '(32,32)' \
  --agent.run-name native-beyond-regression
```

**Passed, exit 0, 161.96 s including first Warp compilation**, two real PPO
iterations. Checkpoints `model_0.pt`, `model_1.pt`, tensorboard metrics,
`params/env.yaml`, `params/agent.yaml`, and `params/sim2real.yaml` were generated
under `logs/rsl_rl/k1_mimic/2026-11-15_02-33-13_native-beyond-regression/`.
Raw output: `logs/native-integration-validation/beyond-train.log`.

The historical command (no algorithm flag, old task ID) was also run with default
network dimensions, one PPO iteration, and the same small rollout/minibatch:

```bash
python scripts/reinforcement_learning/train.py Cyclo-Mimic-K1-Rev1-Dance1 \
  --env.scene.num-envs 2 --agent.max-iterations 1 --agent.num-steps-per-env 8 \
  --agent.algorithm.num-mini-batches 1 --agent.algorithm.num-learning-epochs 1 \
  --agent.run-name native-legacy-regression
```

**Passed, exit 0, 29.32 s**. Raw output:
`logs/native-integration-validation/legacy-train.log`.

Historical play/export was then exercised on that default-network checkpoint:

```bash
python scripts/reinforcement_learning/play.py Cyclo-Mimic-K1-Rev1-Dance1 \
  --checkpoint-file logs/rsl_rl/k1_mimic/2026-11-15_02-38-48_native-legacy-regression/model_0.pt \
  --num-envs 1 --export-only True --device cuda:0
```

**Passed, exit 0, 36.45 s**, loaded the actual PPO checkpoint and generated
`exported/policy.onnx` (927,297 bytes) and `exported/policy.pt` (937,942 bytes).
Raw output: `logs/native-integration-validation/legacy-play-export.log`.
`onnx.checker.check_model` passed on the generated ONNX. The exported
TorchScript loaded on CPU and produced a finite tensor of shape `(1, 23)` for a
zero observation. Evidence: `logs/native-integration-validation/legacy-export-check.log`.
No interactive viewer or physical robot was started.

### Native SuperMimic CUDA learning

Same research runtime/cache as the prior CUDA checks, sequentially for GenCoord
and MaxCoord; `torch.compile` and native CUDA graph behavior enabled, no wall-time
timeout, W&B offline, `OMP_NUM_THREADS=2`:

```bash
# Repeat for REP=GenCoord and REP=MaxCoord.
REP=GenCoord
integrations/supermimic/.venv/bin/python scripts/reinforcement_learning/train.py \
  --algorithm supermimic Cyclo-SuperMimic-${REP}-K1-Rev1 \
  --motion-file .supermimic-data/dance1.npz --device cuda:0 --num-envs 2 \
  --runner.max-iterations 3 --runner.buffer-size 8 \
  --runner.wm.horizon 2 --runner.wm.batch-size 2 --runner.wm.num-batches 1 \
  --runner.policy.horizon 2 --runner.policy.batch-size 2 --runner.policy.num-batches 1 \
  --runner.dynamics-hidden-dims '(32,32)' --runner.actor-hidden-dims '(32,32)' \
  --runner.critic-hidden-dims '(32,)' --runner.log-interval 1 \
  --runner.save-interval 1 --runner.run-name native-source-integration
```

| Result | GenCoord | MaxCoord |
| --- | --- | --- |
| Exit | 0 | 0 |
| Completed outer iterations | 3 | 3 |
| Wall time including learner compilation | 57.56 s | 24.64 s |
| Final WM loss | 0.89698 | 0.29415 |
| Final policy gradient norm | 0.1848 | 0.2651 |
| Final checkpoints | WM + actor + critic | WM + actor + critic |

Checkpoint directories:
- `logs/Cyclo-SuperMimic-GenCoord-K1-Rev1/2026-11-15_11-36-53/`
- `logs/Cyclo-SuperMimic-MaxCoord-K1-Rev1/2026-11-15_11-37-35/`

All checkpoint tensors reloaded with `weights_only=True, map_location='cpu'`
and were finite (WM 22 tensors, actor 42, critic 6, each representation).
Raw evidence: `logs/native-integration-validation/super-{GenCoord,MaxCoord}.log`
and `checkpoints.json`. Existing nonfatal W&B remote artifact-pruning warnings
are unchanged; no remote sync was performed.

## Remaining limits

These are short execution/optimization regressions, not converged motion-quality
or throughput benchmarks. No hardware execution or long training was performed.
SuperMimic viewer dispatch/help was checked, but interactive playback of a trained
W&B bundle and its conditional ONNX deployment exporter were not exercised.
SuperMimic uses the original W&B bundle loader and does not produce the existing
PPO `sim2real.yaml` contract. Existing Docker and research environments remain
separate. Raw logs and generated artifacts stay git-ignored in the worktree.

## Subsequent direct-reference adapter

Separate user conversion is no longer required. See
[direct 50 Hz reference reuse and verified COM conventions](reference-adapter-validation.md)
for the newer input contract, source-preservation evidence and both-runtime checks.
Earlier converted-file commands above record historical executions only.
