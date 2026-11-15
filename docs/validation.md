# Validation

Run commands from the repository root in the indicated runtime. Tests use local
K1 references; no external dataset, W&B upload, hardware or long training is needed.

## Research runtime

```bash
uv sync --project integrations/supermimic --python 3.13 --locked
uv run --project integrations/supermimic python -m pytest \
  tests/supermimic third_party/supermimic/tests -q
uv run --project integrations/supermimic ruff check \
  source/tasks/supermimic tests/supermimic --select E4,E7,E9,F,I,B
uv run --project integrations/supermimic cyclo-supermimic smoke --device cpu
uv run --project integrations/supermimic cyclo-supermimic smoke \
  --device cpu --representation maxcoord
```

If installed with `--no-dev`, add `--group dev` to test commands. CPU smoke uses
two environments and a temporary standing clip, checks a real sim transition,
a dynamics optimizer step, and nonzero finite actor gradients through dynamics.
It does not establish policy quality. CUDA kernels may take minutes to compile
on first use; check GPU occupancy before explicitly selecting `--device cuda:0`.

The suite covers actual Dance1/2 source preservation, all-body FK and COM/link
velocity conventions, named-joint/body permutations, metadata and finite float32
validation, rejected unsupported inputs, multi-clip boundaries, CLI selection,
duplicate motion options, and the 50 Hz control-rate guard. The upstream tests
exercise replay, representations, world-model and policy learner behavior.

## Existing BeyondMimic runtime

Inside the existing Cyclo Docker environment:

```bash
python tests/beyondmimic/test_task_aliases.py
```

This compares Dance1/2 train/play/PPO configurations under historical
`Cyclo-Mimic-*` and explicit `Cyclo-BeyondMimic-*` names and verifies that research
modules do not leak into the legacy runtime. Full task commands and checkpoint
export are in [beyondmimic.md](beyondmimic.md).

For a deliberately short PPO execution check, reduce environment/rollout counts
and iterations explicitly, rather than running the normal training defaults:

```bash
python scripts/reinforcement_learning/train.py Cyclo-Mimic-K1-Rev1-Dance1 \
  --gpu-ids None --motion-file source/assets/motions/K1_rev1/dance1/dance1.npz \
  --env.scene.num-envs 2 --agent.max-iterations 1 --agent.num-steps-per-env 4 \
  --agent.algorithm.num-mini-batches 1 --agent.algorithm.num-learning-epochs 1
```

## Checked scope

The integration review verified the following, with generated logs/checkpoints
kept out of Git:

| Check | Result |
| --- | --- |
| Research unit/upstream suite after review fixes | 258 passed; upstream torch.jit deprecation warning |
| Environment/package metadata | Locked package versions unchanged; compatibility wheel embeds Apache-2.0 LICENSE; pytest is development-only |
| Legacy Dance1/2 alias/config test | Passed in Python 3.11 / mjlab 1.2 |
| Original Cyclo NPZ direct input | Dance1 and Dance2 preserve source bytes, frame counts and 50 Hz |
| Current existing CSV converter output | Accepted as current K1 COM profile |
| Native SuperMimic execution | GenCoord and MaxCoord short CPU/CUDA world-model + policy updates and finite checkpoint reloads passed |
| Historical PPO execution/export | Short CPU/CUDA PPO runs, checkpoint loading, YAML generation, ONNX check and finite `(1,23)` TorchScript output passed |

The reviewed input fixes reject float32 overflow, repeated motion-file flags and
control periods other than 0.02 s; compensating timestep/decimation changes are
allowed. COM profiles are checked against every named body/frame, not selected
from filenames or by relaxing tolerances. See [supermimic.md](supermimic.md) for
the current input contract and [UPSTREAM.md](../third_party/supermimic/UPSTREAM.md)
for source modifications.

These checks are execution/compatibility evidence, not convergence or throughput
benchmarks. Interactive playback of a trained W&B bundle, SuperMimic's conditional
ONNX deployment exporter and real-robot behavior have not been validated here.
The existing PPO sim2real schema does not apply automatically to SuperMimic.
