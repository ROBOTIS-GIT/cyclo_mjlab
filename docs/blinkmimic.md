# BlinkMimic on K1 Rev.1

Task: `Cyclo-BlinkMimic-K1-Rev1`. This is a separate motion-only task, based on
the complete `feature-supermimic` result at
`ab98ea17ba64726a3fd3b569c5058f8516e8cb69`. BeyondMimic and SuperMimic source,
their core submodule, task configurations and checkpoints are unchanged.

## Primary sources and lineage

- Official source: https://github.com/fuyichen0725/blinkmimic at
  `d264658d6c9ce61fb52f9f0f8985a369f3ccc409`, inspected 2026-11-16.
- Paper: https://arxiv.org/html/2611.16422v1 (sections 4, B, C).
- Released recipe: `third_party/blinkmimic/scripts/presets.sh`, `PRESET[main]`.
  Bare upstream `python -m blinkmimic.train` selects the SuperMimic baseline;
  this task always starts with the released BlinkMimic preset.

SuperMimic supplies collection, learned dynamics and BPTT through imagined
rollouts. BlinkMimic inherits InstantTrack's improvements and adds the progress
schedule, deployment-failure reset allocation and full-iteration graph recipe.
Its source is a separate reimplementation; it is not imported from or patched
into the existing Cyclo SuperMimic core.

The official implementation is vendored with MIT LICENSE, all inherited notices,
original source hashes and one explicit change in `UPSTREAM.json`: contact feet
are recognized by body names, allowing K1 to use the same contact reward.
The remaining upstream Python source files retain their original bytes.

## Applied recipe

| Component | This task's default |
| --- | --- |
| Motion encoder and decoder | Both 2048–1024–512–256, latent 32; upstream full architecture |
| World model | 1024–1024–1024, structured velocity increments / position integration |
| History / future references / imagination | 10 / 10 / 32 frames |
| Optimizers | Muon on policy and world-model hidden matrices, Adam elsewhere |
| Policy | Reference-action residual, EMA 0.995 |
| Data augmentation | World-model left/right mirror share 0.5 |
| Learning rates | Base 4e-4; policy 4× then progress/deadline-triggered decay to 1× |
| Learner batches | WM 2048 × 5 batches; policy 2048 |
| Warm-up | No WM-only phase; initial simulator collection remains upstream |
| Resets | Return bins, failure reset 0.3, duplicate-bin fix |
| Evaluation probes | Share 0.15; start 1000, every 125, decay 0.7, pre-failure windows |
| Performance | Fused Adam / rotations, compilation, update/collection/reset/probe CUDA graphs |
| Probe execution / checkpointing | Async 20 probe steps per iteration; lightweight background checkpoints |
| Training duration / saving | 3500 iterations, every 100; full final checkpoint |
| Environments / replay | 4096 / 1000 vector steps; no automatic reduction |

No human encoder is instantiated. The motion encoder is trained jointly with the
policy decoder through the official imagined-rollout objective.

## K1 adaptations and differences

- Public K1 Rev.1 asset submodule `611a070fad83d2a9ef9c1bc82213988450cf471c`,
  23 joints, 14 tracked bodies, K1 gains, effort limits, armature and action scales.
  Fixed head stays in the physical asset. No Rev.2 model is used.
- K1 nominal elbow 0.6 rad matches the recorded K-POP producer; a deep copy
  prevents mutation of the common asset's 0.95 default.
- K1 simulation capacities `njmax=300`, `nconmax=128` match existing Cyclo K1;
  collision configuration is the unchanged K1 full-collision configuration.
- The exact-state adapter replaces upstream CSV interpolation and velocity
  differentiation. It retains all 223812 original 50 Hz frames and declared
  link-origin velocities from the audited manifest. FK reconstructs reward bodies
  from these physical states. Joint order is matched by name.
- The existing data conversion explicitly used xyzw → wxyz and Newton fixed-head
  merged torso COM → link-origin velocity. Source/derived hashes are checked;
  no motion regeneration, clipping, resampling or new retargeting occurs here.
- The producer's root COM/link temporal inconsistency is retained, as documented
  in the existing manifest. Nineteen out-of-limit joint frames in five songs are
  retained; this task does not silently repair source data.
- Runtime is the existing locked Cyclo Python 3.12 / MJLab 1.6.1, Torch 2.14.0
  CUDA 13.0, MuJoCo 3.11.0 and Warp 1.14.0. The official measurements used
  Torch 2.9.0 CUDA 12.8 and Warp 1.16.0 on a 4090. Timing/convergence claims
  for G1/LAFAN1 do not establish K1/K-POP behavior on an 8 GB laptop.

All recipe flags are present by default. Execution of later mechanisms (policy
updates, graph capture, probes, schedule) requires reaching their corresponding
stage; merely configuring them is not validation that they ran.

The explicit `--execution compiled_no_torch_graphs` comparison preserves every
model, learning rate, batch, history, horizon, mirror and reset-allocation setting.
It removes only `--cuda_graph --graph_early --collect_graph --reset_masked
--eprobe_graph --eprobe_async 20`. Torch compilation remains enabled and probes
run synchronously. MJLab's underlying native MuJoCo physics graph remains.
This is an explicitly recorded departure from the released systems recipe;
its wall-clock results cannot be labeled the complete official recipe.
`run-record.json` lists the removed systems flags. The default remains `official`.

## Isolation and execution

The common launcher routes only the new task to `source/tasks/blinkmimic`.
It starts an independent worker process. Upstream's MJLab performance monkey
patches are confined to that process and never enter the existing learners.
Ordinary task auto-discovery excludes BlinkMimic, just as it excludes SuperMimic.
The task is registered in Cyclo's algorithm dispatcher; its simulator is the
official BlinkMimic TrackingSim rather than an unrelated MJLab/PPO task config.

Use the existing locked environment read-only. No global install, container
restart or source reset is needed. Training requires a verified idle local GPU.
Output directories must be new; logs/checkpoints are never overwritten.

```bash
python scripts/reinforcement_learning/train.py --algorithm blinkmimic \
  Cyclo-BlinkMimic-K1-Rev1 --motion-dir /absolute/path/to/native_xy_newton_com_v1 \
  --run-dir /absolute/path/to/new-run

# Headless all-clip, first-frame evaluation with the official evaluator.
python scripts/reinforcement_learning/play.py --algorithm blinkmimic \
  Cyclo-BlinkMimic-K1-Rev1 --motion-dir /absolute/path/to/native_xy_newton_com_v1 \
  --run-dir /absolute/path/to/new-evaluation --checkpoint /absolute/path/to/ckpt.pt
```

Extra upstream flags use their original underscore spelling. Every explicit
override is recorded separately from the preset. The task never silently changes
model widths, history, batches, environments or replay size to fit memory.
`run-record.json` records command, child PID, source/runtime identities, dataset
manifest and exit status; `console.log`, `args.json` and `train_log.csv` carry
actual runtime evidence. Verify the PID's command before SIGINT. No automatic
restart is implemented. Upstream `last.pt` warm-resume restores models,
optimizers, EMA and reset/probe control state; replay, simulator state and RNG
are not restored. Periodic light checkpoints contain the deployed policy.

## Validation

```bash
PYTHONPATH=.:third_party/supermimic/src:integrations/supermimic/src \
  python -m pytest tests -q
PYTHONPATH=.:third_party/supermimic/src python -m pytest third_party/supermimic/tests -q
python tests/blinkmimic/validate_native.py --motion-dir /absolute/path/to/data \
  --output /absolute/path/to/new-diagnostic
```

The bounded native diagnostic uses all 73 clips, 73 environments and full-sized
networks. It takes eight real steps per clip and one-step Muon gradient updates,
checks finite values and WM freezing during the actor update, and saves/reloads
a diagnostic checkpoint. These bounded diagnostic dimensions are explicit;
they do not alter production defaults. It is not the complete horizon-32
official learner acceptance, a trained baseline or evidence of convergence.
Consult the task-local report and run records for actual execution outcomes.

This task provides simulation training/evaluation only. No robot service,
network transfer, live actuation, push or PR is part of it.
