# BlinkMimic on K1 Rev.1

Task: `Cyclo-BlinkMimic-K1-Rev1`. This motion-only task is selected explicitly
through the common launcher. Existing BeyondMimic and SuperMimic implementations,
task configurations, asset defaults and shared-core revision remain unchanged.

## Primary sources and lineage

- Official source: https://github.com/fuyichen0725/blinkmimic at
  `d264658d6c9ce61fb52f9f0f8985a369f3ccc409`, inspected 2026-11-16.
- Paper: https://arxiv.org/html/2611.16422v1 (sections 4, B, C).
- Released recipe: `third_party/blinkmimic/scripts/presets.sh`, `PRESET[main]`.
  Bare upstream `python -m blinkmimic.train` selects the SuperMimic baseline;
  this task always starts with the released BlinkMimic preset.

The algorithm follows the SuperMimic collection → learned dynamics → BPTT
training pattern. The code provenance is separate: this task vendors Fuyi Chen’s
MIT BlinkMimic implementation, derived from the MIT InstantTrack and
`supermimic_vibed` implementations. Cyclo’s existing SuperMimic task uses the
Apache-licensed HybridRobotics/ROBOTIS shared core with its own BSD math notices.
They belong to the same algorithm family, but are not the same codebase or
license lineage. This task does not import the shared SuperMimic learner.

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
- BlinkMimic's nominal elbow is 0.6 rad. A deep copy preserves the common asset's
  0.95 default for existing tasks. Checkpoints retain the policy's nominal pose
  and action scale; consumers must use that same contract.
- K1 simulation capacities `njmax=300`, `nconmax=128` and the full-collision
  configuration match the existing public K1 task.
- The exact-state adapter replaces upstream CSV interpolation and velocity
  differentiation. It retains every supplied 50 Hz frame and declared world
  link-origin velocity, and matches joint columns by name. It does not resample,
  smooth, crop or re-differentiate references.
- The released training contact term still uses the original 0.045 m simulated
  ankle-origin threshold. Public K1 foot spheres extend 0.065 m below the ankle;
  the simulator criterion has not been calibrated to K1 geometry. The per-clip
  reference floor fix is enabled, but does not alter that simulator threshold.
  Evaluation uses the original tracking reward without the training contact term.
- Runtime is the existing locked Cyclo Python 3.12 / MJLab 1.6.1, Torch 2.14.0
  CUDA 13.0, MuJoCo 3.11.0 and Warp 1.14.0. The official measurements used
  Torch 2.9.0 CUDA 12.8 and Warp 1.16.0 on a 4090. Timing/convergence claims
  for G1/LAFAN1 do not establish K1 convergence or performance on another GPU.

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

## Reference preparation

The library accepts any nonempty number of clips. No dataset or checkpoint is
bundled with this feature. Inputs must be compact K1 NPZ files with scalar `fps=50`,
all 23 unique `joint_names`, and these finite arrays in the float32 loader format:

| Field | Shape | Convention |
| --- | --- | --- |
| `root_pos_w` | T × 3 | metres, root-link origin in world |
| `root_quat_w` | T × 4 | normalized wxyz, root-to-world rotation |
| `root_lin_vel_w` | T × 3 | m/s, world-frame link-origin velocity, not COM velocity |
| `root_ang_vel_w` | T × 3 | rad/s, world frame |
| `joint_pos`, `joint_vel` | T × 23 | radians and rad/s, named joint order |

Every clip needs at least two frames and the explicit scalar declarations
`backend=mujoco`, `quaternion_order=wxyz`, `velocity_frame=world`, and
`velocity_point=link_origin`. Unknown producer conventions must be resolved
before preparation; do not relabel data to bypass that requirement.

For a supported full-body Cyclo reference, the existing input converter can
produce this compact format. It is a data adapter, not a call to the SuperMimic
learner. For example, using the pre-existing Dance1 reference:

```bash
mkdir -p .blinkmimic-data
python scripts/tools/motion/to_supermimic.py \
  source/assets/motions/K1_rev1/dance1/dance1.npz .blinkmimic-data/dance1.npz
python scripts/tools/motion/prepare_blinkmimic.py .blinkmimic-data/dance1.npz \
  --output .blinkmimic-data/example
```

For several already normalized files, pass them in the desired clip order to
`prepare_blinkmimic.py`. Basenames must be unique. Preparation validates inputs,
copies their bytes into a new directory, and writes `manifest.json`. Existing
output directories are rejected. The manifest's `clips` array preserves order;
each row has `song` (a clip identifier), `derived`, `derived_sha256`, `frames`,
and `fps`. Paths are resolved relative to the manifest; legacy absolute paths
also work. The helper additionally records the supplied compact source path/hash.
Keep original producer/conversion provenance separately: a hash does not certify
author identity, capture consent or redistribution rights.

All NPZ files in the selected directory must match the manifest exactly. Every
invocation checks derived hashes and declared frame counts. Source hashes are
optional checks for diagnostics with `--verify-sources`; ordinary loading remains
portable when original source paths are unavailable. No corpus count or total
frame count is hardcoded.

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
The released asynchronous probe assumes exclusive GPU compute access; upstream
warns of crashes when it shares a GPU with another job. For an explicitly
shared GPU, use `--eprobe_async 0` and record that override. Do not run a separate
GPU evaluator concurrently with asynchronous training probes.

```bash
python scripts/reinforcement_learning/train.py --algorithm blinkmimic \
  Cyclo-BlinkMimic-K1-Rev1 --motion-dir .blinkmimic-data/example \
  --run-dir logs/blinkmimic/run-001

# Headless all-clip, first-frame evaluation with the official evaluator.
python scripts/reinforcement_learning/play.py --algorithm blinkmimic \
  Cyclo-BlinkMimic-K1-Rev1 --motion-dir .blinkmimic-data/example \
  --run-dir logs/blinkmimic/eval-001 \
  --checkpoint logs/blinkmimic/run-001/ckpt_003500.pt
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

## Policy, EMA checkpoint and external export contract

The checkpoint key `pi` is the deployed EMA policy; `pi_live` is the optimizer's
live policy. Use `pi` for inference/export. Periodic light checkpoints contain
that policy, while final `last.pt` also stores world model, critic, optimizer and
control state. Replay, simulator state and RNG are not saved, so resume is warm.

The original implementation is `blinkmimic.networks.Policy`, whose forward
method takes history and goal separately. An external flat wrapper must use
raw float32 **[B, 1420]**: ten observations **[10, 81]**, oldest first/newest last,
followed by ten future goal blocks **[10, 61]**. History is maintained externally;
on reset, repeat the initial observation ten times and set previous action to
zero. Both history and goal are normalized **inside** the Policy:
`(x - mean) / sqrt(var + eps**2)`, with `eps=0.01` and frozen checkpoint buffers.
There is no normalization clipping or external second normalization.

Each observation is `[q-q0-joint_offset (23), qdot (23), R_actual^T*omega_world (3),
-R_actual[2,:] (3), previous final action (23), sixd(R_actual^T*R_ref[f]) (6)]`.
`TrackingSim.observe` does not apply the separate `mdp.obs_scale` helper:
observation qdot and gyro are raw, without generic pre-scales.

For k=1…10, j=min(f+k,T−1), each goal is `[H_ref[f]^T*(p_ref[j]-p_ref[f]) (3),
sixd(H_ref[f]^T*R_ref[j]) (6), R_ref[j]^T*v_world[j] (3),
R_ref[j]^T*omega_world[j] (3), q_ref[j] (23), 0.1*qdot_ref[j] (23)]`.
Here H is current **reference** yaw and velocity blocks use the full **future
reference** rotation. Its projected x axis uses `fx=R00`, `fy=R10`,
`n=sqrt(fx*fx+fy*fy+1e-9)`, `c=fx/n`, `s=fy/n`. Quaternion order is wxyz and
normalization uses `torch.nn.functional.normalize` with epsilon 1e-12; root
linear velocity is at the link origin. Six-dimensional rotation is the first two columns flattened in
row-major order: `[R00,R01,R10,R11,R20,R21]`.

The decoder consumes `[latent_z (32), normalized flattened history (810)]`.
Its output has the reference skip added using the **raw** first-future goal:
`action = decoder_residual + (goal[15:38] - q0) / action_scale`.
The **[B, 23] output is the final normalized action**, not residual-only or
radians. Nominal PD target is `q0 + action_scale * action`, equivalently
`q_ref[f+1] + action_scale * decoder_residual`. Do not add the reference twice.
Joint order, q0 and action scale must match the same K1 task/checkpoint.
The nominal evaluation joint offset is zero; a nonzero simulated joint offset
is added to the PD target and subtracted in the observation as shown above.

This PR supplies training/evaluation and documents the export contract; it does
not add an ONNX export command, ONNX runtime, robot controller or deployment
assets. Existing PPO exporters are not interchangeable. Any future FP32 export
of a BF16-autocast policy must disclose that precision choice and establish
numerical parity separately before claiming compatibility.

## Validation

```bash
PYTHONPATH=.:third_party/supermimic/src:integrations/supermimic/src \
  python -m pytest tests -q
PYTHONPATH=.:third_party/supermimic/src python -m pytest third_party/supermimic/tests -q
python tests/blinkmimic/validate_native.py --motion-dir /absolute/path/to/data \
  --output /absolute/path/to/new-diagnostic
```

The optional native diagnostic uses one environment per clip and full-sized
networks. It initializes every clip at its first frame, takes eight vector steps,
and checks one-step Muon gradients, finite values, world-model freezing during
the actor update, and checkpoint save/reload. It is not the complete horizon-32
learner acceptance, a trained baseline or evidence of convergence. This command
performs bounded learner updates and needs an idle GPU; it is not part of the
CPU-only publication tests.

Unit input tests use synthetic references and cover portable path relocation,
clip order/boundaries, input hashes, named joints and float32 overflow rejection.
The existing BeyondMimic/SuperMimic regression suites remain separate.

Software licenses do not grant rights to reference motions, private datasets,
trained weights or exported policies. Keep those assets outside Git and review
their own terms. No robot execution or deployment is provided by this task.
