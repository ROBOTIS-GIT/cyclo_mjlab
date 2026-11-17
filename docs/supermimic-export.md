# SuperMimic policy export with external motion goals

The common export entry point writes a policy-only ONNX graph. It reuses the
shared core's `_SingleEncoderOnnx` and `_export_module`, selecting one goal
encoder, its normalizer, and the shared observation normalizer/decoder. It omits
the upstream motion-clip wrapper, lookup tables, time-step input, world model,
critic, and motion outputs. Training, policy mathematics and network widths are
unchanged.

```bash
python scripts/reinforcement_learning/export.py --algorithm supermimic \
  Cyclo-SuperMimic-GenCoord-K1-Rev1-001 \
  --checkpoint-dir /path/to/local-bundle --output-dir /path/to/new-export
# Equivalent module entry point; MaxCoord is supported too:
python -m source.tasks.supermimic.export Cyclo-SuperMimic-MaxCoord-K1-Rev1-001 \
  --checkpoint-dir /path/to/local-bundle --output-dir /path/to/another-new-export
```

A bundle contains `manifest.json`, `actor.pt`, and `world_model.pt` from the
current native task. The manifest must completely specify the runner and match
the task, observation groups, robot fingerprint and environment. The actor is
reconstructed with its saved mode, history, latent and hidden widths and loaded
strictly. The dynamics file is read only to verify the trained affine action
buffers against the native action contract; no dynamics network or simulator is
constructed. Missing/unknown configuration, incompatible shapes or invalid
normalizer buffers produce errors. There is no checkpoint conversion or fallback
to guessed settings. Output directories must be new.

`policy.onnx` has one `obs` float32 input and one `actions` float32 output, with
fixed batch size 1 (the existing shared exporter contract). The default motion
encoder is selected explicitly for the entire inference batch. `--encoder
velocity` selects the existing body-twist encoder instead; metadata records its
training slot fraction, which may be zero. Export does not establish training
quality for either encoder.

## Inputs and outputs

For the public 23-joint K1 task:

| Saved actor configuration | Input width, motion encoder | Output width |
| --- | --- | --- |
| proprio, history 10 | 10 × 75 + 520 = **1270** | 23 |
| GenCoord rep_state, history 1 | 56 + 520 = **576** | 23 |
| MaxCoord rep_state, history 1 | 214 + 520 = **734** | 23 |

Other supported proprio history lengths use the recorded length. `rep_state`
uses history 1, as required by the learner. Both observation modes export
normally; the graph and metadata retain the trained mode. Changing training
defaults does not turn a rep_state policy into a proprio policy. A consumer must
supply the exported mode's actual observations.

The proprio frame is `[projected_gravity(3), IMU_gyro(3), q-q0+encoder_bias(23),
qdot-qdot0(23), previous_raw_action(23)]`. Gravity is world down expressed in the
actual root frame; gyro uses the configured pelvis IMU site's axes. Positions
are radians and velocities rad/s, without extra gyro/qdot pre-scales. Metadata
records the IMU site orientation, q0 and qdot0.

Flatten history in **term-major** order: all H oldest-to-newest gravity frames,
then all H gyro frames, then all H position, velocity and previous-action frames.
Append the raw goal once. The graph converts term-major input into temporal
`[1,H,75]` history internally. On reset, zero the previous action and repeat the
first valid observation H times. `pack_input(history, goal, term_widths)` provides
the corresponding Python packing operation; temporal `history.flatten(1)` is a
different order when H > 1.

GenCoord rep_state is `[root_z, gravity_body(3), root_linear_velocity_body(3),
root_angular_velocity_body(3), absolute_joint_position(23), joint_velocity(23)]`.
MaxCoord rep_state uses the 14 tracked bodies in metadata order:
`[root-relative_position(42), root-relative_rot6d(84), body_linear_velocity_in_root(42),
body_angular_velocity_in_root(42), root_z, gravity_body(3)]`. Each body's world
velocity is rotated into the root frame; the root velocity is not subtracted.
These modes require additional state estimates beyond the proprio sensor vector.

The motion goal is **520** raw features:

1. Ten consecutive 46-wide blocks `[absolute_q_ref(23), raw_qdot_ref(23)]`.
2. Ten consecutive 6-wide relative-root-orientation blocks.

For native command index t, use `min(t + 5*k, clip_exclusive_end - 1)` for k=0…9.
This window never crosses a clip boundary. Resetting physics to source frame f
sets native `motion.time_steps=f+1`; the consumer must preserve this reference
clock alignment rather than silently treating t as the physics reset frame.
Each orientation is `sixd(R_actual_current_root.T @ R_reference_window_root)`.
It depends on the **actual current root**, not the current reference yaw.
Sixd is `[R00,R01,R10,R11,R20,R21]` and input quaternions are normalized wxyz.
No q0 subtraction, qdot scaling or anchor-translation block belongs in this actor
goal. `motion_goal(bundle, motion, frames, actual_root_pose_w)` calls the original
training goal extractor; the external motion object provides joint_pos,
joint_vel, root_pos_w, root_quat_w and exclusive motion_ends arrays. It does not
load, resample or differentiate a file. No particular external storage format is
required by the graph.

Normalization stays **inside ONNX**: `(x - saved_mean) / (saved_std + 0.01)` for
both history and goal. Saved `_std` buffers are used directly; do not recompute
sqrt(var), use BlinkMimic's eps-squared formula, clip features, or normalize twice.
The selected encoder's output is bounded with tanh and concatenated before the
normalized temporal history for the shared decoder.

Output is the raw 23-wide decoder action, with **no reference-action skip** and
no graph action clipping. Nominal targets are `offset + scale * actions` in the
recorded native joint order. Native simulated encoder bias is subtracted from
the physical target. The previous-action observation contains the raw returned
action, not the target in radians. A consumer must reproduce its trained action
contract; adding q_ref or an unrecorded action clamp changes the policy.

## Metadata and runtime integration

The graph embeds `cyclo.policy_contract`; `metadata.json` carries the same
contract plus the output ONNX SHA256. The export also includes `NOTICE.txt` and
`licenses/` with the Apache-2.0 text and the installed RSL-RL BSD-3-Clause text
for its traced normalizer. Keep these files with redistributed export bundles;
their paths/hashes are recorded in metadata. The existing Isaac Lab BSD notices
remain in the source repository; FK/IK math is not traced into this policy-only
graph. Framework/runtime binaries and development tools are not bundled.
These software notices do not grant rights to training data or learned weights.

The contract records task/encoder, mode, ordered
observation terms and offsets, history/goal layout, normalization, affine action
values, body/joint order, physical fingerprint, source file hashes and shared
exporter revision/hash. No source directory, checkpoint tensors or motion arrays
are copied into the JSON. The graph contains the selected policy weights only.

The checked-in public AI Sapiens source is pinned at
`611a070fad83d2a9ef9c1bc82213988450cf471c`. Its generic observation manager supports
term histories and its policy runtime loads ONNX inputs, but its legacy
`MotionReference` path reads root xyz + xyzw quaternion + joint positions,
re-differentiates velocities and interpolates playback. That path does not build
the exact SuperMimic strided 520-feature goal. Matching 1270/23 dimensions alone
does not establish that a runtime's ordering, reference clock, root/IMU frames,
normalization and action processing match this contract.

A runtime adapter must preserve source qdot, clip boundaries and training frame
alignment, pack the terms above, and use the matching q0/scales/gains. Existing
robot selector mappings, hardware calibration and per-robot file bundles are
outside this exporter. No robot configuration or running service is modified by
export. Integration against a particular ai_sapiens_sim2real development checkout
and a trained model requires its own source/asset comparison and loading checks.

## Validation

The locked environment includes ONNX. Install CPU `onnxruntime` separately for
parity checks. Export always runs ONNX full checker validation.

```bash
python -c 'import onnxruntime'
PYTHONPATH=.:third_party/supermimic/src \
  python -m pytest tests/supermimic/test_policy_export.py -q
```

Tests use synthetic weights and motions, full default network widths, and
nontrivial mean/std buffers. They compare the original multi-encoder actor with
the shared flat slice exactly, then ONNX Runtime within `rtol=2e-4, atol=2e-5`.
They cover GenCoord/MaxCoord, proprio/rep_state, external-goal frame ordering and
clip clamps, actual-root orientation, term-major packing, and incompatible bundle
rejection. These are CPU policy-equivalence checks, not trained-model or physical
robot validation.
