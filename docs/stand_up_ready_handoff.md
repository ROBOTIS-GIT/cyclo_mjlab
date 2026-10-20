# StandUp ReadyPose handoff training

Task: `Cyclo-Mimic-K1-Rev1-UMR-StandUp-ReadyHandoff`.

This is a separate variant of StandUp SoftLanding. The original task and motion
are unchanged. It trains a **provisional** lying ReadyPose controller handoff,
not the complete hardware WalkReady routine. No training is launched by setup.

## Controller and assumptions

ReadyPose uses all 23 joint angles from frame 0 of
`source/assets/motions/K1_rev1/umr_stand_up/k1_0008_stand_up_soft_landing_converted.npz`.
The body starts in that frame's lying orientation/position with the existing
initial-state randomization. It does not start standing.

Reported hardware ReadyPose gains are used only where known:

| Joints | Ready Kp | Ready Kd | Policy Kp | Policy Kd |
| --- | ---: | ---: | ---: | ---: |
| Knees | 150 | 4.867 (assumption) | 76.452 | 4.867 |
| Shoulder pitch/roll/yaw, elbows | 40 | 10 | 22.301 | 1.420 |
| Hips, ankle pitch, waist | 76.452 (assumption) | 4.867 (assumption) | 76.452 | 4.867 |
| Ankle roll, wrists | 22.301 (assumption) | 1.420 (assumption) | 22.301 | 1.420 |

Unreported gains fall back to the policy gains; they are **not verified hardware
values**. Gain overrides are named by joint regex in `ReadyHandoffActionCfg`.
No additional gain randomization is applied in this first version.

The fixed 30% frame-zero training group executes:

1. ReadyPose hold: minimum 0.4–0.8 s, randomized per attempt. Physics advances
   under ReadyPose PD while the reference stays at frame 0 with zero reference
   velocities. Initial pose and residual velocity DR are retained.
2. Stability check: maximum joint error <0.2 rad, maximum joint speed <0.5 rad/s,
   torso angular speed norm <0.4 rad/s for 0.1 s. This checks settling, not upright
   orientation. Failure to settle within 3 s terminates as `ready_timeout`.
3. Linear target and Kp/Kd handoff: 0.6–1.0 s, randomized per attempt. Policy
   inference continues; frame 0 and zero reference velocities remain held.
   `target=(1-alpha)*ready_target+alpha*policy_target`, with the same alpha for
   gains. Both alpha=0 and alpha=1 are applied at control steps.
4. Motion starts after the full-policy handoff step. Regular policy gains apply.

The other 70% retain intermediate-start training (including preparation-window
sampling). Playback runs the same controller for every reset, with 0.6 s minimum
hold and 0.8 s handoff. Ready timeout, early falls and clip-end teleports all
reset handoff history. Each environment has independent clocks and gains.

Pushes are withheld during ReadyPose/handoff. The existing success-gated reset,
friction and push curriculum remains active during motion. Full attempts include
ReadyPose and handoff failures. Training episode limit is 35 s. Actor/critic
observation sizes stay 124/256, so existing SoftLanding weights remain loadable.
No phase observation was added; phase timing is controlled by the external handoff
controller. `last_action` remains the preceding **raw policy output**, not the
blended joint target.

## Rewards and metrics

Existing SoftLanding rewards remain. During preparation, handoff, and the first
0.3 s of motion, add:

- Applied target change penalty: mean squared delta / 0.05 rad, weight -0.2.
- Actual actuator torque change between control steps: mean squared delta /
  10 Nm, weight -0.05. The first sample after reset initializes history; it is
  not compared to the previous attempt.
- Pelvis/torso ground-force peak penalty: squared excess above 420 N per body,
  weight -0.2, using physics-substep history. This threshold is a reward tuning
  parameter, not a hardware safety limit. Original pelvis impact reward remains.

Each normalized squared penalty is capped at 25. Torque-change logging samples
control steps; contact-force peaks include physics substeps.

TensorBoard command metrics: `handoff_alpha`, `handoff_phase` (0 ready, 1 handoff,
2 motion), `handoff_ready_timeout`, `handoff_target_delta_rad`,
`handoff_torque_delta_nm`, `handoff_peak_target_delta_rad`,
`handoff_peak_torque_delta_nm`, `handoff_body_peak_force_n`, plus existing
`full_motion_success_rate` and `difficulty_stage`.
These use the framework's episode averaging; transition metrics are diluted by
other environments and later motion. Peak target/torque buffers persist per
attempt, not as population percentiles. Use dedicated rollout traces for a final
before/after impact comparison. Success rate is first-frame training attempts,
not a held-out hardware success rate.

## Start training (inside the container)

Recommended first experiment: initialize from the completed SoftLanding policy.
The runner loads actor and critic weights but resets optimizer, iteration count,
and difficulty to stage 0 when loading an old checkpoint. Use a smaller initial
learning rate for this additional training (the adaptive schedule can change it).

```bash
cd /workspace/cyclo_mjlab
python scripts/reinforcement_learning/train.py \
  Cyclo-Mimic-K1-Rev1-UMR-StandUp-ReadyHandoff \
  --checkpoint logs/rsl_rl/k1_mimic/2026-10-02_09-38-04_umr_stand_up_soft_landing_scratch/model_29999.pt \
  --env.scene.num-envs 2048 \
  --agent.resume False \
  --agent.algorithm.learning-rate 0.0001 \
  --agent.run-name umr_stand_up_ready_handoff \
  --agent.max-iterations 10000
```

For entirely fresh training, omit `--checkpoint`, set initial learning rate to
`0.001`, and use `--agent.max-iterations 30000` with a different run name.
To resume this new task's checkpoint, pass it through `--checkpoint`; marked
ReadyHandoff checkpoints restore optimizer/curriculum and continue numbering.
`max-iterations` is the number of additional learning iterations for that run.

Playback uses the new task too:

```bash
python scripts/reinforcement_learning/play.py \
  Cyclo-Mimic-K1-Rev1-UMR-StandUp-ReadyHandoff \
  --checkpoint-file /workspace/cyclo_mjlab/logs/rsl_rl/k1_mimic/NEW_RUN/model_XXXX.pt \
  --num-envs 1 --viewer viser --export-policy False
```

## Deployment contract

At training startup, `params/ready_handoff.yaml` records all 23 ReadyPose targets,
ReadyPose and policy gains, timing ranges, thresholds, and observation convention.
The checkpoint also stores this specification. The ordinary `sim2real.yaml`
contains policy gains as before.

**ONNX does not contain this handoff controller.** Deploying the ONNX alone with
the old moving-reference transition will recreate a training/deployment mismatch.
The real controller must implement the same settling gate, reference/velocity
hold, target/gain interpolation, raw-action history and motion-release order.
The YAML is a specification; no existing sim2real parser/controller was modified
or claimed to consume it automatically. On real hardware, a ReadyPose settling
timeout should prevent policy entry, rather than reproduce simulation reset.

Verify the unreported gains before hardware validation. Compare old/new policies
under the same controller, friction and seeds using transition peak forces,
actual torque changes, joint speeds, and complete stand-up rate. Passing smoke
tests verifies wiring; it does not establish reduced real-robot impact.
