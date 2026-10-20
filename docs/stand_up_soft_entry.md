# AiSapiens StandUp SoftEntry v2

Task: `Cyclo-Mimic-K1-Rev1-UMR-StandUp-AiSapiens-SoftEntry`.
Warm start: `2026-10-19_10-15-20_umr_stand_up_ai_sapiens/model_9999.pt`.

The unchanged ai_sapiens controller ramps ReadyPose targets/gains to policy targets/gains using a 0.3 s smoothstep. Reference indices advance at 50 Hz, and the previous raw action is observed. ReadyPose command positions must remain separate from settled measured positions under gravity. No ai_sapiens C++/ROS changes are required by this task.

The original reference starts at pelvis height 0.117544 m while the nominal deployment ReadyPose settles at 0.079937 m, with up to 0.105916 rad command-to-measured joint deflection. This mismatch is a plausible contributor to the remaining initial body jump; reference correction alone has not established causal sufficiency.

`build_ai_sapiens_soft_entry_motion.py` builds a separate CSV/NPZ pair from nominal reset-bank state 0. The first 0.3 s is the settled pose; 0.3–1.5 s uses quintic interpolation and shortest-path quaternion slerp into the original trajectory at its original time. CSV frames at/after 1.5 s are unchanged. All NPZ body kinematics are regenerated using the deployment model snapshot. The clip remains 560 frames at 50 Hz. The JSON manifest contains input/output hashes and FK round-trip errors.

Full-start environments retain the 64-state settled reset bank, with 75% of training environments assigned to full starts. The actual actuator handoff still starts from the original ReadyPose COMMAND, not the settled reference joint positions. Partial starts bypass the handoff and entry penalties.

New penalties measure pelvis and torso at each 2 ms substep, including the final substep before reward computation. Each policy interval uses its peak. Reset history is invalidated to exclude teleportation acceleration. Velocity errors use matching COM reference quantities; positive height overshoot uses link positions. Penalties have full weight through 1 s and fade to zero at 1.5 s.

| Quantity | Allowed excess | Normalization | Weight |
|---|---:|---:|---:|
| Vertical velocity error | 0.10 m/s | 0.15 m/s | -0.30 |
| Angular velocity error | 0.60 rad/s | 0.80 rad/s | -0.20 |
| Vertical acceleration magnitude | 3 m/s² | 5 m/s² | -0.05 |
| Angular acceleration magnitude | 15 rad/s² | 25 rad/s² | -0.05 |
| Height above reference | 0.015 m | 0.030 m | -0.30 |

Each squared normalized excess is capped at 25. Existing target-rate, torque-rate, foot-clearance, impact and trajectory-tracking rewards remain enabled. These thresholds/weights are initial training choices, not validated deployment acceptance limits.

The v2 runner loads v1 actor/critic weights with fresh optimizer, iteration and curriculum. It resumes v2 checkpoints only with a matching reference hash. Learning rate is fixed at 5e-5, 512 environments, 10,000 iterations, checkpoint every 250 iterations. Logs copy the new CSV into params; deployment must distribute the corresponding model AND new CSV/config together.

Validation before promotion must compare identical settled ReadyPose starts and the existing 0.3 s deployed blend against v1: first 1.5 s torso/pelvis vertical and angular speed, height overshoot, peak floor contact force, foot clearance; and full-clip stand-up success/stability. Training starting successfully is not evidence that the remaining jump has been solved. Keep existing deployed assets until that comparison passes.
