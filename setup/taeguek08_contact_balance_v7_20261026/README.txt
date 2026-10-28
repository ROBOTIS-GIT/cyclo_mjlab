Taegeuk 8 ContactBalanceV7 — 2026-10-26

Manual training preparation. The V6 process was interrupted at the user's
request. V7 main training is NOT started by this preparation.

Starting checkpoint
Run: 2026-10-26_00-27-10_taeguek08_contact_precision_v6_finetune
Checkpoint: source/checkpoint/model_13500.pt
SHA256: 22d4b584d1817916b768be4bf98e11a5e30da96212359b3fbf27a75e485257f2
Saved iter=13500, completed updates=13501, common_step_counter=324024.
Actor and critic load strictly; V7 uses a fresh optimizer and a new counter.
Source params, CSV and inference exports are retained under source/.
This is the user's latest stopped policy, not a claim that it dominates all
older checkpoints. The prior full-motion audit found residual tracking errors.

Changes and rationale
1. Preserve V6's mature support and total-GRF objectives. The extra V6 drift
   term now holds displacement error relative to the recorded reference,
   rather than holding a competing world position while pose rewards request
   reference movement. Actual contact loss still cannot reset its anchor.
2. Add reference-gated support for both phase 1 and phase 2. No yaw gate is
   imposed. Reference lowest-material-point speed, support labels, height and
   an 80 ms reference phase grace define eligibility; actual contact cannot
   turn the term off. Runtime positive coverage is 100% of the target windows
   7.88–8.54 s right, 41.82–42.92 s right, 32.4–33.4 s left.
3. Penalize slip of the slowest low material point to permit a planted pivot.
   Speed allowance is 5 mm/s + reference speed capped at 10 mm/s; normalization
   is 20 mm/s. Include a small 0.02 BW substep contact floor, extra sole height,
   and reference-relative roll/pitch tilt. Do not demand a fixed yaw.
4. Add a path-recovery cost with no exponential saturation. A 3 cm allowance
   and 15 cm normalization keep the recovery gradient at large XY errors.
   Existing pelvis XY reward remains 1.3.
5. Only the first landing receives extra braking in 6.90–7.55 s. The right-foot
   speed envelope uses sqrt(0.35^2 + 2*6*height); actual descent beyond that and
   additional pelvis descent beyond reference+0.05 m/s are penalized. The
   improved 38 s landing keeps inherited braking/GRF terms.
6. Arm joint recovery weight 1.8 -> 2.2, hand-position weight 1.2 -> 1.4.
   Wrist joint relative weight 1.5 -> 2.0, elbow stays 1.5. Preserve hold damping
   instead of broadly smoothing actions or suppressing fast arm motion.
7. New V7 terms ramp over 2000 PPO updates. Fixed learning rate 3e-6 (V6 5e-6),
   entropy 0.00015 (V6 0.0003). 4096 environments, 20000 additional V7 updates.
   Finite quality regression warnings do not halt; invalid states remain errors.

The motion CSV/NPZ, PD, observations, action scaling, physics and reset noise
use the stopped V6 setup. No retargeting or motion-phase relabeling was done in
this revision. This addresses reward target conflicts and missing pivot support;
selective contact-constrained retargeting remains a separate option if needed.

Validation
28 behavior tests: reference-gate exclusions, pivot inclusion, small slip,
contact-loss non-evasion, normal load transfer, first-landing-only braking,
large-error path recovery, reference-relative anchor, and inherited behavior.
GPU full-strength validation: 64 environments and six target starts, finite
metrics, weighted-cost identity, target coverage, caching and reset. Exported
sim2real config equals the stopped V6 source exactly.
4096-environment smoke + resume verifies checkpoint/optimizer continuity. Read
validation.json and validation_full.json for completed results. Smoke is not
policy-quality evidence; evaluate full motions after substantive training.

Run inside the existing cyclo_mjlab container
cd /workspace/cyclo_mjlab
MPLCONFIGDIR=/tmp/t8-v7-mpl WARP_CACHE_PATH=/workspace/cyclo_mjlab/setup/taeguek08_landing_support_v5_20261023/warp_cache python -u setup/taeguek08_contact_balance_v7_20261026/launch.py 0 --num-envs 4096 --allow-shared-gpu

The shared-GPU flag is explicit because other user applications currently use
GPU 0. It leaves those processes running and requires >=10 GiB free at launch.
The prior 4096-env V6 training used about 6.3 GiB; this free-memory check cannot
reserve memory against future allocations by another application.

If V7 is interrupted, append --resume with a V7 main model checkpoint. The
launcher verifies revision, environment count, counters and smoke/main identity,
restores the optimizer, and computes updates remaining out of the V7 20000.
An existing V7 main run blocks accidental fresh launch; use its checkpoint.

Required outcome evaluation
Compare source model_13500, V6 model_10000/model_13000, old V5.1 seed/final and
deployed V4 using the full 50.36 s reference, original sole and native PD.
Use both reference reset and standing transition starts. Track GRF peaks and
excess impulse, small sole speed, unloading, 7.9/32.5/42 s support, arm tracking
and pelvis path together. No V7 motion improvement is claimed before that test.
