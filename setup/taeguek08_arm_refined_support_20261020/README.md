# Taegeuk 8 arm-refined reference: fresh support training

Source: final Taegeuk 8 motion in “UMR로 HIFAR 모션 리타게팅” (see manifest.json).
The 60 Hz source is resampled to 50 Hz without added smoothing or speed changes.
Training trajectory: 2519 frames; last frame 50.36 s. The paired CSV belongs to this NPZ.

Target: robotis@192.168.60.177, /home/robotis/workspace_kes/cyclo_mjlab, container cyclo_mjlab, GPU 0.
4096 environments, 30000 PPO updates, resume=False, no checkpoint, default scratch optimizer (adaptive learning rate 0.001).

Rewards preserve base tracking and add SupportRefinement: reference-aware contact loss, long unloading, repeated liftoff, excess sole slip/rotation, support-foot height and upward departure. Reference kicks and jumps are excluded from support penalties. Flight-height tracking stays enabled. Geometry penalties ramp from 20% to 100% over the first 1000 PPO updates. This entry point does not use old reference timing focus windows or entry-state banks.

Episode limit 52.38 s covers the full motion. 20% environments reset at frame zero; remaining starts use adaptive sampling with 20% uniform mixture.

Run `python3 setup/taeguek08_arm_refined_support_20261020/launch.py` on the target host only. A launch.json guard refuses duplicate launches; detached docker exec survives SSH disconnect. Do not remove launch.json to rerun an active job.

Validation: converted CSV/NPZ joint agreement <1e-6; finite trajectory; FK round-trip max position 2.377e-7 m and orientation 2.397e-7 rad. GPU PPO smoke validation and actual launch status are recorded separately.

## Actual run

Started 2026-10-20 15:31 KST on remote GPU 0.

Remote training output: `/home/robotis/workspace_kes/cyclo_mjlab/logs/rsl_rl/k1_mimic/2026-10-20_06-31-32_taeguek08_arm_refined_support_scratch_20261020_063125`

Console log: `/home/robotis/workspace_kes/cyclo_mjlab/setup/taeguek08_arm_refined_support_20261020/taeguek08_arm_refined_support_scratch_20261020_063125.log`

Validation: 3-update / 128-environment smoke completed; 68 checkpoint tensors finite; six support geometry behavioral tests passed. Main run confirmed 4096 environments, 30000 iterations, resume false, no loaded checkpoint; model_0.pt has iter=0, common_step_counter=24, finite actor/critic parameters.

At 2026-10-20T15:33:08.430112+09:00, 103 updates completed. Recent 50 iterations averaged 0.752 s. Initial ETA around 21:50–22:10 KST on October 20, subject to changing episode length, throughput and checkpoint overhead. This is a runtime estimate, not a completion guarantee or motion-quality validation.
