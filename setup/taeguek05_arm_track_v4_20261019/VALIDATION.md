# ArmTrackV4 validation (2026-10-19)

Reward revision: `28a9f7a9dbd8ab5f5624c85845482fa9f205e72e37fb66f2a83853f101cc2cc7`.

- Six CPU behavior tests passed: large-error recovery, exact fast-reference motion,
  independent rigid-frame invariance, reset/frame-jump/cache handling, reference hold
  gating and catch-up movement, and source-task/leg isolation.
- The original Compose actor and critic load exactly; the first optimizer is fresh.
- CPU integration: 16 environments, 3 PPO updates followed by a fresh
  environment/runner loading the saved checkpoint and completing one more update.
- Completed updates 3 → 4, environment counter 72 → 96, Adam steps 60 → 80.
- 2050 additional policy steps at the full new-reward strength, with
  finite rewards and all arm diagnostics; maximum continuous reference age
  40.30 s.
- Smoke checkpoints are excluded from the production launcher.

These are readiness checks, not evidence that arm vibration or tracking has improved.
The production 10,000-update V4 run has not been started by this preparation.

## Laptop integration

- NVIDIA RTX 5070 Laptop GPU (8 GB), PyTorch 2.13.0+cu130, mjlab 1.2.0,
  rsl-rl-lib 5.0.1, MuJoCo 3.5.0, Warp 1.12.0.
- Same reward revision `28a9f7a9dbd8ab5f5624c85845482fa9f205e72e37fb66f2a83853f101cc2cc7`.
- 1024 GPU environments: source actor/critic exact load, 3 updates plus
  a fresh-runner 1-update resume, Adam steps 60 → 80; 256 additional
  rollout steps, finite diagnostics and rewards. Smoke excluded by launcher.
- During the check, nvidia-smi showed total GPU memory use about 1.9 GB with
  the existing 276 MB stand-up viewer still running. This is a sampled use,
  not a guaranteed peak or a reservation for future workloads.
- Existing cyclo_mjlab container failed NVML/CUDA initialization; a separate
  cyclo_mjlab_taeguek05_v4 container from the same image restored GPU access.
  No existing container or viewer was stopped.
- Code is in a separate clone under ~/cyclo_mjlab/work/taeguek05_arm_v4_20261019.
  The checkpoint was transferred separately and its SHA256 matches.
- Laptop available disk space was approximately 3 GB during preparation.
