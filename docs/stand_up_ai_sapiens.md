# StandUp fine-tuning for the unchanged ai_sapiens controller

Task: `Cyclo-Mimic-K1-Rev1-UMR-StandUp-AiSapiens`.
The existing ReadyHandoff task is preserved. Deployment code is not modified.

## Entry contract

- Load the existing 124-observation / 23-action actor and 256-observation critic.
- Start full attempts from 64 physically settled, supine ReadyPose states. Sample 0 is deterministic nominal playback; training uses perturbed starting orientations/joints settled for 2.5–4 seconds. These are simulated states, not hardware recordings.
- Use all 23 ReadyPose targets and PD gains captured from the current local ai_sapiens configuration. The settled measured angles can differ from the commanded ReadyPose target (maximum approximately 0.106 rad in this bank).
- Advance reference immediately at 50 Hz. Use forward-difference reference joint velocities, matching the deployment CSV loader.
- Blend ReadyPose command and gains into the actor target with `s*s*(3-2*s)` over 0.3 s. Apply interpolation every 2 ms physics step; the deployed command publisher is 1 kHz, so its intermediate 1 ms commands are not modeled separately.
- Preserve the previous raw actor output in last_action. Reset raw history and transition clock on both termination and clip-end resets.
- Reserve 75% of training worlds for frame-zero entry, 25% for intermediate recovery/preparation starts; playback uses frame-zero entry in all worlds.

## Physics and losses

The snapshot in `source/assets/robots/ai_sapiens_entry/` includes current deployment XML and meshes, contract, manifest and settled reset bank. Compared to the laptop's existing model, the deployed XML has different inertial values and hip collision geometry. It is loaded only by this new task. Training uses 2 ms physics / 20 ms policy, 100 solver iterations, 50 line-search iterations, contact dimension 3, priority 0 and friction 1. Position actuators preserve the existing policy gains, action scales and offsets.

Home-offset and torso-COM perturbations are disabled for this initial simulator-matching task; material randomization is fixed at 1. Intermediate reset noise and later pushes remain, with pushes excluded during the first 1.5 seconds. This is not a hardware robustness validation.

Entry losses penalize target changes, torque changes, body-floor impact and a conservative sole-volume proximity proxy. Four spheres per foot form the proxy; it is not an exact mesh-intersection test or a replacement for physical collision geometry. The extra penalties apply for the first 1.5 seconds of full starts. The reference motion file itself is unchanged.

## Warm start and resume

Old ReadyHandoff checkpoints load actor/critic weights only, reset optimizer and curriculum counters, and start at iteration zero. New checkpoints carry `ai_sapiens_entry_version: 1`; resuming those restores optimizer/curriculum and starts at the next iteration. Initial learning rate is fixed at 1e-4. Default run: 512 worlds, 10,000 additional iterations, checkpoint every 250 iterations.

```bash
cd /home/robotis-insu/cyclo_mjlab
bash scripts/reinforcement_learning/train_ai_sapiens_stand_up.sh
```

To resume a new-task checkpoint, set `AI_SAPIENS_TRAIN_CHECKPOINT` to its path and invoke the same script. `--agent.max-iterations` is the number of additional iterations. Do not start a second copy while the current job is running.

Training params include `sim2real.yaml`, `ai_sapiens_entry.yaml`, agent/environment settings and the reference CSV. Export the selected checkpoint with the existing play/export command after evaluation; deployment does not need a new controller wrapper.

## Validation on 2026-10-19

- 3 new CPU contract tests and 6 existing ReadyHandoff tests passed.
- 64-world, 3-iteration warm-start training completed with finite losses.
- Resume from new-task `model_2.pt` ran iterations 3–32; metadata/optimizer resume worked.
- 8-world CUDA inference from the ORIGINAL checkpoint ran 600 policy steps, checking all first-0.3-second alpha values, advancing reference, raw history, reset history, and finite observations/rewards. All 8 nominal full clips met the existing completion criterion. This deterministic nominal check is not evidence that fine-tuning has removed the observed deployment foot collision.
- A separate full run was launched from original `model_29999.pt`; smoke checkpoints were not promoted as the training seed.

```bash
PY=/home/robotis-insu/miniconda3/envs/k1_rl_mjlab/bin/python
$PY -m unittest discover -s tests -p 'test_ai_sapiens_entry.py' -v
$PY -m unittest discover -s tests -p 'test_ready_handoff.py' -v
$PY scripts/reinforcement_learning/validate_ai_sapiens_entry.py --checkpoint CHECKPOINT
$PY scripts/reinforcement_learning/build_ai_sapiens_ready_bank.py
```

After training, compare old/new policies in unchanged ai_sapiens from the same settled ReadyPose starts. Measure actual foot geometry separation, peak target/torque changes, body impacts and full-motion success across several starts. CPU MuJoCo and GPU MuJoCo Warp can still differ; a successful training rollout alone is not sufficient.

## Active run

- Started 2026-10-19 10:15:20 KST; initial PID 3671103.
- Logs/checkpoints: `logs/rsl_rl/k1_mimic/2026-10-19_10-15-20_umr_stand_up_ai_sapiens/`.
- Console: `work/ai_sapiens_entry/training_20261019.log`.
- Registry backup: `work/ai_sapiens_entry_backup/registry_before.py` (contains all pre-existing user changes).
- Existing viewer was left running. Local ai_sapiens and robotis_mujoco files were not edited.
