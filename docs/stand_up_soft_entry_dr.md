# SoftEntry with restored original StandUp randomization

Prepared 2026-10-20. Setup only; training was not started.

## Scope

New task: `Cyclo-Mimic-K1-Rev1-UMR-StandUp-AiSapiens-SoftEntry-DR`.
Existing SoftEntry and other tasks are unchanged. The new task inherits the
settled 0.3 s motion prefix, quintic join at 1.5 s, five early dynamics penalties,
0.3 s controller handoff, Ready bank, and deployment-matched 2 ms / 20 ms timing.
The existing entry push protection is retained.

Training restores exactly these events from the original SoftLanding task:

- `physics_material`: matching robot and floor friction; the existing curriculum
  chooses stage 0: 0.8–1.2, stage 1: low 0.4–0.6 with probability 0.3,
  stage 2: low 0.3–0.6 with probability 0.5, stage 3: low 0.2–0.6 with
  probability 0.7. The regular branch is 0.8–1.2. `curriculum_command` overrides
  the static ranges written in event params. Resampling is restored on every
  new attempt, including motion completion that does not reset the environment.
- `joint_home_offset_noise`: startup encoder bias uniformly ±0.01 rad.
- `torso_com_offset_noise`: startup COM offset x ±0.025 m, y/z ±0.05 m.

These are per-environment calibration/model variations. Play mode remains the
original deterministic SoftEntry setup; it is not a randomized robustness test.

## Original StandUp audit

The deployed `umr_stand_up_v2/params/env.yaml` and the laptop's original
SoftLanding/base configuration have friction, encoder bias, torso COM noise,
and existing velocity-push perturbations. They do not have payload/body-mass,
actuator-gain/strength/response, or observation/command-delay randomization.
Actuator gains/effort limits are fixed, and observation delay min/max are zero.
Ordinary observation noise, reset velocity noise, and push perturbations are not
actuator lag or payload randomization. COM offset changes mass location, not mass.

Per the user's request, no new payload, actuator-response or latency DR is added.

## Start later

On the laptop:

```bash
cd /home/robotis-insu/cyclo_mjlab
bash scripts/reinforcement_learning/train_soft_entry_dr.sh
```

Defaults: 512 environments, 10,000 iterations, fixed learning rate 5e-5.
Warm-start checkpoint:
`logs/rsl_rl/k1_mimic/2026-10-19_15-03-14_umr_stand_up_ai_sapiens_soft_entry/model_9999.pt`.

The v2 checkpoint supplies actor/critic weights; optimizer, iteration and
curriculum start fresh at iteration 0 / stage 0. In particular, the finished
v2 run's iteration 9999 and no-DR curriculum are not resumed as the new run.
Outputs use the new `umr_stand_up_ai_sapiens_soft_entry_dr` run name.

To resume a DR checkpoint instead:

```bash
SOFT_ENTRY_DR_CHECKPOINT=/absolute/path/to/new_dr/model_250.pt \
  bash scripts/reinforcement_learning/train_soft_entry_dr.sh
```

DR checkpoints use entry version 3 and preserve the DR specification. Matching
v3 checkpoints restore optimizer/curriculum and continue at the next iteration;
mismatched motion or v3 DR contracts are rejected. Existing v2 checkpoints and
original runners are not edited. No model is deployed by this launcher.

## Setup verification (no training)

```bash
/home/robotis-insu/miniconda3/envs/k1_rl_mjlab/bin/python -m unittest \
  tests.test_soft_entry_dr tests.test_soft_entry tests.test_ai_sapiens_entry tests.test_ready_handoff -v
bash -n scripts/reinforcement_learning/train_soft_entry_dr.sh
bash scripts/reinforcement_learning/train_soft_entry_dr.sh --help
```

These inspect config/checkpoint migration and run CPU unit tests only. Actual GPU
rollouts and learning have deliberately not been launched as part of setup.
Randomization restoration is not itself proof of real-robot success: validate
entry motion and the later torso-lift segment against the original policy after
training.
