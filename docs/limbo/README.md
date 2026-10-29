# Limbo — standard BeyondMimic

`feature-limbo` uses the existing Cyclo MJLab K1 Rev.1 Mimic environment and PPO without limbo-specific training overrides. The only task-specific environment setting is the 120cm motion NPZ path, following the Dance1 task pattern.

## Train from scratch

On the local PC:

```bash
docker exec -it -w /workspace/cyclo_mjlab cyclo_mjlab bash
```

Inside the container:

```bash
python scripts/reinforcement_learning/train.py Cyclo-Mimic-K1-Rev1-Limbo-1p2m \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 30000 \
  --agent.experiment-name k1_limbo_1p2_basic \
  --agent.run-name bar120_beyondmimic_scratch
```

There is no checkpoint argument; the actor, critic and optimizer start fresh. Training has not been started by this reset. Output is written to `logs/rsl_rl/k1_limbo_1p2_basic/<run>/`.

## Defaults retained

- Standard K1 model, actuator settings, joint soft limits and direct position actions.
- Actor: original 124 channels; critic: original privileged observation terms.
- Original six tracking rewards and all original regularization costs.
- Original failure-based motion-frame sampling, reset noise, material/COM/encoder randomization and interval pushes.
- Standard PPO: learning rate 1e-3 adaptive, clip 0.2, entropy 0.005, 5 epochs, trainable Gaussian std initialized to 1.0, MLP 512/256/128 ELU.
- Physics timestep 5ms, policy timestep 20ms, episode limit 30s, standard motion-end resampling.

## Motion and cleanup

The selected CSV has 376 frames at 50Hz (7.5s between first and last frames). The added 2s preparation prefix is removed. The NPZ is converted from the original CSV using the current default K1 model and the existing `csv_to_npz.py`; source poses and timing are preserved. See `conversion.json` beside the NPZ for hashes and verification.

Original height-specific motion data and Desktop CSV copies remain available as source data. Legacy preparation/reset-bank assets, limbo-specific MDP/actions/PPO/runners, teacher/replay/warm-start checkpoints, custom robot model, old evaluator/launcher scripts and local V6 experiment outputs are removed. Their tracked code remains recoverable through prior Git commit `ca4a7ab`.

There is no physical bar, limbo-specific success gate, preparation gate or action blending in this task. Tracking metrics and episode length follow the standard Mimic implementation.
