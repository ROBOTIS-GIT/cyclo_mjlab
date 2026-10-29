# Like Jennie · 50 Hz

`like_jennie_source.csv` is the reviewed, early-motion-smoothed reference with
10 extra interpolated frames. Its SHA-256 is
`5889806581a7b1d1f1cebca1e14a6e77e785689ab8c7a069094cf450912a70a9`.
`like_jennie_50hz.npz` and `like_jennie_50hz.csv` were produced together by
`scripts/tools/motion/csv_to_npz.py` at 50 Hz (2,620 frames). The NPZ is the
training reference; the matching converted CSV is for subsequent export.

Task: `Cyclo-Mimic-K1-Rev1-LikeJennie-Taeguek04Rewards`. It reuses the reward
terms and weights from the selected Taegeuk 4 sidekick stability run. Its
sidekick-specific left-support penalty retains the original 14.4–14.85 s
window, which was designed for Taegeuk 4 and has not been validated for this
dance. The PPO settings use the standard from-scratch defaults; the selected
Taegeuk run's low learning rate and checkpoint were for fine-tuning.

Run in the existing Cyclo MJLab container:

```sh
python scripts/reinforcement_learning/train.py \
  Cyclo-Mimic-K1-Rev1-LikeJennie-Taeguek04Rewards \
  --env.scene.num-envs 4096
```

The 16-environment, one-iteration wiring smoke run confirmed initialization,
reward construction, rollout, and PPO update; it is not a trained policy or a
performance evaluation. Evaluate the dance, support masks, and the inherited
time-window reward before relying on a full run or transferring to hardware.
