# Taegeuk 2–5 SupportV1 scratch training

Independent tasks: `Cyclo-Mimic-K1-Rev1-Taeguek02-SupportV1` through `Taeguek05-SupportV1`. Original tasks, reference files, PPO settings, observation/action interfaces, disturbances, tracking rewards and termination rules are preserved. New tasks add the `performance_feet` world-frame foot-floor force-history sensor and one `taeguek_support_quality` reward.

## Support phase eligibility

The existing sole-height/vertical-motion support mask is refined using reference data only:

- Retain a low foot when its slowest low sole material point moves below 0.20 m/s, or the opposite reference foot is more than 0.15 m above its own minimum height.
- The latter retains supporting pivots during kicks. A low, fast-moving foot while both feet are low is excluded from contact-maintenance penalties.
- Trim 40 ms around internal eligibility boundaries and discard retained intervals shorter than 100 ms. Clip endpoints are not trimmed.
- Rotation is included in material-point velocity. Sphere surface points within 5 mm of the lowest surface form the candidate patch. Eligibility never uses actual robot speed, preventing escape from the penalty by moving faster.

These are conservative geometric confidence labels, not ground-truth contact annotations. Slow intended steps and changing contact patches can remain ambiguous. All chapter labels are saved under setup/taeguek_support_v1_20261016/labels_*.pt.

## Cost

Reuse FootSupportQuality: missing support 0.45, long unload 0.15, unexpected liftoff 0.02/event, excess sole slip 0.15, excess rotation 0.05; outer weight -1.0. Single-support periods receive 1.5×. Strength grows from 20% to 100% over 24,000 environment steps (not PPO updates). Reset/landing grace and 10 ms contact debounce are retained. Reference velocity allowances preserve intended pivots/sliding. No reference CSV/NPZ was modified.

## Sequence

Inside server02's existing container, from /workspace/cyclo_mjlab:

```bash
NUM_ENVS=4096 MAX_ITERATIONS=30000 GPU_ID=0 \
  bash scripts/reinforcement_learning/train_taeguek_02_to_05_support.sh
```

Chapters 02, 03, 04, 05 train in order as four independent fresh policies (`--agent.resume False`, no checkpoint). A failure stops the sequence. Logs: logs/sequences/taeguek_02_to_05_support_*. Policy run names: taeguekNN_support_v1_scratch_*. The GPU 1 chapter 6–8 sequence is separate.

Checks:

```bash
python -m unittest discover -s tests -p test_taeguek_support.py -v
CUDA_VISIBLE_DEVICES=0 python scripts/reinforcement_learning/check_taeguek_support.py
bash scripts/reinforcement_learning/train_taeguek_02_to_05_support.sh --dry-run
```

All four integration checks use 32 environments and 64 policy steps. Chapter 2's 23.66–24.24 s left support is retained throughout; its 9.74–9.90 s left relocation and 7.22–7.52 s right relocation are excluded. A separate 2-update PPO training checks optimization and checkpoint output before replacing the old sequence. These checks verify execution, not reduced microsteps. Evaluate later exports against the existing full-motion audit, especially left support near 23.8 s and 27 s, and also compare tracking and intended steps/kicks.

Registration backup and validation logs: setup/taeguek_support_v1_20261016. Old training logs/checkpoints are retained. The new run starts from random weights as requested; it does not load the old model.
