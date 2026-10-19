# Corrected Koryo reference (2026-10-15)

The shared Koryo train/play configuration now selects
`source/assets/motions/K1_rev1/koryo_support_v2/k1_koryo_support_v2_50hz_converted.npz`.
This also applies to Performance, Curriculum, ContactV2, and SupportV3-Max variants
that inherit the Koryo configuration.

The source is `k1_koryo_support_v2_50hz.csv` in the same directory: a headered UMR
CSV with time, root XYZ, root quaternion WXYZ and 23 named joint angles. It has
4,293 frames at 50 Hz (last timestamp 85.84 s). The NPZ is generated from this CSV
using the training model. It is not the retargeter's qpos-only NPZ.

Regenerate inside the `cyclo_mjlab` container, from `/workspace/cyclo_mjlab`:

```bash
python scripts/tools/motion/umr_csv_to_npz.py \
  --input_file source/assets/motions/K1_rev1/koryo_support_v2/k1_koryo_support_v2_50hz.csv \
  --input_fps 50 --output_fps 50 --smooth-window 1
```

Conversion adds no smoothing or time scaling. The accompanying `_converted.csv`
is the converter's numeric 30-column format, with quaternion XYZW and no time
column; use the headered source above for regeneration.

The correction preserves foot rotation relative to the root while reducing
excess support-foot translation and resolving training-model self-collisions.
Joint-speed exceedances remain; physical feasibility is not established by this
kinematic correction.

The original `koryo/k1_koryo_converted.npz` is retained. For evaluations that must
reproduce an older run, explicitly select that reference instead of the new
default. This configuration change takes effect when a new train/play process
loads the task; it does not change an already-loaded trajectory or start training.
