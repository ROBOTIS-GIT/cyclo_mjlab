# Like Jennie policy comparison

The baseline is the existing downloaded ONNX and its original config/reference,
copied into `baseline_original`. The original policy SHA-256 starts `a1817e33`;
the original reference NPZ SHA-256 starts `faa7341d`. The fixed comparison
reference is `source/assets/motions/K1_rev1/like_jennie_early_plus10/`.

The existing policy was evaluated in two conditions:

- `results/baseline_original_reference`: original policy with original motion.
- `results/baseline_updated_reference`: original policy with the edited motion.

For a fair baseline-versus-candidate comparison, run both on the **same updated
CSV and NPZ** and compare against `baseline_updated_reference`. The original
reference run is a historical diagnostic; comparing its numbers directly to
the new policy would conflate the policy and reference changes.

Both use the same MuJoCo model, starting pose (first frame, zero velocity),
0.3 s action transition, 50 Hz policy, 500 Hz physics, and no randomization or
external disturbances. The trajectory evaluator uses each policy's matching
`sim2real.yaml`; its observation/action order and dimensions are checked. The
previously displayed policy rollout used a different local MJCF file; these
comparison outputs use the repository's `third_party` model for both policies.

After training and exporting a candidate, use the policy and `sim2real.yaml`
from that **same run**. The example requires a shell environment with
`mujoco`, `onnxruntime`, `numpy`, `scipy`, and `pyyaml`:

```sh
python setup/like_jennie_comparison/evaluate.py \
  --policy /absolute/run/exported/policy.onnx \
  --config /absolute/run/params/sim2real.yaml \
  --csv source/assets/motions/K1_rev1/like_jennie_early_plus10/like_jennie_50hz.csv \
  --npz source/assets/motions/K1_rev1/like_jennie_early_plus10/like_jennie_50hz.npz \
  --output setup/like_jennie_comparison/results/candidate_run_name

python setup/like_jennie_comparison/compare.py \
  --baseline setup/like_jennie_comparison/results/baseline_updated_reference \
  --candidate setup/like_jennie_comparison/results/candidate_run_name \
  --output setup/like_jennie_comparison/results/candidate_run_name_comparison.json
```

The comparison checks reference, robot XML, physics rate, frame count, and
rollout completion before producing full/early/middle/late metrics. It records
joint and arm tracking RMSE, world-frame pelvis XY and foot position RMSE,
pelvis height/tilt, non-foot ground contact, contact-force peaks, and the share
of physics samples with loaded contact-point speed above 0.15 m/s. This speed
is an instantaneous simulation diagnostic, not a directly measured slip rate;
low pelvis height or high tilt can be intentional choreography.

**Time alignment:** the original frames 46–67 were retimed to 46–77. Earlier
frames correspond directly, later original frame `i` corresponds to updated
frame `i+10`, and the interior maps linearly to the original phase. The
revised early arm/leg trajectories differ from the original in addition to
this retiming. An actual robust comparison should also repeat with varied
friction, initial conditions and disturbances; the current baseline is one
nominal run and does not prove hardware safety.
