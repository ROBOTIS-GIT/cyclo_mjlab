# Direct 50 Hz reference reuse: implementation and validation

Date: 2026-11-15. Repository `/home/hc/Documents/Codex/k1-walking/cyclo_mjlab`;
worktree `/home/hc/Documents/Codex/2026-11-15/task-3/feature-supermimic`;
branch `feature-supermimic`. This extends native integration at
`f1d4dcf3ccdfff6a99c749ea783e07e762cf4cb1`.

## Final user contract

The same existing Cyclo CSV-converter NPZ is accepted directly by both learners:

```bash
# Existing Python 3.11 / mjlab 1.2 Docker runtime
python scripts/reinforcement_learning/train.py --algorithm beyondmimic \
  Cyclo-BeyondMimic-K1-Rev1-Dance1 \
  --motion-file source/assets/motions/K1_rev1/dance1/dance1.npz

# Research Python 3.13 runtime
uv run --project integrations/supermimic python scripts/reinforcement_learning/train.py \
  --algorithm supermimic Cyclo-SuperMimic-GenCoord-K1-Rev1 \
  --motion-file source/assets/motions/K1_rev1/dance1/dance1.npz
```

`MaxCoord` works identically. SuperMimic play accepts the same original NPZ with
its existing `--wandb-path` argument. Native train/play and the compatibility
`cyclo-supermimic train|play` CLI share the adapter. The standalone conversion
command is optional, not a prerequisite. A comma list or quoted glob resolves
all clips; unmatched patterns are rejected, clip boundaries are retained.

Body-state input: unique named K1 joints and all 25 named MJCF bodies, consistent
finite arrays, at least two frames, normalized wxyz quaternions, 50 Hz. Joint/body
column permutations are handled by names. World pose and twist compatibility
is checked on every frame/body against K1 FK and supported velocity points.
No resampling, smoothing, automatic quaternion permutation or source overwrite.
Explicit conflicting backend/frame/point metadata, unknown models/body sets,
inconsistent velocities or ambiguous different root-origin twists are errors.

Root-only inputs must declare `backend=mujoco`, `quaternion_order=wxyz`,
`velocity_frame=world`, `velocity_point=link_origin`. This includes new optional
converter output and internal normalized files. Older unannotated root-only
files have insufficient convention evidence and must be re-exported from the
complete original body reference rather than guessed.

Normalized files are an internal original-loader bridge under
`.supermimic-data/adapter/`. Source bytes, ordering and model kinematic/COM signature
identify the derived group. Its manifest records source paths and SHA-256 values.
Originals are revalidated every invocation; a previous cache never replaces
validation. Normalized files/manifests are atomically replaced; temporary files
do not match the loader's `*.npz` pattern. All are git-ignored.

## Velocity mismatch: root cause and resolution

The initial draft incorrectly tried validating body twist only against the
current K1 COM model. A separate diagnostic using the **same XML and NPZ** under
both MuJoCo 3.5 (existing Docker) and 3.16 (research runtime) returned the same
max errors: treating velocities as body-origin gives 1.28856857; using current
COM gives 0.0403430154 (at frame 196, right knee). Thus this was not a MuJoCo
version difference.

`git log --follow` shows the committed references originate at `874c7bd`, whose
K1 XML was then stored at `source/assets/robots/robotis_k1/xmls/k1.xml`. Current
assets have revised inertial/COM properties. The old reference pairs link-origin
poses (`xpos`/`xquat`) with COM linear velocities (`mj_objectVelocity(BODY)`).
The adapter includes the **25 exact historical COM offsets** from that tracked
XML in `source/tasks/supermimic/reference_profiles.py`, with source revision,
ROBOTIS copyright and Apache-2.0 notice. It does not replace or modify the robot's
simulation model, inertias, actuator settings, reward, or randomization.

For each supported origin/COM profile, recover the root-origin twist using
`v_origin = v_com - omega × (R * com_offset)`, then use MuJoCo body Jacobians to
check all named body angular/linear velocities with the file's named joint rates.
Only profiles whose maximum residual is <= 2e-4 are accepted. If several match,
their derived root-origin velocities must also agree within 1e-5. Pose tolerance
is 2e-4 m, sign-invariant quaternion Euclidean tolerance 2e-4. The original 0.04
mismatch was **not** accepted by widening tolerances.

Producer/backend labels are retained as declared or `unspecified`. Geometry and
velocity checks validate a physical convention; they do not authenticate an
unknown producer's identity. Explicit non-MuJoCo backend declarations are rejected
rather than silently transformed. These profiles derive only from existing Cyclo
assets; the learner remains the official HybridRobotics SuperMimic implementation.

## Actual data evidence

| Input | Frames / Hz | Matching profile | Maximum velocity residual |
| --- | --- | --- | --- |
| committed Dance1 | 300 / 50 | legacy K1 `874c7bd` COM | 8.697726e-7 |
| committed Dance2 | 299 / 50 | legacy K1 `874c7bd` COM | 5.553136e-7 |
| fresh existing-converter Dance1 output | 300 / 50 | current K1 COM | 8.697726e-7 |

Committed source SHA-256 values before **and after** adaptation:
- Dance1: `96d216da809bb08ddd6308cde052aed7650a7b13a6a3bb85b2a09a39fa79e9dc`
- Dance2: `aec71b4e754516af9a02be74a7d72b3e50c41c6ecbf7ce1d0c8cdd259a913dde`

The fresh output was produced by the existing, unmodified converter in a
new disposable **CPU-only**, network-disabled `cyclo-mjlab:latest` container:

```bash
python scripts/tools/motion/csv_to_npz.py \
  -f source/assets/motions/K1_rev1/dance1/dance1.csv \
  --input_fps 50 --output_fps 50 \
  --output_name logs/reference-adapter-validation/current-dance1.npz
```

Converter FK round-trip: position 6.102e-8 m, orientation 2.292e-7 rad.
Original CSV/NPZ files were not overwritten. Raw evidence:
`logs/reference-adapter-validation/{current-producer.log,data-evidence.json}`.

## Tests and execution

Research test suite command:

```bash
integrations/supermimic/.venv/bin/python -m pytest \
  tests/supermimic third_party/supermimic/tests -q
```

Tests include real Dance1/2 source-hash preservation and actual research
MotionLoader/FK loading, reordered joints/bodies, current/legacy COM compatibility,
wrong quaternion order/backend/frame/names/point, corrupt velocities, FPS errors,
missing root metadata, multi-clip boundary preservation, and train/play CLI
adaptation before upstream invocation. Ruff, compileall and diff checks accompany
these tests. Final result: **249 passed in 5.87 s**, one upstream torch.jit deprecation
warning. Full log: `logs/reference-adapter-validation/tests-final.log`.

Actual native training used the **original Dance1 NPZ**, not a user-converted
file, sequentially for GenCoord and MaxCoord. Other work occupied the GPU, so
`CUDA_VISIBLE_DEVICES=''`, `--device cpu`, `WANDB_MODE=offline`,
`TORCH_COMPILE_DISABLE=1`, `OMP_NUM_THREADS=2` were used. Two environments,
three outer iterations, two-step WM/policy horizons, batch size two, one batch,
small `(32,32)` dynamics/actor, `(32,)` critic and save interval 1:

```bash
# Repeat for REP=GenCoord and REP=MaxCoord, in research runtime.
REP=GenCoord
python scripts/reinforcement_learning/train.py --algorithm supermimic \
  Cyclo-SuperMimic-${REP}-K1-Rev1 \
  --motion-file source/assets/motions/K1_rev1/dance1/dance1.npz \
  --device cpu --num-envs 2 --runner.max-iterations 3 --runner.buffer-size 8 \
  --runner.wm.horizon 2 --runner.wm.batch-size 2 --runner.wm.num-batches 1 \
  --runner.policy.horizon 2 --runner.policy.batch-size 2 --runner.policy.num-batches 1 \
  --runner.dynamics-hidden-dims '(32,32)' --runner.actor-hidden-dims '(32,32)' \
  --runner.critic-hidden-dims '(32,)' --runner.log-interval 1 \
  --runner.save-interval 1 --runner.run-name direct-reference-adapter
```

| Result | GenCoord | MaxCoord |
| --- | --- | --- |
| Exit status | 0 | 0 |
| Completed outer iterations | 3 | 3 |
| Wall time | 10.68 s | 7.95 s |
| Final WM loss | 0.25791 | 0.20804 |
| Final policy gradient norm | 0.5355 | 0.6142 |
| Checkpoints saved and finite on reload | WM/actor/critic | WM/actor/critic |

Artifact roots:
- `logs/Cyclo-SuperMimic-GenCoord-K1-Rev1/2026-11-15_12-04-32/`
- `logs/Cyclo-SuperMimic-MaxCoord-K1-Rev1/2026-11-15_12-04-40/`

The existing BeyondMimic alias/config test passed again in its existing Docker
runtime (`python tests/beyondmimic/test_task_aliases.py`). Additionally the old
`Cyclo-Mimic-K1-Rev1-Dance1` command trained on the same original NPZ using CPU,
2 environments, 1 PPO iteration, 4 rollout steps, one minibatch/epoch:

```bash
python scripts/reinforcement_learning/train.py Cyclo-Mimic-K1-Rev1-Dance1 \
  --gpu-ids None --motion-file source/assets/motions/K1_rev1/dance1/dance1.npz \
  --env.scene.num-envs 2 --agent.max-iterations 1 --agent.num-steps-per-env 4 \
  --agent.algorithm.num-mini-batches 1 --agent.algorithm.num-learning-epochs 1 \
  --agent.run-name reference-adapter-beyond-regression
```

**Passed, exit 0**, checkpoint/YAML and PPO metrics saved under
`logs/rsl_rl/k1_mimic/2026-11-15_03-09-50_reference-adapter-beyond-regression/`.
No existing BeyondMimic source, converter, actuator or reward implementation was
changed. Raw logs are in `logs/reference-adapter-validation/`, git-ignored.
No GPU rerun was needed for this data-contract change; previous CUDA validation
is retained separately. No other process/container was stopped, no hardware or
long training was run, and nothing was pushed/merged. Interactive W&B playback
and SuperMimic deployment export remain outside the executed checks.

## Publication state checked before completion

`git ls-remote --heads origin main feature-supermimic` returned only remote main
`e659f69af227d244a4edab0bb9228e7aeaadef45`; the feature branch is local only.
GitHub REST `pulls?state=all&head=ROBOTIS-GIT:feature-supermimic` returned `[]`.
No PR exists for that head at this check, and no push/merge/PR publication was
performed. This report records a local implementation awaiting separate review.
