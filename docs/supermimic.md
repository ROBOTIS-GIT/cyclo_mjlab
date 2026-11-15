# SuperMimic on K1 (MuJoCo research runtime)

This integration uses the actual HybridRobotics SuperMimic differentiable world
model, replay buffer, supervised dynamics learner, imagination rollouts and
BPTT/actor-critic policy learners. It does not route to Cyclo's PPO imitation
runner. Both `Cyclo-SuperMimic-GenCoord-K1-Rev1` and
`Cyclo-SuperMimic-MaxCoord-K1-Rev1` use Cyclo's 23-joint K1 Rev.1 MJCF,
actuator gains/action scales, and 14 tracked bodies.
The research robot adapter copies the licensed Cyclo asset settings and supplies
explicit collision masks/default priorities required by mjlab 1.6; the main
asset configuration is unchanged. Root/anchor is pelvis.
The original multi-encoder motion/velocity architecture and exporter remain
available through the upstream runner configuration.

## Independent environment

Cyclo main uses Python 3.11 / mjlab 1.2.0. SuperMimic requires Python 3.13 and
research APIs, including FK-based named-joint motions and sim-context features.
Do not install the integration into the existing Cyclo environment or container.
From the repository root, with uv installed:

```bash
git submodule update --init third_party/ai_sapiens
uv sync --project integrations/supermimic --python 3.13 --locked
```

This creates `integrations/supermimic/.venv`; root `pyproject.toml` is unchanged.
SuperMimic source is vendored at commit
`abc124c1bd5976341034668af3d12c087f2f1f55`.
`mjlab_research` is pinned to
`dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd` and resolved in the integration lock.
See `third_party/supermimic/UPSTREAM.md` for modifications and attribution.
There is no checkout `.agents/skills` or `AGENTS.md` at the fetched base.

## Data

Generate normal Cyclo body-state NPZ with `scripts/tools/motion/csv_to_npz.py`
in the normal Cyclo environment, then convert in the isolated environment:

```bash
uv run --project integrations/supermimic cyclo-supermimic convert \
  source/assets/motions/K1_rev1/dance1/dance1.npz \
  .supermimic-data/dance1.npz
```

The converter requires explicit body/joint names, uses the pelvis world-frame
state, reorders joints to the K1 MJCF hinge order, validates finite shapes,
normalized **wxyz** root quaternions and 50 Hz control rate. It refuses to
overwrite source/output files. Resample other rates with the existing CSV
converter first. Research motions contain `fps`, `joint_names`, `joint_pos`,
`joint_vel`, `root_pos_w`, `root_quat_w`, `root_lin_vel_w`, `root_ang_vel_w`.
Body states are reconstructed by upstream differentiable FK. G1 motion data
cannot be used directly for K1; retarget it first. No licensed dataset or
pretrained checkpoint is bundled. The smoke's temporary standing clip tests
plumbing only and is not a training dataset.

## Train and play

Check GPU occupancy before running. These are user-run training commands,
not claims of a completed trained policy. Use a free GPU and an appropriate
number of environments for its memory:

```bash
WANDB_MODE=offline uv run --project integrations/supermimic cyclo-supermimic train \
  Cyclo-SuperMimic-GenCoord-K1-Rev1 --motion-file .supermimic-data/dance1.npz \
  --num-envs 64 --runner.run-name k1-dance1
```

Swap `GenCoord` for `MaxCoord` to use body-space dynamics. The original runner
selects the representation from the task ID, exposes all nested runner/WM/
policy settings through tyro, and writes `logs/<task>/<timestamp>/` relative
to the working directory. `--runner.actor-obs proprio` selects deployable
sensor inputs; default `rep_state` uses privileged state. Offline W&B avoids
remote logging but preserves local results; omit it for normal authenticated
W&B tracking. Local `--motion-file` disables the upstream G1 registry default.

```bash
uv run --project integrations/supermimic cyclo-supermimic play \
  Cyclo-SuperMimic-GenCoord-K1-Rev1 --motion-file .supermimic-data/dance1.npz \
  --wandb-path ENTITY/PROJECT/RUN --viewer viser
```

Playback uses the upstream W&B bundle loader (`world_model.pt`, `actor.pt` and
serialized runner config), so the displayed run must have been trained on the
matching K1 task/representation. Offline runs need W&B sync before this viewer
path can retrieve them. Existing Cyclo PPO checkpoints are incompatible.
This integration does not authorize real-robot deployment or claim the existing
PPO `sim2real.yaml` contract applies to a SuperMimic actor.

## Short verification

```bash
uv run --project integrations/supermimic python -m pytest tests/supermimic -q
uv run --project integrations/supermimic python -m pytest third_party/supermimic/tests -q
uv run --project integrations/supermimic cyclo-supermimic smoke --device cpu
uv run --project integrations/supermimic cyclo-supermimic smoke --device cpu --representation maxcoord
```

Smoke uses two environments and a temporary standing clip, takes a real sim
step, fits one dynamics update to that transition, and verifies finite nonzero
actor gradients through imagined dynamics. It is deliberately bounded and does
not start the million-iteration upstream training default. Test results and
machine-specific limitations are recorded in `docs/supermimic-validation.md`.

## K1 maximal-coordinate IK change

Upstream handles two/three-hinge chains; K1's waist is one hinge. Its axis is
completed with an orthogonal auxiliary axis, and padded coordinates are discarded
by the existing permutation. Numerical tests compare all 23 recovered angles
and rates against MuJoCo FK and check gradients. This is the only algorithm
source change in the vendored upstream tree. K1 task configuration lives outside
that tree and leaves the normal Cyclo registration untouched.
