# SuperMimic provenance and local changes

- Official source: https://github.com/HybridRobotics/supermimic
- Pinned revision: `abc124c1bd5976341034668af3d12c087f2f1f55`
- Copyright 2026 The SuperMimic Authors; Apache-2.0.
- Original [LICENSE](LICENSE) and [README acknowledgements](README.md#acknowledgements)
  are preserved byte-for-byte. That revision has no `NOTICE` file to copy.

Changes to the vendored distribution:

| File | Local change |
| --- | --- |
| `src/supermimic/utils/analytic_ik.py` | Support K1 single-hinge chains with an orthogonal auxiliary axis; marked at file top and at the changed block |
| `pyproject.toml` | Replace editable sibling `../mjlab_research` with immutable research Git revision `dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd`; marked at file top |
| `src/supermimic/runners/supermimic_runner.py` | Save the final iteration, include native manifest when present, await final artifact future before cleanup |
| `src/supermimic/utils/wandb_artifacts.py` | Keep local artifact logging but skip remote enumeration/pruning in offline/disabled mode |

Other vendored Python sources, tests and original README remain unchanged.
Native `source/tasks/supermimic/train.py` adapts the official launcher, retains
its author/license notice, and adds the K1 reference/control-rate guard while
reusing the original configuration, environment factory and learner. Native
`checkpoint.py` adds declarative inference metadata through a runner subclass;
`play.py` reconstructs the matching native environment and uses the original
world-model/actor builders and viewer classes.
The K1 robot adapter and historical COM facts derive from existing Cyclo assets
with their original notices retained; see [component scope](../../THIRD_PARTY_LICENSES.md).

The upstream README acknowledges code inspired by ControlVAE (BSD-3-Clause),
robotic_world_model (Apache-2.0), and DreamerV3 (MIT). Those acknowledgements are
retained. This is not a claim that every inspired algorithm is copied code, or
an independent clearance of the upstream authors' line-by-line provenance.
No files from those repositories were separately imported by this integration.
