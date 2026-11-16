# Third-Party Licenses

This project uses third-party open-source software and includes code adapted
from third-party open-source projects.

## mujocolab/mjlab

- Source: https://github.com/mujocolab/mjlab
- Runtime version: 1.6.1
- Retained 1.2.0 formulas: `foot_height` in
  `source/tasks/velocity/mdp/observations.py` and `feet_clearance` in
  `source/tasks/velocity/mdp/rewards.py`, preserving the existing flat-ground
  observation and reward contract.
- License: Apache-2.0
- Used as: The core simulation and reinforcement-learning framework

```text
Copyright 2025, The mjlab Developers
```

MJLab and cyclo_mjlab are both distributed under the Apache License 2.0. The
license text shared by both projects is provided in `LICENSE`.

## HybridRobotics/whole_body_tracking

- Source: https://github.com/HybridRobotics/whole_body_tracking
- License: MIT
- Used in:
  - `source/tasks/mimic`
  - `scripts/tools/motion/csv_to_npz.py`

The mimic task stack and motion-conversion workflow adapt reference-motion
tracking components from `whole_body_tracking`.

```text
Copyright (c) 2024, The Isaac Lab Project Developers.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## HybridRobotics/SuperMimic

- Official algorithm source: https://github.com/HybridRobotics/supermimic
- Official base: `abc124c1bd5976341034668af3d12c087f2f1f55`.
- Shared fork: https://github.com/ROBOTIS-GIT/supermimic, gitlink revision
  `10dd5dd3f02601caf1991790952a25db63a6fb99`.
- Copyright 2026 The SuperMimic Authors; original Apache License 2.0 retained at
  [third_party/supermimic/LICENSE](third_party/supermimic/LICENSE).
- Portable math extracted from official `mjlab_research` retains its Isaac Lab
  BSD-3-Clause attribution and [license](third_party/supermimic/licenses/IsaacLab-BSD-3-Clause.txt).
  The shared package declares `Apache-2.0 AND BSD-3-Clause`.
- [PATCHES.md](third_party/supermimic/PATCHES.md) records the minimal core changes
  and primary-source revisions. Its README retains acknowledgements to ControlVAE
  (BSD-3-Clause), robotic_world_model (Apache-2.0), and DreamerV3 (MIT).

## Native adapter provenance and NOTICE scope

`source/tasks/supermimic/forward_kinematics.py` preserves the implementation of
`HybridRobotics/mjlab_research` revision
`dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd:src/mjlab/utils/forward_kinematics.py`.
`native_motion.py` adapts that revision's tracking command/termination interfaces
to stable MJLab 1.6.1, including FK-backed body state, concatenated clip boundaries
and external reset sampling. It adds strict compact-input validation, stable
manager reset dispatch and pending-forward handling. The original Apache license
is retained at [LICENSE.mjlab_research](source/tasks/supermimic/LICENSE.mjlab_research),
and source/copyright/modification notices are in both files. Research MJLab is
not a dependency of this native runtime.

The official SuperMimic and research MJLab revisions contain Apache-2.0 licenses
and no tracked NOTICE file; none was removed. The fork preserves the original
license and README attribution, with explicit setup corrections for optional
research dependencies. Core modifications are recorded in its patch record.

`source/tasks/supermimic/train.py` retains the SuperMimic Authors copyright and
Apache identifier for its adapted launcher. `robot.py` imports the existing
Cyclo asset and retains the ROBOTIS/Insu Park notice. `reference_profiles.py`
identifies its exact historical Cyclo XML source (`874c7bd`) and Apache notice.
The optional checkout compatibility package declares Apache-2.0 and includes its
license copy when built. These checks preserve notices; they do not guarantee
title or non-infringement. Apache redistribution conditions are in
[section 4 of the license](https://www.apache.org/licenses/LICENSE-2.0).

## Robot assets, reference data and weights

| Material | Checked scope | Remaining limits |
| --- | --- | --- |
| K1 MJCF, meshes and robot assets | Existing `ROBOTIS-GIT/ai_sapiens` submodule at `611a070fad83d2a9ef9c1bc82213988450cf471c`; its Apache-2.0 LICENSE is byte-identical to `source/assets/robots/robotis_k1/LICENSE.ai_sapiens` | Upstream license declaration was checked; independent CAD/mesh authorship and third-party permissions were not audited |
| Historical COM profile | Numeric offsets from tracked Cyclo `874c7bd:source/assets/robots/robotis_k1/xmls/k1.xml`; source/copyright recorded in `reference_profiles.py` | This profile changes reference interpretation only, not asset ownership or simulation inertias |
| Dance1/2 CSV and NPZ | Pre-existing `source/assets/motions/K1_rev1` files, unchanged by this feature; used for compatibility tests | No dataset-specific provenance, capture-consent or separate redistribution-permission record was found with these files; the root software license alone does not document those facts |
| Trained/pretrained weights | No `.pt`, `.pth`, `.onnx` or `.ckpt` files are tracked or added by the feature; local smoke outputs stay ignored | User-provided W&B bundles and external datasets/weights were not inspected or cleared by this review; their terms are separate from the code license |

The existing references remain because they are used by both tasks and tests.
Their uncertain provenance is reported rather than treating them as newly
licensed data or deleting existing user assets. No new dataset or external model
weights were fetched or redistributed as part of this integration.

## Installed dependency licenses

The common Python 3.12 runtime resolves stable MJLab and the shared core in
`requirements.lock`; local core source is installed from the pinned submodule.
The optional compatibility project's lock uses the same native dependency set.
Installed distributions retain their bundled licenses. The following declarations
were checked for the integration; they are not a full binary redistribution audit:

| Dependency | Declared license scope |
| --- | --- |
| SuperMimic core; MJLab; MuJoCo/MuJoCo Warp | Apache-2.0 (core math also BSD-3-Clause); package license files retained by their distributions |
| Warp | Apache-2.0 plus bundled component license files, including CUDA-related notices |
| RSL-RL | BSD-3-Clause plus dependency notices |
| TensorDict, Viser, W&B | MIT declarations |
| PyTorch | Composite expression including Apache/BSD/MIT/ISC/Zlib and LLVM-exception components; not merely this repository's Apache license |
| NVIDIA CUDA runtime and cuDNN wheels | `LicenseRef-NVIDIA-Proprietary`, with their own `License.txt` files |
| pytest | MIT; development dependency of the environment project |

These metadata checks are not an exhaustive license compatibility or binary
redistribution audit. In particular, redistributing a built container, vendored
wheels, CUDA libraries, downloaded data or trained weights requires reviewing
the actual artifacts and their bundled notices/terms. Upstream acknowledgements
are preserved separately from verified copied-source notices; they do not by
themselves establish that third-party code was copied or that all downstream
rights have been independently cleared.

## BlinkMimic

`third_party/blinkmimic` vendors Fuyi Chen's official BlinkMimic at
`d264658d6c9ce61fb52f9f0f8985a369f3ccc409` from
https://github.com/fuyichen0725/blinkmimic under MIT. Its LICENSE and
THIRD_PARTY_NOTICES.md retain the Fuyi Chen, InstantTrack / Huy-Quang-Dao,
and supermimic_vibed / ETH Zurich, Chong Zhang notices.
`UPSTREAM.json` records original source hashes and the one contact-foot adaptation.
The K1 exact-state motion adapter follows the vendored MotionLib indexing API.
Software licensing does not grant rights to the separate K-POP motion data.
