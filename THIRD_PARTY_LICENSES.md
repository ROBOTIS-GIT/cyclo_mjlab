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
  - `source/tasks/beyondmimic`
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
- Shared fork: https://github.com/ROBOTIS-GIT/supermimic in `third_party/supermimic`.
- Copyright 2026 The SuperMimic Authors; original Apache License 2.0 retained at
  [third_party/supermimic/LICENSE](third_party/supermimic/LICENSE).
- Portable math extracted from official `mjlab_research` retains its Isaac Lab
  BSD-3-Clause attribution and [license](third_party/supermimic/licenses/IsaacLab-BSD-3-Clause.txt).
  The shared package declares `Apache-2.0 AND BSD-3-Clause`.
- [PATCHES.md](third_party/supermimic/PATCHES.md) records the minimal core changes
  and primary-source revisions. Its README retains acknowledgements to ControlVAE
  (BSD-3-Clause), robotic_world_model (Apache-2.0), and DreamerV3 (MIT).

## Native adapter provenance and NOTICE scope

`source/tasks/supermimic/reference/forward_kinematics.py` preserves the implementation of
`HybridRobotics/mjlab_research` revision
`dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd:src/mjlab/utils/forward_kinematics.py`.
`reference/loader.py`, `mdp/commands.py` and `mdp/terminations.py` adapt that revision's
tracking command/termination interfaces to stable MJLab 1.6.1, including FK-backed
body state, concatenated clip boundaries and external reset sampling. The adapters
add strict compact-input validation, stable
manager reset dispatch and pending-forward handling. The original Apache license
is retained at [LICENSE.mjlab_research](source/tasks/supermimic/LICENSE.mjlab_research),
and the adapted files retain their copyright notices. Research MJLab is
not a dependency of this native runtime.

The official SuperMimic and research MJLab revisions contain Apache-2.0 licenses
and no tracked NOTICE file; none was removed. The fork preserves the original
license and README attribution, with explicit setup corrections for optional
research dependencies. Core modifications are recorded in its patch record.

`source/tasks/supermimic/train.py` retains the SuperMimic Authors copyright and
Apache identifier for its adapted launcher.
These checks preserve notices; they do not guarantee
title or non-infringement. Apache redistribution conditions are in
[section 4 of the license](https://www.apache.org/licenses/LICENSE-2.0).

## Robot assets, reference data and weights

| Material | Checked scope | Remaining limits |
| --- | --- | --- |
| K1 MJCF, meshes and robot assets | [ai_sapiens](https://github.com/ROBOTIS-GIT/ai_sapiens) materials in `third_party/ai_sapiens`; see the [upstream license](third_party/ai_sapiens/LICENSE) and [local notice](source/assets/robots/ai_sapiens/LICENSE.ai_sapiens) | Upstream license declaration was checked; independent CAD/mesh authorship and third-party permissions were not audited |
| Trained/pretrained weights | No `.pt`, `.pth`, `.onnx` or `.ckpt` files are tracked or added by the feature; local smoke outputs stay ignored | User-provided W&B bundles and external datasets/weights were not inspected or cleared by this review; their terms are separate from the code license |

Reference motions are subject to the terms of their source datasets.

## Installed dependency licenses

The common Python 3.12 runtime resolves stable MJLab and the shared core in
`requirements.lock`; local core source is installed from the pinned submodule.
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

`third_party/blinkmimic` pins the [ROBOTIS BlinkMimic fork](https://github.com/ROBOTIS-GIT/blinkmimic),
based on Fuyi Chen's [BlinkMimic](https://github.com/fuyichen0725/blinkmimic).
Its [MIT license](third_party/blinkmimic/LICENSE) and
[third-party notices](third_party/blinkmimic/THIRD_PARTY_NOTICES.md) retain the Fuyi Chen, InstantTrack / Huy-Quang-Dao,
and supermimic_vibed / ETH Zurich, Chong Zhang notices.
The K1 exact-state motion adapter follows the upstream MotionLib indexing API.
The existing HybridRobotics SuperMimic core has separate Apache/BSD code provenance;
algorithmic lineage does not make these codebases or licenses identical.
No new motion library, private dataset, trained weight or exported model is
distributed by the BlinkMimic feature. Those materials have separate provenance
and usage/redistribution conditions; the software license does not clear them.
