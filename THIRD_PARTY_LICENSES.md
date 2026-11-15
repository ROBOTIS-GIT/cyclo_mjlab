# Third-Party Licenses

This project uses third-party open-source software and includes code adapted
from third-party open-source projects.

## mujocolab/mjlab

- Source: https://github.com/mujocolab/mjlab
- Version: 1.2.0
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

- Source: https://github.com/HybridRobotics/supermimic
- Revision: `abc124c1bd5976341034668af3d12c087f2f1f55`
- Copyright 2026 The SuperMimic Authors; Apache License 2.0.
- Original license: [third_party/supermimic/LICENSE](third_party/supermimic/LICENSE).
- Original README acknowledgements and local modification record:
  [third_party/supermimic/UPSTREAM.md](third_party/supermimic/UPSTREAM.md).
- Separate research runtime depends on HybridRobotics/mjlab_research revision
  `dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd` (Apache-2.0), without replacing
  the main project's pinned mjlab dependency.

## Modification notices and NOTICE scope

The official SuperMimic revision above and the pinned `mjlab_research` revision
were inspected directly. Both contain an Apache-2.0 `LICENSE`; neither contains a
`NOTICE` file in its tracked tree. No upstream NOTICE was removed. SuperMimic's
license and README are byte-identical to the pinned official source. Changed
vendored files carry explicit modification comments; the complete list is in
[UPSTREAM.md](third_party/supermimic/UPSTREAM.md).

`source/tasks/supermimic/train.py` retains the SuperMimic Authors copyright and
Apache identifier for its adapted launcher. `robot.py` retains the existing
ROBOTIS/Insu Park notice; `reference_profiles.py` identifies its exact historical
Cyclo XML source (`874c7bd`) and Apache notice. The integration compatibility
package declares Apache-2.0 and includes its own license copy when built.
These steps preserve distribution notices; they are not a guarantee of title
or non-infringement. Apache redistribution conditions are in
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

The research environment resolves the official SuperMimic package and pinned
`mjlab_research` in `integrations/supermimic/uv.lock`; it does not vendor those
installed wheels into Git. The following were checked from installed package
metadata/license-file declarations for the locked environment:

| Dependency | Declared license scope |
| --- | --- |
| SuperMimic; mjlab research; MuJoCo/MuJoCo Warp | Apache-2.0; package license files retained by their distributions |
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
