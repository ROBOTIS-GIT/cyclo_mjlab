# ROBOTIS AI Sapiens K1 Rev.1 assets

The K1 model is generated from the `ROBOTIS-GIT/ai_sapiens` submodule:

```text
third_party/ai_sapiens/ai_sapiens_description/urdf/k1_rev1/k1.urdf
```

Regenerate the MJCF and copy its mesh assets with:

```bash
python scripts/convert_k1_urdf_to_mjcf.py
```

Compare visual and collision geometry with:

```bash
python scripts/visualize_robot.py k1 --mode both
python scripts/visualize_robot.py k1 --mode collision
```

The upstream model and copied meshes are licensed under Apache-2.0. See
`LICENSE.ai_sapiens` and the pinned submodule for source and attribution.
