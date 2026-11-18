"""The Unitree G1 tracking scene as a MuJoCo model, with the simulation options BlinkMimic applies at runtime.

``assets/g1/scene.xml`` and ``meta.json`` were exported from BlinkMimic's mjlab scene (tools/export_g1_from_blinkmimic.py);
the meshes are downloaded by tools/download_data.sh. The XML prints floats with about 7 significant digits, so the fields
it rounds (actuator gains, control ranges, armature, masses, inertias) are restored from the exact values in meta.json:
every compared model array then matches mjlab's.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "g1"


def load_meta(path: Path = ASSETS) -> dict:
    """Robot metadata: joint and body order, actuators, default pose, action scale, domain randomization, solver options."""
    return json.loads((path / "meta.json").read_text())


def load_model(path: Path = ASSETS) -> tuple[mujoco.MjModel, dict]:
    """(MjModel with BlinkMimic's solver options and exact parameter values, metadata)."""
    path = Path(path) if path else ASSETS
    if path == ASSETS and not (path / "assets" / "meshes").is_dir():
        raise FileNotFoundError(f"G1 meshes not found in {path / 'assets' / 'meshes'}: run tools/download_data.sh first")
    meta = load_meta(path)
    m = mujoco.MjModel.from_xml_path(str(path / "scene.xml"))
    s = meta["sim"]
    m.opt.timestep = s["timestep"]
    m.opt.integrator = s["integrator"]
    m.opt.cone = s["cone"]
    m.opt.solver = s["solver"]
    m.opt.iterations = s["iterations"]
    m.opt.tolerance = s["tolerance"]
    m.opt.ls_iterations = s["ls_iterations"]
    m.opt.ls_tolerance = s["ls_tolerance"]
    m.opt.ccd_iterations = s["ccd_iterations"]
    m.opt.impratio = s["impratio"]
    for field in ("jacobian", "disableflags", "enableflags", "gravity"):
        if field in s:
            if field == "gravity":
                m.opt.gravity[:] = s[field]
            else:
                setattr(m.opt, field, s[field])
    for field, value in meta["exact"].items():
        getattr(m, field)[:] = np.asarray(value, dtype=np.float64).reshape(getattr(m, field).shape)
    return m, meta
