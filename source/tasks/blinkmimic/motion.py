"""Exact 50 Hz native states; no resampling or re-differentiation.

MotionLib indexing/goal API follows Fuyi Chen's MIT BlinkMimic (see vendor notices).
Only this K1 input adapter differs from its CSV/resampling path.
"""

import hashlib
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch
from blinkmimic.motion import MotionLib
from blinkmimic.rotations import quat_to_mat


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_manifest(path, verify_sources=False):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    rows = manifest.get("clips")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest requires a nonempty clips array")
    names, files = set(), set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("song"), str)
            or not row["song"].strip()
        ):
            raise ValueError("Each clip requires a nonempty song identifier")
        if row["song"] in names:
            raise ValueError(f"Duplicate clip identifier: {row['song']}")
        names.add(row["song"])
        if (
            type(row.get("frames")) is not int
            or row["frames"] < 2
            or row.get("fps") != 50
        ):
            raise ValueError(f"Invalid frame count/rate: {row['song']}")
        for key in ("derived", "source") if verify_sources else ("derived",):
            value = row.get(key)
            expected = row.get(f"{key}_sha256")
            if not isinstance(value, str) or not value or not isinstance(expected, str):
                raise ValueError(f"Missing {key} path/hash: {row['song']}")
            resolved = (path.parent / value).resolve()
            if key == "derived":
                if resolved in files:
                    raise ValueError("Duplicate derived clips")
                files.add(resolved)
            if digest(resolved) != expected:
                raise ValueError(f"{key.capitalize()} hash mismatch: {row['song']}")
            row[key] = str(resolved)
    return manifest


@lru_cache(maxsize=1)
def reference_model():
    import mujoco
    from source.assets.robots.robotis_k1.k1_rev1 import K1_REV1_XML_PATH

    return mujoco.MjModel.from_xml_path(str(K1_REV1_XML_PATH))


def load_exact(path, joint_names):
    from source.utils.k1_motion_npz_adapter import BODY_FIELDS, adapt_motion

    with np.load(path, allow_pickle=False) as archive:
        data = dict(archive)
    if any(key in data for key in BODY_FIELDS):
        data = adapt_motion(data, joint_names, reference_model())
    return validate_exact(data, joint_names, path)


def validate_exact(data, joint_names, path):
    for field, value in {
        "backend": "mujoco",
        "quaternion_order": "wxyz",
        "velocity_frame": "world",
        "velocity_point": "link_origin",
    }.items():
        if str(data[field].item()) != value:
            raise ValueError(f"Unsupported {field} in {path}")
    if float(data["fps"]) != 50:
        raise ValueError("K1 BlinkMimic requires exact 50 Hz input")
    names = data["joint_names"].tolist()
    if len(set(names)) != len(names) or set(names) != set(joint_names):
        raise ValueError("Joint names do not match public K1 Rev.1")
    perm = [names.index(n) for n in joint_names]
    arrays = [
        data[k].copy()
        for k in (
            "root_pos_w",
            "root_quat_w",
            "root_lin_vel_w",
            "root_ang_vel_w",
            "joint_pos",
            "joint_vel",
        )
    ]
    arrays[-2:] = [a[:, perm] for a in arrays[-2:]]
    T = arrays[0].shape[0] if arrays[0].ndim == 2 else 0
    dims = (3, 4, 3, 3, len(joint_names), len(joint_names))
    for i, (a, d) in enumerate(zip(arrays, dims, strict=True)):
        if a.shape != (T, d) or T < 2 or a.dtype.kind not in "fiu":
            raise ValueError(f"Invalid shape/values in {path}")
        with np.errstate(over="ignore", invalid="ignore"):
            arrays[i] = np.ascontiguousarray(a, dtype=np.float32)
        if not np.isfinite(arrays[i]).all():
            raise ValueError(f"Invalid values in float32 loader representation: {path}")
    if not np.allclose(np.linalg.norm(arrays[1], axis=-1), 1, atol=1e-4, rtol=0):
        raise ValueError("Unnormalized root quaternion")
    return arrays


class ExactMotionLib(MotionLib):
    @staticmethod
    def from_dir(path, robot, kin, device):
        manifest = verify_manifest(Path(path) / "manifest.json")
        expected = {Path(r["derived"]).resolve() for r in manifest["clips"]}
        if {p.resolve() for p in Path(path).glob("*.npz")} != expected:
            raise ValueError("Motion directory must exactly match the manifest")
        return ExactMotionLib(manifest, robot, kin, device)

    def __init__(self, manifest, robot, kin, device):
        self.device = device
        rows = manifest["clips"]
        self.names = [r["song"] for r in rows]
        clips = [load_exact(r["derived"], robot.joint_names) for r in rows]
        lens = [len(c[0]) for c in clips]
        if lens != [r["frames"] for r in rows]:
            raise ValueError("Manifest frame lengths differ")
        values = [
            torch.as_tensor(
                np.concatenate([c[i] for c in clips]),
                dtype=torch.float32,
                device=device,
            )
            for i in range(6)
        ]
        self.root_p, quat, self.root_v, self.root_w, self.q, self.qd = values
        self.root_R = quat_to_mat(quat)
        self.prepare_reference(lens, kin, chunk_size=8192)

    def ref_state_all(self):
        return self.root_p, self.root_R, self.root_v, self.root_w, self.q, self.qd
