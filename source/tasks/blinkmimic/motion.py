"""Exact 50 Hz native states; no resampling or re-differentiation.

MotionLib indexing/goal API follows Fuyi Chen's MIT BlinkMimic (see vendor notices).
Only this K1 input adapter differs from its CSV/resampling path.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from blinkmimic.motion import BIN_FRAMES, MotionLib
from blinkmimic.rotations import quat_to_mat


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_manifest(path, verify_sources=False):
    manifest = json.loads(Path(path).read_text())
    rows = manifest["clips"]
    if len(rows) != 73 or sum(r["frames"] for r in rows) != 223812:
        raise ValueError("Expected the complete 73-song, 223812-frame library")
    if len({r["derived"] for r in rows}) != 73:
        raise ValueError("Duplicate derived clips")
    for row in rows:
        if row["fps"] != 50 or digest(row["derived"]) != row["derived_sha256"]:
            raise ValueError(f"Derived hash/rate mismatch: {row['song']}")
        if verify_sources and digest(row["source"]) != row["source_sha256"]:
            raise ValueError(f"Source hash mismatch: {row['song']}")
    return manifest


def load_exact(path, joint_names):
    with np.load(path, allow_pickle=False) as data:
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
    T = len(arrays[0])
    dims = (3, 4, 3, 3, len(joint_names), len(joint_names))
    for a, d in zip(arrays, dims, strict=True):
        if a.shape != (T, d) or T < 2 or not np.isfinite(a).all():
            raise ValueError(f"Invalid shape/values in {path}")
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
        self.clip_len = torch.tensor(lens, device=device)
        self.clip_start = self.clip_len.cumsum(0) - self.clip_len
        self.clip_end = self.clip_start + self.clip_len - 1
        self.num_frames = sum(lens)
        self.clip_id = torch.repeat_interleave(
            torch.arange(len(lens), device=device), self.clip_len
        )
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
        with torch.no_grad():
            out = [
                kin.forward(*(v[s : s + 8192] for v in self.ref_state_all()))
                for s in range(0, self.num_frames, 8192)
            ]
            self.body_p, self.body_R, self.body_v, self.body_w = [
                torch.cat([chunk[i] for chunk in out]) for i in range(4)
            ]
        Rt = self.root_R.transpose(-1, -2)
        self.root_v_local = (Rt @ self.root_v.unsqueeze(-1)).squeeze(-1)
        self.root_w_local = (Rt @ self.root_w.unsqueeze(-1)).squeeze(-1)
        self.root_gz = self.root_R[:, 2, 2]
        bin_clip, bin_start, frame_bin = [], [], []
        start = 0
        for c, n in enumerate(lens):
            for b in range(0, n, BIN_FRAMES):
                frame_bin.extend([len(bin_start)] * min(BIN_FRAMES, n - b))
                bin_clip.append(c)
                bin_start.append(start + b)
            start += n
        self.bin_clip = torch.tensor(bin_clip, device=device)
        self.bin_start = torch.tensor(bin_start, device=device)
        self.num_bins = len(bin_start)
        self.frame_bin = torch.tensor(frame_bin, device=device)

    def ref_state_all(self):
        return self.root_p, self.root_R, self.root_v, self.root_w, self.q, self.qd
