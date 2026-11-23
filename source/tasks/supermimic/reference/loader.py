# Copyright 2025 The mjlab Developers
# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Load compact reference motions for the SuperMimic tracking command."""

import glob
import hashlib
from pathlib import Path

import numpy as np
import torch


class NativeMotionLoader:
    """Validated 50 Hz compact clips, concatenated without losing boundaries."""

    def __init__(self, files, fk, device="cpu"):
        patterns = [files] if isinstance(files, str) else files
        files = []
        for pattern in patterns:
            matches = sorted(glob.glob(str(pattern)))
            if not matches:
                raise ValueError(f"No native reference files match {pattern!r}")
            files.extend(matches)
        clips = []
        self.clip_layout = []
        for path in files:
            with np.load(path, allow_pickle=False) as data:
                if data["joint_names"].tolist() != fk.joint_names:
                    raise ValueError(
                        "Reference joint names/order do not match native FK"
                    )
                if data["fps"].size != 1 or data["fps"].item() != 50:
                    raise ValueError("Native references must retain 50 Hz timing")
                clip = {
                    name: data[name]
                    for name in (
                        "root_pos_w",
                        "root_quat_w",
                        "root_lin_vel_w",
                        "root_ang_vel_w",
                        "joint_pos",
                        "joint_vel",
                    )
                }
                frames = len(clip["joint_pos"])
                for name, values in clip.items():
                    width = (
                        len(fk.joint_names)
                        if name.startswith("joint_")
                        else (4 if name == "root_quat_w" else 3)
                    )
                    if (
                        values.shape != (frames, width)
                        or not np.isfinite(values.astype(np.float32)).all()
                    ):
                        raise ValueError(
                            f"Invalid native reference shape or float32 values: {name}"
                        )
                if not np.allclose(
                    np.linalg.norm(clip["root_quat_w"], axis=-1), 1, atol=1e-4
                ):
                    raise ValueError(
                        "Native reference root quaternions must be normalized wxyz"
                    )
                # Stable across adapter cache paths and ordering in the new run.
                digest = hashlib.sha256()
                for values in clip.values():
                    digest.update(np.ascontiguousarray(values, dtype="<f4"))
                self.clip_layout.append((digest.hexdigest(), frames))
                clips.append(clip)
        if not clips or any(len(c["joint_pos"]) < 2 for c in clips):
            raise ValueError("Each native reference clip needs at least two frames")
        for name in clips[0]:
            setattr(
                self,
                name,
                torch.as_tensor(
                    np.concatenate([c[name] for c in clips]),
                    dtype=torch.float32,
                    device=device,
                ),
            )
        self.fk = fk
        self.time_step_total = len(self.joint_pos)
        self.motion_ends = torch.tensor(
            np.cumsum([len(c["joint_pos"]) for c in clips]),
            dtype=torch.long,
            device=device,
        )
        self.motion_names = [Path(path).stem for path in files]

    def body_state(self, frames):
        return self.fk(
            self.root_pos_w[frames],
            self.root_quat_w[frames],
            self.joint_pos[frames],
            self.root_lin_vel_w[frames],
            self.root_ang_vel_w[frames],
            self.joint_vel[frames],
        )
