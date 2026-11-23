# Copyright 2025 The mjlab Developers
# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Official research motion interfaces adapted to stable MJLab 1.6.1.

Source: HybridRobotics/mjlab_research dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd,
src/mjlab/tasks/tracking/mdp/{commands,terminations}.py. Stable MotionCommand
provides metrics, simulation writes and viewer lifecycle. This adapter supplies
the compact motion loader, FK, clip boundaries and external reset distribution
that the official SuperMimic runner requires. Learning stays in the submodule.
"""

import glob
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from mjlab.managers import CommandTerm
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, quat_inv, quat_mul, sample_uniform, yaw_quat

from .forward_kinematics import ForwardKinematics


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


class NativeMotionCommand(MotionCommand):
    """Stable native command with the official compact-reference contract."""

    def __init__(self, cfg, env):
        # Stable MotionCommand hard-codes its incompatible array-only loader.
        # Initialize its manager base, then construct the documented native state.
        CommandTerm.__init__(self, cfg, env)
        self.robot = env.scene[cfg.entity_name]
        self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
        self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )
        fk = ForwardKinematics(
            env.sim.mj_model,
            self.robot.indexing.body_ids,
            self.robot.indexing.joint_q_adr,
            self.body_indexes,
            device=self.device,
        )
        self.motion = NativeMotionLoader(cfg.motion_file, fk, self.device)
        self.time_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        (
            self._body_pos_w,
            self._body_quat_w,
            self._body_lin_vel_w,
            self._body_ang_vel_w,
        ) = self.motion.body_state(self.time_steps)
        self.body_pos_relative_w = torch.zeros_like(self._body_pos_w)
        self.body_quat_relative_w = torch.zeros_like(self._body_quat_w)
        self.body_quat_relative_w[:, :, 0] = 1
        # Preserve the official research sampler's 0.2-second bins.
        self.bin_count = int(self.motion.time_step_total // (0.2 / env.step_dt)) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, device=self.device)
        self._current_bin_failed = torch.zeros_like(self.bin_failed_count)
        self._external_probs = torch.full_like(
            self.bin_failed_count, 1 / self.bin_count
        )
        self.kernel = torch.tensor(
            [cfg.adaptive_lambda**i for i in range(cfg.adaptive_kernel_size)],
            device=self.device,
        )
        self.kernel /= self.kernel.sum()
        for name in (
            "error_anchor_pos",
            "error_anchor_rot",
            "error_anchor_lin_vel",
            "error_anchor_ang_vel",
            "error_body_pos",
            "error_body_rot",
            "error_body_lin_vel",
            "error_body_ang_vel",
            "error_joint_pos",
            "error_joint_vel",
            "sampling_entropy",
            "sampling_top1_prob",
            "sampling_top1_bin",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        self._ghost_model = None
        self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)
        self._pending_forward = False

    @property
    def body_pos_w(self):
        return self._body_pos_w + self._env.scene.env_origins[:, None]

    @property
    def body_quat_w(self):
        return self._body_quat_w

    @property
    def body_lin_vel_w(self):
        return self._body_lin_vel_w

    @property
    def body_ang_vel_w(self):
        return self._body_ang_vel_w

    @property
    def anchor_pos_w(self):
        return self.body_pos_w[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self):
        return self._body_quat_w[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self):
        return self._body_lin_vel_w[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self):
        return self._body_ang_vel_w[:, self.motion_anchor_body_index]

    def set_time_steps(self, env_ids, frames):
        self.time_steps[env_ids] = frames
        (
            self._body_pos_w[env_ids],
            self._body_quat_w[env_ids],
            self._body_lin_vel_w[env_ids],
            self._body_ang_vel_w[env_ids],
        ) = self.motion.body_state(frames)

    def set_sampling_distribution(self, probs):
        probs = probs.detach().to(self.device, dtype=torch.float).clamp_min(0)
        total = probs.sum()
        if probs.numel() == self.bin_count and torch.isfinite(total) and total > 0:
            self._external_probs = probs / total

    def _sample_bins(self, env_ids, probs):
        bins = torch.multinomial(probs, len(env_ids), replacement=True)
        total = self.motion.time_step_total - 2
        bins.clamp_(max=self.bin_count - 1)
        lo = bins * total // self.bin_count
        span = (((bins + 1) * total) // self.bin_count - lo).clamp_(min=1)
        jitter = (
            sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device) * span
        ).long()
        entropy = -(probs * (probs + 1e-12).log()).sum()
        self.metrics["sampling_entropy"][env_ids] = (
            entropy / math.log(self.bin_count) if self.bin_count > 1 else 1
        )
        pmax, imax = probs.max(dim=0)
        self.metrics["sampling_top1_prob"][env_ids] = pmax
        self.metrics["sampling_top1_bin"][env_ids] = imax.float() / self.bin_count
        return (lo + jitter).clamp_(max=total)

    def _sample_frames(self, env_ids):
        if self.cfg.sampling_mode == "start":
            return torch.zeros_like(env_ids)
        if self.cfg.sampling_mode == "uniform":
            self.metrics["sampling_entropy"][env_ids] = 1
            self.metrics["sampling_top1_prob"][env_ids] = 1 / self.bin_count
            self.metrics["sampling_top1_bin"][env_ids] = 0.5
            return torch.randint(
                0, self.motion.time_step_total - 1, (len(env_ids),), device=self.device
            )
        if self.cfg.sampling_mode == "external":
            return self._sample_bins(env_ids, self._external_probs)
        if self.cfg.sampling_mode != "adaptive":
            raise ValueError(f"Unsupported sampling mode {self.cfg.sampling_mode}")
        failed = self._env.termination_manager.terminated[env_ids]
        bins = (
            (self.time_steps[env_ids] * self.bin_count) // self.motion.time_step_total
        ).clamp(0, self.bin_count - 1)
        self._current_bin_failed += torch.bincount(
            bins[failed], minlength=self.bin_count
        )
        counts = self.bin_failed_count
        if self.cfg.adaptive_clip > 0:
            counts = counts.clamp(max=self.cfg.adaptive_clip * counts.mean())
        probs = counts + self.cfg.adaptive_uniform_ratio / self.bin_count
        probs = torch.nn.functional.pad(
            probs[None, None], (0, self.cfg.adaptive_kernel_size - 1), mode="replicate"
        )
        probs = torch.nn.functional.conv1d(probs, self.kernel.view(1, 1, -1)).flatten()
        return self._sample_bins(env_ids, probs / probs.sum())

    def _resample_command(self, env_ids):
        if len(env_ids) == 0:
            return
        frames = self._sample_frames(env_ids)
        frames = torch.where(
            torch.isin(frames, self.motion.motion_ends - 1), frames - 1, frames
        )
        self.set_time_steps(env_ids, frames)
        root_pos = self.motion.root_pos_w[frames] + self._env.scene.env_origins[env_ids]
        root_ori = self.motion.root_quat_w[frames]
        root_lin = self.motion.root_lin_vel_w[frames].clone()
        root_ang = self.motion.root_ang_vel_w[frames].clone()
        keys = ("x", "y", "z", "roll", "pitch", "yaw")
        ranges = torch.tensor(
            [self.cfg.pose_range.get(k, (0.0, 0.0)) for k in keys], device=self.device
        )
        noise = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_pos += noise[:, :3]
        root_ori = quat_mul(
            quat_from_euler_xyz(noise[:, 3], noise[:, 4], noise[:, 5]), root_ori
        )
        ranges = torch.tensor(
            [self.cfg.velocity_range.get(k, (0.0, 0.0)) for k in keys],
            device=self.device,
        )
        noise = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_lin += noise[:, :3]
        root_ang += noise[:, 3:]
        joints = self.joint_pos[env_ids].clone()
        joints += sample_uniform(
            *self.cfg.joint_position_range, joints.shape, device=self.device
        )
        self._write_reference_state_to_sim(
            env_ids,
            root_pos,
            root_ori,
            root_lin,
            root_ang,
            joints,
            self.joint_vel[env_ids],
        )
        self._pending_forward = True

    def _update_command(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.time_steps[ids] += 1
        wrap = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        if env_ids is not None:
            wrap = wrap[torch.isin(wrap, env_ids)]
        self._resample_command(wrap)
        self.set_time_steps(ids, self.time_steps[ids])
        if self._pending_forward:
            self._env.sim.forward()
            self._pending_forward = False
        self.update_relative_body_poses()
        if env_ids is None and self.cfg.sampling_mode == "adaptive":
            self.bin_failed_count = (
                self.cfg.adaptive_alpha * self._current_bin_failed
                + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
            )
            self._current_bin_failed.zero_()

    def reset_to_frame(self, env_ids, frame):
        frames = torch.full_like(env_ids, frame)
        self.set_time_steps(env_ids, frames)
        self._write_reference_state_to_sim(
            env_ids,
            self.motion.root_pos_w[frames] + self._env.scene.env_origins[env_ids],
            self.motion.root_quat_w[frames],
            self.motion.root_lin_vel_w[frames],
            self.motion.root_ang_vel_w[frames],
            self.joint_pos[env_ids],
            self.joint_vel[env_ids],
        )


@dataclass(kw_only=True)
class NativeMotionCommandCfg(MotionCommandCfg):
    motion_file: str | list[str] = ""
    sampling_mode: Literal["adaptive", "uniform", "start", "external"] = "adaptive"
    adaptive_clip: float = 100.0

    def build(self, env):
        return NativeMotionCommand(self, env)


def heading_velocity_tracking(env, command_name, std, angular=False, body_names=None):
    """Backport mjlab_research's heading-relative velocity rewards for SuperMimic.

    Our pinned mjlab 1.6.1 scores velocities in world coordinates. The return
    sampler uses these simulator rewards to choose motion reset frames.
    """
    command = env.command_manager.get_term(command_name)
    heading = yaw_quat(quat_mul(command.robot_anchor_quat_w, quat_inv(command.anchor_quat_w)))
    if angular:
        target, actual = command.body_ang_vel_w, command.robot_body_ang_vel_w
    else:
        target, actual = command.body_lin_vel_w, command.robot_body_lin_vel_w
    if body_names is not None:
        indices = [i for i, name in enumerate(command.cfg.body_names) if name in body_names]
        target, actual = target[:, indices], actual[:, indices]
    aligned = quat_apply(heading[:, None, :].expand(-1, target.shape[1], -1), target)
    error = (actual - aligned).square().sum(dim=-1).mean(dim=-1)
    return torch.exp(-error / std**2)


def motion_end(env, command_name):
    command = env.command_manager.get_term(command_name)
    return torch.isin(command.time_steps + 1, command.motion.motion_ends)
