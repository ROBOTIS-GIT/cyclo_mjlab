"""Contact-aware single-clip limbo tracking; no obstacle or success claim."""

from dataclasses import dataclass

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import ReferenceTrajectoryCommand, ReferenceTrajectoryCommandCfg

FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


class LimboCommand(ReferenceTrajectoryCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    if not np.isclose(self.reference.fps * env.step_dt, 1.0):
      raise ValueError("Limbo requires one reference frame per policy step (50 Hz).")
    with np.load(cfg.trajectory_file, allow_pickle=False) as data:
      plan = data["contact_plan"]
      if plan.dtype != np.bool_ or plan.shape != (self.reference.num_frames, 2):
        raise ValueError("contact_plan must be bool [frames, left/right].")
      if tuple(data["contact_names"]) != FEET:
        raise ValueError("Unexpected contact-plan foot order.")
    self.contact_plan = torch.as_tensor(plan, device=self.device)
    # Exclude +/-60 ms around planned liftoff/landing from binary contact cost.
    stable = plan.copy()
    stable[:] = True
    for offset in range(-3, 4):
      stable &= plan == plan[np.clip(np.arange(len(plan)) + offset, 0, len(plan)-1)]
    self.contact_stable = torch.as_tensor(stable, device=self.device)
    self.foot_ids = [cfg.body_names.index(name) for name in FEET]
    for name in ("stance_speed_mps", "stance_rotation_radps", "contact_mismatch",
                 "foot_position_error_m", "reference_fraction"):
      self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

  def _update_command(self):
    # The base command teleports the robot when the clip wraps. Hold the last
    # frame instead; reference_end explicitly truncates the episode next step.
    self.frame_ids.clamp_(max=self.reference.num_frames - 2)
    super()._update_command()

  def _update_metrics(self):
    super()._update_metrics()
    stance = self.contact_plan[self.frame_ids].float()
    speed = torch.linalg.vector_norm(self.robot_body_lin_vel_w[:, self.foot_ids], dim=-1)
    omega = torch.linalg.vector_norm(self.robot_body_ang_vel_w[:, self.foot_ids], dim=-1)
    self.metrics["stance_speed_mps"] = (speed * stance).sum(-1) / stance.sum(-1).clamp_min(1)
    self.metrics["stance_rotation_radps"] = (omega * stance).sum(-1) / stance.sum(-1).clamp_min(1)
    self.metrics["contact_mismatch"] = contact_mismatch(self._env)
    self.metrics["foot_position_error_m"] = torch.linalg.vector_norm(
      self.body_pos_w[:, self.foot_ids] - self.robot_body_pos_w[:, self.foot_ids], dim=-1
    ).mean(-1)
    # Progress is not success, especially with random reference-state resets.
    self.metrics["reference_fraction"] = self.frame_ids.float() / (self.reference.num_frames - 1)


@dataclass(kw_only=True)
class LimboCommandCfg(ReferenceTrajectoryCommandCfg):
  def build(self, env):
    return LimboCommand(self, env)


def command(env):
  return env.command_manager.get_term("reference_trajectory")


def phase_and_contacts(env):
  c = command(env)
  return torch.cat((c.frame_ids[:, None].float() / (c.reference.num_frames - 1),
                    c.contact_plan[c.frame_ids].float()), dim=-1)


def reference_end(env):
  return command(env).frame_ids >= command(env).reference.num_frames - 1


def contact_mismatch(env):
  c = command(env)
  actual = torch.stack([
    torch.linalg.vector_norm(env.scene[f"limbo_{side}_contact"].data.force, dim=-1).squeeze(-1) > 5.0
    for side in ("left", "right")
  ], dim=-1)
  valid = c.contact_stable[c.frame_ids]
  return ((actual != c.contact_plan[c.frame_ids]) & valid).float().mean(-1)


def stance_motion(env, angular=False):
  c = command(env)
  velocity = c.robot_body_ang_vel_w if angular else c.robot_body_lin_vel_w
  # COM translation + a separate angular term discourage sliding and pivoting.
  squared = velocity[:, c.foot_ids].square().sum(-1)
  return (squared * c.contact_plan[c.frame_ids]).sum(-1)


def foot_position_tracking(env, std=0.04):
  c = command(env)
  error = (c.body_pos_w[:, c.foot_ids] - c.robot_body_pos_w[:, c.foot_ids]).square().sum(-1)
  return torch.exp(-error / std**2).mean(-1)


def foot_orientation_tracking(env, std=0.15):
  c = command(env)
  error = quat_error_magnitude(c.body_quat_w[:, c.foot_ids], c.robot_body_quat_w[:, c.foot_ids])
  return torch.exp(-error.square() / std**2).mean(-1)
