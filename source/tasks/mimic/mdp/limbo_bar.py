"""Fixed-height, physical-bar limbo with explicit full-episode completion."""
from dataclasses import dataclass
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_error_magnitude
from . import limbo

BAR_Y = -0.65
BAR_RADIUS = 0.02
BAR_UNDERSIDE = 1.25


class LimboBarCommand(limbo.LimboCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.started_at_zero = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    model = env.sim.mj_model
    ids = self.robot.indexing.geom_ids
    self.clearance_geom_ids = ids
    self.geom_radius = torch.as_tensor(model.geom_rbound[ids.cpu().numpy()], device=self.device, dtype=torch.float32)
    for name in ("whole_body_clear", "bar_contact", "full_episode_success"):
      self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

  def _sample_start_frames(self, env_ids):
    super()._sample_start_frames(env_ids)
    if not self.cfg.start_from_zero:
      ids = torch.as_tensor(env_ids, device=self.device)
      self.frame_ids[ids[torch.rand(len(ids), device=self.device) < 0.2]] = 0
    self.started_at_zero[env_ids] = self.frame_ids[env_ids] == 0

  def _resample_command(self, env_ids):
    super()._resample_command(env_ids)
    # Exact-state resets: targets already match the reference pose. Initialize
    # these now, including the initial env.reset(), before the first physics step.
    self.aligned_body_position_w[env_ids] = self.body_pos_w[env_ids]
    self.aligned_body_orientation_w[env_ids] = self.body_quat_w[env_ids]

  def _update_metrics(self):
    super()._update_metrics()
    self.metrics["whole_body_clear"] = whole_body_clear(self._env).float()
    self.metrics["bar_contact"] = bar_contact(self._env).float()
    self.metrics["full_episode_success"] = full_success(self._env).float()


@dataclass(kw_only=True)
class LimboBarCommandCfg(limbo.LimboCommandCfg):
  def build(self, env):
    return LimboBarCommand(self, env)


def reset_bar(env, env_ids):
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  entity = env.scene["bar"]
  pose = entity.data.default_root_state[env_ids, :7].clone()
  pose[:, :3] += env.scene.env_origins[env_ids]
  entity.write_mocap_pose_to_sim(pose, env_ids=env_ids)


def bar_contact(env):
  data = env.scene["limbo_bar_contact"].data
  found = data.found.reshape(env.num_envs, -1).any(dim=-1)
  force = torch.linalg.vector_norm(data.force_history, dim=-1)
  return found | (force.reshape(env.num_envs, -1) > 0.1).any(dim=-1)


def whole_body_clear(env):
  c = limbo.command(env)
  # Bounding spheres enclose every robot mesh, including arms and feet.
  rear = (env.sim.data.geom_xpos[:, c.clearance_geom_ids, 1] + c.geom_radius).amax(dim=-1)
  return rear < env.scene.env_origins[:, 1] + BAR_Y - BAR_RADIUS - 0.05


def terminal_pose_ok(env):
  c = limbo.command(env)
  root_error = torch.linalg.vector_norm(c.robot_body_pos_w[:, 0] - c.body_pos_w[:, 0], dim=-1)
  up = torch.zeros((env.num_envs, 3), device=env.device)
  up[:, 2] = 1
  torso = c.cfg.body_names.index("torso_link")
  tilt_cos = quat_apply(c.robot_body_quat_w[:, torso], up)[:, 2]
  feet_down = torch.stack([
    torch.linalg.vector_norm(env.scene[f"limbo_{s}_contact"].data.force, dim=-1).squeeze(-1) > 5
    for s in ("left", "right")], dim=-1).all(dim=-1)
  return whole_body_clear(env) & (root_error < 0.12) & (tilt_cos > 0.9659) & feet_down & ~bar_contact(env)


def full_success(env):
  c = limbo.command(env)
  return c.started_at_zero & limbo.reference_end(env) & terminal_pose_ok(env) & ~nonfoot_support(env)


def nonfoot_support(env):
  force = env.scene["contact_forces"].data.force_history
  return (torch.linalg.vector_norm(force, dim=-1).reshape(env.num_envs, -1) > 10).any(dim=-1)


def leave_lane(env):
  c = limbo.command(env)
  return (c.robot_body_pos_w[:, 0, 0] - env.scene.env_origins[:, 0]).abs() > 0.40


def unsuccessful_end(env):
  return limbo.reference_end(env) & ~terminal_pose_ok(env)


def completion_reward(env):
  return full_success(env).float()
