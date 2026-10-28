"""The contact V2 reference with a physical bar whose underside is 1.20 m."""
from dataclasses import dataclass
import torch
from mjlab.utils.lab_api.math import quat_apply
from . import limbo, limbo_bar

BAR_Y = -1.4
BAR_RADIUS = 0.02
BAR_UNDERSIDE = 1.2

bar_contact = limbo_bar.bar_contact
nonfoot_support = limbo_bar.nonfoot_support
leave_lane = limbo_bar.leave_lane
reset_bar = limbo_bar.reset_bar


class Limbo1p2V2Command(limbo_bar.LimboBarCommand):
  def _update_metrics(self):
    # The original bar command uses its own module's fixed 1.25m-bar location.
    limbo.LimboCommand._update_metrics(self)
    self.metrics["whole_body_clear"] = whole_body_clear(self._env).float()
    self.metrics["bar_contact"] = bar_contact(self._env).float()
    self.metrics["full_episode_success"] = full_success(self._env).float()


@dataclass(kw_only=True)
class Limbo1p2V2CommandCfg(limbo_bar.LimboBarCommandCfg):
  def build(self, env):
    return Limbo1p2V2Command(self, env)


def whole_body_clear(env):
  c = limbo.command(env)
  rear = (env.sim.data.geom_xpos[:, c.clearance_geom_ids, 1] + c.geom_radius).amax(-1)
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
    for s in ("left", "right")], dim=-1).all(-1)
  return whole_body_clear(env) & (root_error < 0.12) & (tilt_cos > 0.9659) & feet_down & ~bar_contact(env)


def full_success(env):
  c = limbo.command(env)
  return c.started_at_zero & limbo.reference_end(env) & terminal_pose_ok(env) & ~nonfoot_support(env)


def unsuccessful_end(env):
  return limbo.reference_end(env) & ~terminal_pose_ok(env)


def completion_reward(env):
  return full_success(env).float()
