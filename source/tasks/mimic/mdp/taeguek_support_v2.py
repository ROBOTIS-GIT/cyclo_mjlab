# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Reference-gated support clearance and kick height for Taegeuk 4/5."""
import torch
from mjlab.utils.lab_api.math import quat_apply
from .taeguek_support import TaegeukFootSupportQuality


def sole_bottom_height(position, quaternion, offsets, radii):
  n, feet, _ = quaternion.shape
  count = offsets.shape[1]
  rotated = quat_apply(quaternion[:, :, None, :].expand(n, feet, count, 4),
                       offsets[None].expand(n, -1, -1, -1))
  return position[..., 2] + (rotated[..., 2] - radii[None]).amin(-1)


def clearance_cost(actual, reference, support, tolerance=0.003, scale=0.01):
  """Excess sole height, with no dependence on sensed contact or robot speed.

  Reference clearance is allowed, including a tilted support foot. The cost
  is bounded so recovery remains possible; contact continuity is separate.
  """
  excess = (actual - reference.clamp_min(0.) - tolerance).clamp_min(0.)
  return (excess / scale).square().clamp_max(4.) * support


def kick_height_cost(actual, reference, pelvis_actual, pelvis_reference, kick,
                     scale=0.10):
  """Track both world and pelvis-relative height; don't reward a body hop.

  Two-sided error discourages both lowering the kick and overshooting it.
  Eligibility depends only on the reference, never the performed kick height.
  """
  world_error = actual - reference
  relative_error = world_error - (pelvis_actual - pelvis_reference)[:, None]
  cost = 0.5 * ((world_error / scale).square()
                + (relative_error / scale).square())
  return cost.clamp_max(4.) * kick


class TaegeukFootSupportV2(TaegeukFootSupportQuality):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.pelvis = self.cmd.cfg.body_names.index('pelvis')
    relative = self.reference_bottom - self.reference_bottom.amin(0)
    self.reference_kick = ((relative > 0.25)
                           & self.reference_support.flip(-1)
                           & ~self.reference_support)
    self.v2_last_step = -1
    self.v2_cached = torch.zeros_like(self.cached)

  def reset(self, env_ids=None):
    super().reset(env_ids)
    if hasattr(self, 'v2_cached'):
      self.v2_cached[slice(None) if env_ids is None else env_ids] = 0
      self.v2_last_step = -1

  def __call__(self, env, support_weight=0.9, long_unload_weight=0.3,
               liftoff_weight=0.04, slip_weight=0.15, rotation_weight=0.05,
               single_support_multiplier=1.5, ramp_steps=1,
               clearance_weight=1.0, kick_height_weight=0.75,
               refinement_ramp_steps=12000):
    if self.v2_last_step == env.common_step_counter:
      return self.v2_cached
    # Keep V1 contact tracking, reset grace period and reference phase labels.
    # Stronger contact costs and geometric costs are introduced gradually.
    ramp = min(1., 0.2 + 0.8 * env.common_step_counter / max(1, refinement_ramp_steps))
    base = super().__call__(env,
      support_weight=0.45 + ramp * (support_weight - 0.45),
      long_unload_weight=0.15 + ramp * (long_unload_weight - 0.15),
      liftoff_weight=0.02 + ramp * (liftoff_weight - 0.02),
      slip_weight=slip_weight, rotation_weight=rotation_weight,
      single_support_multiplier=single_support_multiplier, ramp_steps=ramp_steps)
    frame = self.cmd.frame_ids
    support = self.reference_support[frame]
    kick = self.reference_kick[frame]
    bottom = sole_bottom_height(self.cmd.robot_body_pos_w[:, self.feet],
      self.cmd.robot_body_quat_w[:, self.feet], self.offsets, self.radii)
    # Flat training terrain at each environment's origin; ref has no origins.
    bottom = bottom - env.scene.env_origins[:, 2, None]
    target = self.reference_bottom[frame]
    clearance = clearance_cost(bottom, target, support)
    pelvis_actual = (self.cmd.robot_body_pos_w[:, self.pelvis, 2]
                     - env.scene.env_origins[:, 2])
    pelvis_ref = self.cmd.reference.body_position_w[frame, self.pelvis, 2]
    height = kick_height_cost(bottom, target, pelvis_actual, pelvis_ref, kick)
    multiplier = torch.where(support.sum(-1) == 1, single_support_multiplier, 1.)
    valid = (self.age > 0.15).float()
    extra = clearance_weight * clearance.sum(-1) * multiplier + kick_height_weight * height.sum(-1)
    self.cmd.metrics['support_v2_clearance_cost'] = clearance.sum(-1) * valid
    self.cmd.metrics['support_v2_kick_height_cost'] = height.sum(-1) * valid
    self.cmd.metrics['support_v2_kick_height_abs_error'] = ((bottom-target).abs()*kick).sum(-1)*valid
    self.cmd.metrics['support_v2_ramp'] = torch.full_like(valid, ramp)
    self.v2_cached = base + ramp * extra * valid
    self.v2_last_step = env.common_step_counter
    return self.v2_cached
