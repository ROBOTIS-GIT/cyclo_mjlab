# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Allow pelvis support while penalizing excessive floor contact force."""
import torch


def pelvis_impact(env, sensor_name: str, support_force_n: float,
                  max_penalty: float = 25.0):
  """Squared excess force, using all physics substeps of one policy step.

  support_force_n is a tuning threshold, not a hardware safety limit.
  """
  if support_force_n <= 0 or max_penalty <= 0:
    raise ValueError('Impact threshold and penalty bound must be positive.')
  data = env.scene[sensor_name].data
  force = data.force_history
  if force is None:
    force = data.force
  if force is None:
    raise RuntimeError('Pelvis impact requires a force contact sensor.')
  peak = torch.linalg.vector_norm(force, dim=-1).flatten(1).amax(dim=1)
  env.command_manager.get_term('reference_trajectory').metrics['pelvis_peak_force_n'] = peak
  return (peak / support_force_n - 1).clamp_min(0).square().clamp_max(max_penalty)
