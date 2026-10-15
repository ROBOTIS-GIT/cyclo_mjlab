# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Contact penalties for the face-down LieDownBack landing.

Thresholds tune the simulation objective; they are not hardware safety limits.
"""
import torch


def knee_impact(env, sensor_name: str, support_force_n: float = 350.0):
  """Allow steady support, penalize each knee's excessive substep peak."""
  force = env.scene[sensor_name].data.force_history
  if force is None:
    raise RuntimeError('Knee impact requires contact force history.')
  peak = torch.linalg.vector_norm(force, dim=-1).amax(dim=2)
  command = env.command_manager.get_term('reference_trajectory')
  command.metrics['knee_peak_force_n'] = peak.amax(dim=1)
  return (peak / support_force_n - 1).clamp_min(0).square().clamp_max(25).sum(dim=1)


def toe_load_change(env, sensor_names: tuple[str, ...],
                    allowed_change_n: float = 50.0, scale_n: float = 175.0,
                    peak_fraction: float = 0.0):
  """Suppress abrupt forefoot loading, allowing smooth rolling and load transfer.

  Sum the five toe contacts BEFORE differencing so moving the contact between
  neighboring spheres is not itself penalized. Histories include the previous
  policy step's final substep to cover the policy-step boundary.

  Blend mean and peak penalties so a brief re-contact spike is not diluted by
  quiet substeps. Zero retains the original SoftContact objective.
  """
  if not 0.0 <= peak_fraction <= 1.0:
    raise ValueError('peak_fraction must be between zero and one.')
  if allowed_change_n < 0 or scale_n <= 0:
    raise ValueError('Load allowance must be nonnegative and scale positive.')
  command = env.command_manager.get_term('reference_trajectory')
  penalties, changes = [], []
  for name in sensor_names:
    force = env.scene[name].data.force_history
    if force is None or force.shape[2] < 2:
      raise RuntimeError('Toe load change requires at least two history samples.')
    # netforce sensors are global-frame; the floor normal is vertical.
    load = force[..., 2].sum(dim=1).abs()
    delta = torch.diff(load, dim=1).abs()
    peak_change = delta.amax(dim=1)
    changes.append(peak_change)
    command.metrics[f'{name}_load_change_n'] = peak_change
    excess = (delta - allowed_change_n).clamp_min(0) / scale_n
    cost = excess.square().clamp_max(25)
    penalties.append(
      (1 - peak_fraction) * cost.mean(dim=1)
      + peak_fraction * cost.amax(dim=1)
    )
  command.metrics['toe_load_change_n'] = torch.stack(changes, dim=1).amax(dim=1)
  return torch.stack(penalties, dim=1).sum(dim=1)
