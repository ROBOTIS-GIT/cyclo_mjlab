# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Insu Park

"""Export an MJLab policy interface to ``sim2real.yaml``."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from mjlab.actuator import BuiltinPositionActuator, IdealPdActuator
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction


class _FlowList(list):
  """YAML list rendered in flow style, matching the ROBOTIS config format."""


class _Sim2RealYamlDumper(yaml.SafeDumper):
  pass


def _represent_flow_list(
  dumper: yaml.SafeDumper, data: _FlowList
) -> yaml.SequenceNode:
  return dumper.represent_sequence(
    "tag:yaml.org,2002:seq", data, flow_style=True
  )


_Sim2RealYamlDumper.add_representer(_FlowList, _represent_flow_list)

_FLOW_LIST_KEYS = {
  "scale",
  "offset",
  "clip",
  "lin_vel_x",
  "lin_vel_y",
  "ang_vel_z",
  "heading",
}

# Use deployment-facing term names while preserving the mjlab actor order.
_OBSERVATION_NAME_MAP = {
  "base_ang_vel": "base_ang_vel",
  "projected_gravity": "projected_gravity",
  "velocity_commands": "velocity_commands",
  "gait_phase": "gait_phase",
  "joint_pos": "joint_pos",
  "joint_vel": "joint_vel",
  "actions": "actions",
}


def _plain_value(value: Any) -> Any:
  """Convert tensors and containers to values supported by safe YAML."""
  if isinstance(value, torch.Tensor):
    value = value.detach().cpu().tolist()
  if isinstance(value, dict):
    return {str(key): _plain_value(item) for key, item in value.items()}
  if isinstance(value, (tuple, list)):
    return [_plain_value(item) for item in value]
  if isinstance(value, float):
    if not math.isfinite(value):
      raise ValueError(f"Cannot export non-finite value: {value}")
    return float(f"{value:.7g}")
  if isinstance(value, (str, int, bool)) or value is None:
    return value
  return value


def _vector(value: torch.Tensor | float | int, length: int) -> list[float]:
  """Expand an action/observation scalar or first environment row to a vector."""
  if isinstance(value, torch.Tensor):
    tensor = value.detach().cpu()
    if tensor.ndim > 1:
      tensor = tensor[0]
    values = tensor.reshape(-1).tolist()
  else:
    values = [float(value)]

  if len(values) == 1:
    values *= length
  if len(values) != length:
    raise ValueError(
      f"Expected vector of length {length}, received {len(values)} values."
    )
  return [float(f"{float(item):.7g}") for item in values]


def _position_action(env: ManagerBasedRlEnv) -> JointPositionAction:
  terms = [
    env.action_manager.get_term(name)
    for name in env.action_manager.active_terms
  ]
  position_terms = [
    term for term in terms if isinstance(term, JointPositionAction)
  ]
  if len(position_terms) != 1 or len(terms) != 1:
    raise ValueError(
      "sim2real export requires exactly one JointPositionAction term."
    )
  return position_terms[0]


def _pd_gains(
  robot: Entity, joint_names: Sequence[str]
) -> tuple[list[float], list[float]]:
  gains: dict[str, tuple[float, float]] = {}
  for actuator in robot.actuators:
    if not isinstance(actuator, (BuiltinPositionActuator, IdealPdActuator)):
      continue
    for name in actuator.target_names:
      gains[name] = (
        float(actuator.cfg.stiffness),
        float(actuator.cfg.damping),
      )

  missing = [name for name in joint_names if name not in gains]
  if missing:
    raise ValueError(
      "PD gains are unavailable for policy joints: " + ", ".join(missing)
    )

  stiffness = [gains[name][0] for name in joint_names]
  damping = [gains[name][1] for name in joint_names]
  return _plain_value(stiffness), _plain_value(damping)


def _format_yaml(value: Any, key: str | None = None) -> Any:
  value = _plain_value(value)
  if isinstance(value, list):
    items = [_format_yaml(item) for item in value]
    return _FlowList(items) if key in _FLOW_LIST_KEYS else items
  if isinstance(value, dict):
    return {
      item_key: _format_yaml(item, item_key)
      for item_key, item in value.items()
    }
  return value


def _commands(env: ManagerBasedRlEnv) -> dict[str, Any]:
  if "base_velocity" in env.cfg.commands:
    ranges = env.cfg.commands["base_velocity"].ranges
    return {
      "base_velocity": {
        "ranges": {
          "lin_vel_x": _plain_value(ranges.lin_vel_x),
          "lin_vel_y": _plain_value(ranges.lin_vel_y),
          "ang_vel_z": _plain_value(ranges.ang_vel_z),
          "heading": _plain_value(ranges.heading),
        }
      }
    }
  if "reference_trajectory" in env.cfg.commands:
    # Match Cyclo Lab: reference trajectory data is bundled with the Mimic
    # policy, so there is no separately configured runtime command generator.
    return {}
  raise ValueError(
    "Expected either a 'base_velocity' or 'reference_trajectory' command."
  )


def _observations(
  env: ManagerBasedRlEnv, group_name: str
) -> dict[str, Any]:
  if group_name not in env.observation_manager.active_terms:
    raise ValueError(f"Observation group '{group_name}' does not exist.")
  if not env.observation_manager.group_obs_concatenate[group_name]:
    raise ValueError(
      f"Observation group '{group_name}' must concatenate its terms."
    )

  names = env.observation_manager.active_terms[group_name]
  dims = env.observation_manager.group_obs_term_dim[group_name]
  exported: dict[str, Any] = {}
  is_mimic = "reference_trajectory" in env.cfg.commands

  for name, dims_with_history in zip(names, dims, strict=True):
    if not is_mimic and name not in _OBSERVATION_NAME_MAP:
      raise ValueError(
        f"Observation '{name}' is not supported by the sim2real schema."
      )

    term_cfg = env.observation_manager.get_term_cfg(group_name, name)
    history_length = max(int(term_cfg.history_length), 1)
    total_dim = math.prod(dims_with_history)
    if total_dim % history_length != 0:
      raise ValueError(
        f"Observation '{name}' shape {dims_with_history} is incompatible with "
        f"history length {history_length}."
      )
    term_dim = total_dim // history_length

    if term_cfg.scale is None:
      scale = [1.0] * term_dim
    elif isinstance(term_cfg.scale, torch.Tensor):
      scale = _vector(term_cfg.scale, term_dim)
    elif isinstance(term_cfg.scale, (float, int)):
      scale = [float(term_cfg.scale)] * term_dim
    else:
      scale = list(term_cfg.scale)
      if len(scale) == 1:
        scale *= term_dim
      if len(scale) != term_dim:
        raise ValueError(
          f"Observation '{name}' scale has {len(scale)} values; "
          f"expected {term_dim}."
        )

    params = _plain_value(term_cfg.params)
    if not is_mimic and name == "velocity_commands":
      params = {"command_name": "base_velocity"}
    elif not is_mimic and name == "gait_phase":
      # The deployment schema owns the clock update and only needs the cycle period.
      params = {"period": float(term_cfg.params["cycle_time_s"])}
    exported_name = name if is_mimic else _OBSERVATION_NAME_MAP[name]
    exported[exported_name] = {
      "params": _plain_value(params),
      "clip": _plain_value(term_cfg.clip),
      "scale": _plain_value(scale),
      "history_length": history_length,
    }

  expected_dim = math.prod(env.observation_manager.group_obs_dim[group_name])
  exported_dim = sum(
    len(term["scale"]) * term["history_length"]
    for term in exported.values()
  )
  if exported_dim != expected_dim:
    raise ValueError(
      f"Exported observation dimension {exported_dim} does not match policy "
      f"dimension {expected_dim}."
    )
  return exported


def export_sim2real_cfg(
  env: ManagerBasedRlEnv,
  log_dir: str | Path,
  *,
  observation_group: str = "actor",
) -> Path:
  """Write ``params/sim2real.yaml`` from the resolved environment."""
  action = _position_action(env)
  robot = env.scene[action.cfg.entity_name]
  if not isinstance(robot, Entity):
    raise ValueError(
      f"Action entity '{action.cfg.entity_name}' is not an articulated Entity."
    )

  policy_joint_names = list(action.target_names)
  target_ids = action.target_ids.detach().cpu().tolist()
  stiffness, damping = _pd_gains(robot, policy_joint_names)
  default_positions = _plain_value(
    robot.data.default_joint_pos[0, target_ids].detach().cpu()
  )

  joint_properties = {
    joint_name: {
      "default_position": default_position,
      "stiffness": joint_stiffness,
      "damping": joint_damping,
      "position_limit": None,
    }
    for joint_name, default_position, joint_stiffness, joint_damping in zip(
      policy_joint_names,
      default_positions,
      stiffness,
      damping,
      strict=True,
    )
  }

  cfg = {
    "policy_joints": policy_joint_names,
    "step_dt": float(env.step_dt),
    "joint_properties": joint_properties,
    "commands": _commands(env),
    "actions": {
      "joint_pos": {
        "clip": None,
        "scale": _vector(action.scale, action.action_dim),
        "offset": _vector(action.offset, action.action_dim),
      }
    },
    "observations": _observations(env, observation_group),
  }

  output_path = Path(log_dir) / "params" / "sim2real.yaml"
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", encoding="utf-8") as file:
    yaml.dump(
      _format_yaml(cfg),
      file,
      Dumper=_Sim2RealYamlDumper,
      sort_keys=False,
      default_flow_style=False,
      allow_unicode=True,
    )
  return output_path
