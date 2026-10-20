# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Provisional hardware ReadyPose gains, settling, and phase-zero PD handoff."""
from dataclasses import fields
import torch
from mjlab.managers import EventTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from source.tasks.mimic.mdp.ready_handoff import (
  ReadyHandoffCommandCfg, ReadyHandoffActionCfg, handoff_model_fields,
  handoff_timeout, handoff_push, handoff_target_rate, handoff_torque_rate,
)
from .umr_stand_up_soft_landing_env_cfg import k1_rev1_umr_stand_up_soft_landing_env_cfg


def handoff_body_impact(env, sensor_name, support_force_n):
  c = env.command_manager.get_term("reference_trajectory")
  data = env.scene[sensor_name].data
  force = data.force_history if data.force_history is not None else data.force
  peak = torch.linalg.vector_norm(force, dim=-1).flatten(1).amax(dim=1)
  c.metrics["handoff_body_peak_force_n"] = peak * c.penalty_mask
  return (peak / support_force_n - 1.).clamp_min(0.).square().clamp_max(25.) * c.penalty_mask


def k1_rev1_umr_stand_up_ready_handoff_env_cfg(play=False):
  cfg = k1_rev1_umr_stand_up_soft_landing_env_cfg(play=play)
  old = cfg.commands["reference_trajectory"]
  command = ReadyHandoffCommandCfg(**{f.name: getattr(old, f.name) for f in fields(old)})
  command.preparation_start_seconds = (5.4875, 8.2)
  command.first_frame_env_fraction = 1.0 if play else 0.3
  # Ready hold increases the wall-clock duration of complete attempts.
  cfg.episode_length_s = int(1e9) if play else 35.
  if play:
    command.ready_hold_seconds = (0.6, 0.6)
    command.handoff_seconds = (0.8, 0.8)
    command.resample_material_on_reset = False
  cfg.commands["reference_trajectory"] = command
  old_action = cfg.actions["joint_pos"]
  cfg.actions["joint_pos"] = ReadyHandoffActionCfg(**{f.name: getattr(old_action, f.name) for f in fields(old_action)})
  cfg.events["handoff_gains"] = EventTermCfg(func=handoff_model_fields, mode="startup")
  if "push_robot" in cfg.events:
    cfg.events["push_robot"].func = handoff_push
  cfg.terminations["ready_timeout"] = TerminationTermCfg(func=handoff_timeout)
  cfg.rewards["handoff_target_rate"] = RewardTermCfg(func=handoff_target_rate, weight=-0.2)
  cfg.rewards["handoff_torque_rate"] = RewardTermCfg(func=handoff_torque_rate, weight=-0.05)
  cfg.scene.sensors = (*cfg.scene.sensors, ContactSensorCfg(
    name="handoff_body_floor_contact",
    primary=ContactMatch(mode="body", pattern="(pelvis|torso_link)", entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="terrain"),
    secondary_policy="error", fields=("force",), reduce="netforce",
    num_slots=1, history_length=cfg.decimation,
  ))
  cfg.rewards["handoff_body_impact"] = RewardTermCfg(
    func=handoff_body_impact, weight=-0.2,
    params={"sensor_name": "handoff_body_floor_contact", "support_force_n": 420.},
  )
  return cfg
