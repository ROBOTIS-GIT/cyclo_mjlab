# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Warm-start old StandUp weights without inheriting its completed curriculum."""
import torch
from pathlib import Path
import yaml
from mjlab.rl import MjlabOnPolicyRunner
from .curriculum_runner import StandUpCurriculumRunner


class ReadyHandoffRunner(StandUpCurriculumRunner):
  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    super().__init__(env, train_cfg, log_dir, device)
    self._refresh()
    if log_dir is not None:
      target = Path(log_dir) / "params" / "ready_handoff.yaml"
      target.parent.mkdir(parents=True, exist_ok=True)
      target.write_text(yaml.safe_dump(self.handoff_spec(), sort_keys=False))

  def handoff_spec(self):
    env = self.env.unwrapped
    a = env.action_manager.get_term("joint_pos")
    c = a.command.cfg
    return {
      "version": 1,
      "status": "provisional; unreported ReadyPose gains use policy gains",
      "controller_required_outside_onnx": True,
      "policy_dt": env.step_dt,
      "joint_names": list(a.target_names),
      "ready_joint_target": a.ready_target.cpu().tolist(),
      "ready_kp": a.ready_kp.cpu().tolist(), "ready_kd": a.ready_kd.cpu().tolist(),
      "policy_kp": a.policy_kp.cpu().tolist(), "policy_kd": a.policy_kd.cpu().tolist(),
      "ready_hold_seconds": list(c.ready_hold_seconds),
      "handoff_seconds": list(c.handoff_seconds),
      "ready_timeout_seconds": c.ready_timeout_seconds,
      "stable_seconds": c.stable_seconds,
      "stable_joint_error_rad": c.stable_joint_error_rad,
      "stable_joint_speed_rad_s": c.stable_joint_speed_rad_s,
      "stable_body_speed_rad_s": c.stable_body_speed_rad_s,
      "interpolation": "linear, including alpha=0 and alpha=1 control steps",
      "reference_during_handoff": "frame 0; reference velocities zero",
      "last_action_observation": "previous raw policy output, before blend and scaling",
    }

  def _refresh(self):
    self.env.unwrapped.command_manager.get_term("reference_trajectory").refresh_reference()

  def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    return super().learn(num_learning_iterations, init_at_random_ep_len=False)

  def save(self, path, infos=None):
    super().save(path, {**(infos or {}), "ready_handoff_version": 1,
                        "ready_handoff_spec": self.handoff_spec()})

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    if load_cfg is not None:
      return super().load(path, load_cfg, strict, map_location)
    metadata = torch.load(path, map_location="cpu", weights_only=False).get("infos") or {}
    if metadata.get("ready_handoff_version") == 1:
      infos = super().load(path, None, strict, map_location)
      self.current_learning_iteration += 1
      self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]
    else:
      # The old actor and critic have identical dimensions. New rewards and
      # initial-state distribution need a fresh optimizer and curriculum.
      infos = MjlabOnPolicyRunner.load(self, path, {"actor": True, "critic": True}, strict, map_location)
      self.current_learning_iteration = 0
      self.env.unwrapped.common_step_counter = 0
      c = self.env.unwrapped.command_manager.get_term("reference_trajectory")
      c.load_curriculum_state_dict(dict(difficulty_stage=0, attempts=0, successes=0,
        last_success_rate=0., completed_successes=0, failed_attempts=0, interrupted_attempts=0))
      self.env.reset()
      print("[INFO] ReadyHandoff warm start: policy/value weights loaded; optimizer, iteration and DR stage reset.")
    self._refresh()
    return infos
