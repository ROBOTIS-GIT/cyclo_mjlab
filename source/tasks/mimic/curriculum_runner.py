# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Persist success-gated curriculum without modifying the installed MJLab."""
from mjlab.rl import MjlabOnPolicyRunner


class StandUpCurriculumRunner(MjlabOnPolicyRunner):
  def save(self, path, infos=None):
    command = self.env.unwrapped.command_manager.get_term("reference_trajectory")
    infos = dict(infos or {})
    if hasattr(command, "curriculum_state_dict"):
      infos["stand_up_curriculum"] = command.curriculum_state_dict()
    super().save(path, infos)

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    infos = super().load(path, load_cfg, strict, map_location)
    command = self.env.unwrapped.command_manager.get_term("reference_trajectory")
    # Actor-only loading for playback/evaluation must not restore training state.
    if load_cfg is None and hasattr(command, "load_curriculum_state_dict"):
      state = (infos or {}).get("stand_up_curriculum")
      if state is not None:
        command.load_curriculum_state_dict(state)
        # Start fresh attempts with the restored material and reset-noise stage.
        self.env.reset()
      else:
        print("[INFO] Checkpoint has no stand-up curriculum state; starting stage 0.")
    return infos
