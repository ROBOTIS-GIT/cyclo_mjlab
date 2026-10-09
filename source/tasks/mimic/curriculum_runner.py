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


class KoryoCurriculumRunner(MjlabOnPolicyRunner):
  """Restore Koryo difficulty independently of the PPO iteration counter."""

  def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    # A random initial episode clock would falsely count shorter frame-zero
    # attempts as 30-second successes.
    return super().learn(num_learning_iterations, init_at_random_ep_len=False)

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._refresh_reference()

  def _refresh_reference(self):
    # Explicit env.reset() forwards physics but leaves reference alignment caches
    # stale until the first command update. Refresh without advancing the clip.
    command = self.env.unwrapped.command_manager.get_term("reference_trajectory")
    command.frame_ids -= 1
    command._update_command()

  def save(self, path, infos=None):
    command = self.env.unwrapped.command_manager.get_term("reference_trajectory")
    infos = dict(infos or {})
    if hasattr(command, "curriculum_state_dict"):
      infos["koryo_curriculum"] = command.curriculum_state_dict()
    super().save(path, infos)

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    infos = super().load(path, load_cfg, strict, map_location)
    command = self.env.unwrapped.command_manager.get_term("reference_trajectory")
    if load_cfg is None and hasattr(command, "load_curriculum_state_dict"):
      state = (infos or {}).get("koryo_curriculum")
      if state is not None:
        command.load_curriculum_state_dict(state)
      else:
        print("[INFO] Baseline checkpoint: starting Koryo curriculum at stage 0.")
        command.trial_stage.fill_(-1)
      self.env.reset()
      self._refresh_reference()
    return infos
