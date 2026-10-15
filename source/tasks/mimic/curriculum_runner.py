# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Persist success-gated curriculum without modifying the installed MJLab."""
from mjlab.rl import MjlabOnPolicyRunner
from .motion_curriculum import MOTIONS, motion_identity, validate_checkpoint_motion


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
  motion_name = "koryo"

  def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    # A random initial episode clock would falsely count shorter frame-zero
    # attempts as 30-second successes.
    print(f"[INFO] {self.motion_name} stage promotion requires the full-motion curriculum driver.")
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
    spec = MOTIONS[self.motion_name]
    if hasattr(command, "curriculum_state_dict"):
      infos[spec.state_key] = command.curriculum_state_dict()
      infos["motion_identity"] = motion_identity(
        self.motion_name, getattr(command.cfg, "trajectory_file", None))
      if hasattr(command, "failure_score_by_bin"):
        infos[spec.sampler_key] = command.failure_score_by_bin.detach().cpu().clone()
    super().save(path, infos)

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    infos = super().load(path, load_cfg, strict, map_location)
    command = self.env.unwrapped.command_manager.get_term("reference_trajectory")
    if load_cfg is None and hasattr(command, "load_curriculum_state_dict"):
      spec = MOTIONS[self.motion_name]
      validate_checkpoint_motion(infos, self.motion_name, motion_identity(
        self.motion_name, getattr(command.cfg, "trajectory_file", None)))
      state = (infos or {}).get(spec.state_key)
      if state is not None:
        command.load_curriculum_state_dict(state)
        if state.get("version", 1) == 1:
          print(f"[INFO] Migrated legacy {self.motion_name} curriculum; full-motion gate counters reset.")
      else:
        print(f"[INFO] Baseline checkpoint: starting {self.motion_name} curriculum at stage 0.")
        command.trial_stage.fill_(-1)
      scores = (infos or {}).get(spec.sampler_key)
      if scores is not None:
        if (scores.shape != command.failure_score_by_bin.shape
            or not bool(scores.isfinite().all()) or bool((scores < 0).any())):
          raise ValueError("Invalid Koryo adaptive-start checkpoint state.")
        command.failure_score_by_bin.copy_(scores.to(command.device))
      self.env.reset()
      self._refresh_reference()
      # RSL-RL stores the last completed iteration, not the next one. Do not
      # repeat its number when the full-motion driver resumes a training block.
      self.current_learning_iteration += 1
      # PPO restores optimizer groups but leaves its adaptive-LR scalar at the
      # constructor default. Keep the next update at the checkpoint's rate.
      self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]
    return infos


class Taeguek01CurriculumRunner(KoryoCurriculumRunner):
  """Use independent checkpoint state and start-sampling statistics for Taeguek01."""
  motion_name = "taeguek01"


class Taeguek02CurriculumRunner(KoryoCurriculumRunner):
  """Persist Taeguek02 curriculum and sampler independently of other motions."""
  motion_name = "taeguek02"
