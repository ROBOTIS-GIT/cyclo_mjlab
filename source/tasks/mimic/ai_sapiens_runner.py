# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Distinct checkpoint identity: old ReadyHandoff weights are a warm start only."""
from pathlib import Path
import hashlib
import shutil
import torch
import yaml
from mjlab.rl import MjlabOnPolicyRunner
from .curriculum_runner import StandUpCurriculumRunner
from .mdp.ai_sapiens_entry import ASSET, CONTRACT

class AiSapiensRunner(StandUpCurriculumRunner):
  def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
    super().__init__(env, train_cfg, log_dir, device)
    self._refresh()
    if log_dir is not None:
      folder = Path(log_dir) / 'params'
      folder.mkdir(parents=True, exist_ok=True)
      (folder / 'ai_sapiens_entry.yaml').write_text(yaml.safe_dump(self.entry_spec(), sort_keys=False))
      motion = Path(self.env.unwrapped.command_manager.get_term('reference_trajectory').cfg.trajectory_file).with_suffix('.csv')
      if motion.exists():
        shutil.copy2(motion, folder / motion.name)

  def entry_spec(self):
    return {**CONTRACT, 'ready_state': 'settled reset bank; no reference hold',
      'last_action': 'previous raw policy output', 'physics_dt': self.env.unwrapped.physics_dt,
      'policy_dt': self.env.unwrapped.step_dt,
      'model_sha256': hashlib.sha256((ASSET / 'mujoco/k1/k1.xml').read_bytes()).hexdigest(),
      'bank_sha256': hashlib.sha256((ASSET / 'ready_bank.npz').read_bytes()).hexdigest()}

  def _refresh(self):
    env = self.env.unwrapped
    env.sim.forward()
    env.command_manager.get_term('reference_trajectory').refresh_reference()

  def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    return super().learn(num_learning_iterations, init_at_random_ep_len=False)

  def save(self, path, infos=None):
    super().save(path, {**(infos or {}), 'ai_sapiens_entry_version': 1,
      'ai_sapiens_entry_spec': self.entry_spec()})

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    if load_cfg is not None:
      return super().load(path, load_cfg, strict, map_location)
    metadata = torch.load(path, map_location='cpu', weights_only=False).get('infos') or {}
    if metadata.get('ai_sapiens_entry_version') == 1:
      infos = super().load(path, None, strict, map_location)
      self.current_learning_iteration += 1
      self.alg.learning_rate = self.alg.optimizer.param_groups[0]['lr']
    else:
      infos = MjlabOnPolicyRunner.load(self, path, {'actor': True, 'critic': True}, strict, map_location)
      self.current_learning_iteration = 0
      self.env.unwrapped.common_step_counter = 0
      c = self.env.unwrapped.command_manager.get_term('reference_trajectory')
      c.load_curriculum_state_dict(dict(difficulty_stage=0, attempts=0, successes=0,
        last_success_rate=0., completed_successes=0, failed_attempts=0, interrupted_attempts=0))
      self.env.reset()
      print('[INFO] AiSapiens warm start: actor/critic loaded; optimizer, iteration and curriculum reset.')
    self._refresh()
    return infos
