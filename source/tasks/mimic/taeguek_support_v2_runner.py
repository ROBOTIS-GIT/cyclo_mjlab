# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Warm-start V1 weights with a fresh optimizer; resume V2 normally."""
import hashlib
from pathlib import Path
import torch
from mjlab.rl import MjlabOnPolicyRunner


class TaegeukSupportV2Runner(MjlabOnPolicyRunner):
  def load(self, path, load_cfg=None, strict=True, map_location=None):
    data = torch.load(path, map_location='cpu', weights_only=False)
    marker = (data.get('infos') or {}).get('taeguek_support_v2')
    if marker is not None:
      self.warmstart_provenance = marker
      return super().load(path, load_cfg, strict, map_location)
    if load_cfg is not None:
      raise ValueError('V1 warm-start has a fixed actor/critic-only load policy.')
    if self.alg.optimizer.state:
      raise RuntimeError('Warm-start requires a fresh optimizer.')
    infos = super().load(path, dict(actor=True, critic=True, optimizer=False,
                                    iteration=False, rnd=False), strict, map_location)
    self.current_learning_iteration = 0
    self.env.unwrapped.common_step_counter = 0
    for model, key in [(self.alg.actor,'actor_state_dict'),(self.alg.critic,'critic_state_dict')]:
      state = model.state_dict()
      for name, value in data[key].items():
        if not torch.equal(state[name].cpu(),value):
          raise RuntimeError(f'Warm-start tensor mismatch: {key}/{name}')
    rates = [group['lr'] for group in self.alg.optimizer.param_groups]
    if self.alg.optimizer.state or any(rate != self.alg.learning_rate for rate in rates):
      raise RuntimeError('Source optimizer/rate leaked into fine-tuning.')
    self.warmstart_provenance = dict(source=str(Path(path).resolve()),
      sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
      source_iteration=int(data['iter']), fresh_optimizer=True,
      learning_rate=self.alg.learning_rate)
    print(f'SUPPORT_V2_WARMSTART {self.warmstart_provenance}',flush=True)
    return infos

  def save(self, path, infos=None):
    metadata = {**(infos or {}), 'taeguek_support_v2':getattr(self,'warmstart_provenance',{})}
    super().save(path, metadata)
