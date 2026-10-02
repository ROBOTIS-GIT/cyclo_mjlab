# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Stand-up fine-tuning with a mixture of low and regular floor friction."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from source.tasks.mimic.mdp.friction_events import randomize_support_friction

from .umr_stand_up_env_cfg import k1_rev1_umr_stand_up_env_cfg


def k1_rev1_umr_stand_up_low_friction_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = k1_rev1_umr_stand_up_env_cfg(play=play)
  cfg.events["physics_material"] = EventTermCfg(
    mode="reset",
    func=randomize_support_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_collision.*",)),
      "terrain_cfg": SceneEntityCfg("terrain", geom_names=(".*",)),
      "low_range": (0.2, 0.6),
      "regular_range": (0.3, 1.2),
      "low_probability_start": 0.3,
      "low_probability_end": 0.7,
      # 2,000 PPO iterations at 24 policy steps per iteration.
      "ramp_steps": 48_000,
    },
  )
  if play:
    # Repeatable low-friction playback; no training mixture during evaluation.
    cfg.events["physics_material"].params.update(
      low_range=(0.4, 0.4),
      low_probability_start=1.0,
      low_probability_end=1.0,
    )
  return cfg
