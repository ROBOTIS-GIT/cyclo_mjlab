# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""K1 Rev.1 face-down lie-down task for the UMR lie_down_back asset."""
from mjlab.envs import ManagerBasedRlEnvCfg
from source import SRC_PATH
from .umr_lie_down_env_cfg import k1_rev1_umr_lie_down_env_cfg


TRAJECTORY_FILE = (
  SRC_PATH / "assets" / "motions" / "K1_rev1" / "umr_lie_down_back"
  / "k1_lie_down_back_30hz_converted.npz"
)


def k1_rev1_umr_lie_down_back_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = k1_rev1_umr_lie_down_env_cfg(play=play)
  cfg.commands["reference_trajectory"].trajectory_file = str(TRAJECTORY_FILE)
  # Deep reference poses exceed the 90% soft range but respect hard limits.
  articulation = cfg.scene.entities["robot"].articulation
  assert articulation is not None
  articulation.soft_joint_pos_limit_factor = 1.0
  return cfg
