# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Geumgang motion with the same robot, contact rewards and PPO as Koryo."""
from source import SRC_PATH
from .agents.rsl_rl_ppo_cfg import k1_rev1_mimic_ppo_runner_cfg
from .koryo_env_cfg import k1_rev1_koryo_env_cfg
from .koryo_curriculum_env_cfg import apply_taekwondo_curriculum
from .koryo_contact_max_env_cfg import (
  k1_rev1_koryo_contact_max_env_cfg, k1_rev1_koryo_contact_max_ppo_cfg,
)

TRAJECTORY_FILE = SRC_PATH / 'assets/motions/K1_rev1/geumgang/tkd_geumgang_converted.npz'


def k1_rev1_geumgang_env_cfg(play=False):
  cfg = k1_rev1_koryo_env_cfg(play=play)
  cfg.commands['reference_trajectory'].trajectory_file = str(TRAJECTORY_FILE)
  return cfg


def k1_rev1_geumgang_curriculum_env_cfg(play=False):
  cfg = k1_rev1_geumgang_env_cfg(play=play)
  return cfg if play else apply_taekwondo_curriculum(cfg)


def k1_rev1_geumgang_contact_max_env_cfg(play=False):
  cfg = k1_rev1_koryo_contact_max_env_cfg(play=play)
  cfg.commands['reference_trajectory'].trajectory_file = str(TRAJECTORY_FILE)
  return cfg


def k1_rev1_geumgang_contact_max_ppo_cfg():
  cfg = k1_rev1_koryo_contact_max_ppo_cfg()
  cfg.run_name = 'geumgang_contact_max_from_scratch'
  return cfg


def k1_rev1_geumgang_ppo_cfg():
  cfg = k1_rev1_mimic_ppo_runner_cfg()
  cfg.run_name = 'geumgang'
  return cfg
