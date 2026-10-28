# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Rev.2 Taegeuk 1 only; fresh PPO, 50 Hz reference and 200 Hz physics."""
from source import SRC_PATH
from source.assets.robots.robotis_k1.k1_rev2 import get_k1_rev2_cfg, K1_REV2_ACTION_SCALE
from source.tasks.mimic.config.k1_rev1.taeguek_env_cfg import k1_rev1_taeguek_env_cfg
from source.tasks.mimic.config.k1_rev1.agents.rsl_rl_ppo_cfg import k1_rev1_mimic_ppo_runner_cfg

TRAJECTORY_FILE = SRC_PATH / 'assets/motions/K1_rev2/taeguek_01/k1_rev2_taeguek01_50hz.npz'


def k1_rev2_taeguek01_env_cfg(play=False):
    cfg = k1_rev1_taeguek_env_cfg(1, play=play)
    cfg.scene.entities['robot'] = get_k1_rev2_cfg()
    cfg.actions['joint_pos'].scale = K1_REV2_ACTION_SCALE.copy()
    reference = cfg.commands['reference_trajectory']
    reference.trajectory_file = str(TRAJECTORY_FILE)
    reference.first_frame_env_fraction = 0.2 if not play else 0.0
    reference.start_sampling_uniform_mix = 0.2
    # 31.62 s reference must not be truncated by Rev.1's 30 s episode limit.
    cfg.episode_length_s = 33.62 if not play else int(1e9)
    return cfg


def k1_rev2_taeguek01_ppo_cfg():
    cfg = k1_rev1_mimic_ppo_runner_cfg()
    cfg.experiment_name = 'k1_rev2_taeguek01'
    cfg.run_name = 'rev2_taeguek01_scratch'
    cfg.max_iterations = 30000
    cfg.resume = False
    return cfg
