# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Register only the requested Rev.2 Taegeuk 1 task."""
from mjlab.tasks.registry import register_mjlab_task
from .taeguek01_env_cfg import k1_rev2_taeguek01_env_cfg, k1_rev2_taeguek01_ppo_cfg

register_mjlab_task(
    task_id='Cyclo-Mimic-K1-Rev2-Taeguek01',
    env_cfg=k1_rev2_taeguek01_env_cfg(),
    play_env_cfg=k1_rev2_taeguek01_env_cfg(play=True),
    rl_cfg=k1_rev2_taeguek01_ppo_cfg(),
)
