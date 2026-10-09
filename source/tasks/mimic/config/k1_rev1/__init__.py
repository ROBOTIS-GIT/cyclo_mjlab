# Copyright 2026 ROBOTIS CO., LTD.
# Copyright 2025, The mjlab Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Insu Park
#
# This file includes modifications by ROBOTIS CO., LTD. to code derived
# from mujocolab/mjlab.

"""K1 Rev.1 Mimic task registration."""

from mjlab.tasks.registry import register_mjlab_task
from source.tasks.mimic.curriculum_runner import StandUpCurriculumRunner, KoryoCurriculumRunner

from .agents.rsl_rl_ppo_cfg import k1_rev1_mimic_ppo_runner_cfg
from .dance1_env_cfg import k1_rev1_dance1_env_cfg
from .dance2_env_cfg import k1_rev1_dance2_env_cfg
from .koryo_env_cfg import k1_rev1_koryo_env_cfg
from .koryo_curriculum_env_cfg import k1_rev1_koryo_curriculum_env_cfg
from .taeguek01_env_cfg import k1_rev1_taeguek01_env_cfg
from .umr_getup_env_cfg import k1_rev1_umr_getup_env_cfg
from .umr_stand_up_env_cfg import k1_rev1_umr_stand_up_env_cfg
from .umr_lie_down_env_cfg import k1_rev1_umr_lie_down_env_cfg
from .umr_stand_up_low_friction_env_cfg import (
  k1_rev1_umr_stand_up_low_friction_env_cfg,
)
from .umr_stand_up_scratch_env_cfg import k1_rev1_umr_stand_up_scratch_env_cfg
from .umr_stand_up_soft_landing_env_cfg import k1_rev1_umr_stand_up_soft_landing_env_cfg


register_mjlab_task(
  task_id="Cyclo-Mimic-K1-Rev1-Dance1",
  env_cfg=k1_rev1_dance1_env_cfg(),
  play_env_cfg=k1_rev1_dance1_env_cfg(play=True),
  rl_cfg=k1_rev1_mimic_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Cyclo-Mimic-K1-Rev1-Dance2",
  env_cfg=k1_rev1_dance2_env_cfg(),
  play_env_cfg=k1_rev1_dance2_env_cfg(play=True),
  rl_cfg=k1_rev1_mimic_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Cyclo-Mimic-K1-Rev1-UMR-Getup",
  env_cfg=k1_rev1_umr_getup_env_cfg(),
  play_env_cfg=k1_rev1_umr_getup_env_cfg(play=True),
  rl_cfg=k1_rev1_mimic_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Cyclo-Mimic-K1-Rev1-UMR-LieDown",
  env_cfg=k1_rev1_umr_lie_down_env_cfg(),
  play_env_cfg=k1_rev1_umr_lie_down_env_cfg(play=True),
  rl_cfg=k1_rev1_mimic_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Cyclo-Mimic-K1-Rev1-UMR-StandUp",
  env_cfg=k1_rev1_umr_stand_up_env_cfg(),
  play_env_cfg=k1_rev1_umr_stand_up_env_cfg(play=True),
  rl_cfg=k1_rev1_mimic_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Cyclo-Mimic-K1-Rev1-UMR-StandUp-LowFriction",
  env_cfg=k1_rev1_umr_stand_up_low_friction_env_cfg(),
  play_env_cfg=k1_rev1_umr_stand_up_low_friction_env_cfg(play=True),
  rl_cfg=k1_rev1_mimic_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Cyclo-Mimic-K1-Rev1-UMR-StandUp-Scratch",
  env_cfg=k1_rev1_umr_stand_up_scratch_env_cfg(),
  play_env_cfg=k1_rev1_umr_stand_up_scratch_env_cfg(play=True),
  rl_cfg=k1_rev1_mimic_ppo_runner_cfg(),
  runner_cls=StandUpCurriculumRunner,
)


register_mjlab_task(
  task_id="Cyclo-Mimic-K1-Rev1-UMR-StandUp-SoftLanding",
  env_cfg=k1_rev1_umr_stand_up_soft_landing_env_cfg(),
  play_env_cfg=k1_rev1_umr_stand_up_soft_landing_env_cfg(play=True),
  rl_cfg=k1_rev1_mimic_ppo_runner_cfg(),
  runner_cls=StandUpCurriculumRunner,
)

register_mjlab_task(
  task_id="Cyclo-Mimic-K1-Rev1-Koryo",
  env_cfg=k1_rev1_koryo_env_cfg(),
  play_env_cfg=k1_rev1_koryo_env_cfg(play=True),
  rl_cfg=k1_rev1_mimic_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Cyclo-Mimic-K1-Rev1-Taeguek01",
  env_cfg=k1_rev1_taeguek01_env_cfg(),
  play_env_cfg=k1_rev1_taeguek01_env_cfg(play=True),
  rl_cfg=k1_rev1_mimic_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Cyclo-Mimic-K1-Rev1-Koryo-Curriculum",
  env_cfg=k1_rev1_koryo_curriculum_env_cfg(),
  play_env_cfg=k1_rev1_koryo_curriculum_env_cfg(play=True),
  rl_cfg=k1_rev1_mimic_ppo_runner_cfg(),
  runner_cls=KoryoCurriculumRunner,
)
