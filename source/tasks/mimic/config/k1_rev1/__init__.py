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
from source.tasks.mimic.curriculum_runner import StandUpCurriculumRunner, KoryoCurriculumRunner, Taeguek01CurriculumRunner, Taeguek02CurriculumRunner

from .agents.rsl_rl_ppo_cfg import k1_rev1_mimic_ppo_runner_cfg
from .dance1_env_cfg import k1_rev1_dance1_env_cfg
from .dance2_env_cfg import k1_rev1_dance2_env_cfg
from .koryo_env_cfg import k1_rev1_koryo_env_cfg
from .koryo_curriculum_env_cfg import k1_rev1_koryo_curriculum_env_cfg
from .koryo_performance_env_cfg import k1_rev1_koryo_performance_env_cfg, k1_rev1_koryo_performance_ppo_cfg
from source.tasks.mimic.performance_runner import KoryoPerformanceRunner, KoryoContactV2Runner, KoryoContactCurriculumRunner
from .koryo_contact_v2_env_cfg import k1_rev1_koryo_contact_v2_env_cfg, k1_rev1_koryo_contact_v2_ppo_cfg, k1_rev1_koryo_contact_curriculum_env_cfg
from .koryo_contact_max_env_cfg import k1_rev1_koryo_contact_max_env_cfg, k1_rev1_koryo_contact_max_ppo_cfg
from source.tasks.mimic.performance_runner import KoryoContactMaxRunner
from source.tasks.mimic.performance_runner import GeumgangContactMaxRunner
from source.tasks.mimic.curriculum_runner import GeumgangCurriculumRunner
from .taeguek_collision_only_env_cfg import register_taeguek_collision_only_tasks
from .geumgang_env_cfg import (
  k1_rev1_geumgang_env_cfg, k1_rev1_geumgang_curriculum_env_cfg,
  k1_rev1_geumgang_contact_max_env_cfg, k1_rev1_geumgang_contact_max_ppo_cfg, k1_rev1_geumgang_ppo_cfg,
)

from .koryo_support_max_env_cfg import k1_rev1_koryo_support_max_env_cfg, k1_rev1_koryo_support_max_ppo_cfg
from source.tasks.mimic.performance_runner import KoryoSupportMaxRunner
from .taeguek01_env_cfg import k1_rev1_taeguek01_env_cfg
from .taeguek01_curriculum_env_cfg import k1_rev1_taeguek01_curriculum_env_cfg
from .taeguek02_env_cfg import k1_rev1_taeguek02_env_cfg
from .taeguek02_curriculum_env_cfg import k1_rev1_taeguek02_curriculum_env_cfg
from .taeguek_env_cfg import k1_rev1_taeguek_env_cfg
from .umr_getup_env_cfg import k1_rev1_umr_getup_env_cfg
from .umr_stand_up_env_cfg import k1_rev1_umr_stand_up_env_cfg
from .umr_stand_up_back_env_cfg import k1_rev1_umr_stand_up_back_env_cfg
from .umr_lie_down_env_cfg import k1_rev1_umr_lie_down_env_cfg
from .umr_lie_down_back_env_cfg import k1_rev1_umr_lie_down_back_env_cfg
from .umr_lie_down_back_soft_contact_env_cfg import k1_rev1_umr_lie_down_back_soft_contact_env_cfg
from .umr_lie_down_back_toe_stable_env_cfg import k1_rev1_umr_lie_down_back_toe_stable_env_cfg
from .umr_stand_up_low_friction_env_cfg import (
  k1_rev1_umr_stand_up_low_friction_env_cfg,
)
from .umr_stand_up_scratch_env_cfg import k1_rev1_umr_stand_up_scratch_env_cfg
from .umr_stand_up_soft_landing_env_cfg import k1_rev1_umr_stand_up_soft_landing_env_cfg
from .umr_stand_up_ready_handoff_env_cfg import k1_rev1_umr_stand_up_ready_handoff_env_cfg
from source.tasks.mimic.ready_handoff_runner import ReadyHandoffRunner

register_mjlab_task(
  task_id="Cyclo-Mimic-K1-Rev1-UMR-StandUp-ReadyHandoff",
  env_cfg=k1_rev1_umr_stand_up_ready_handoff_env_cfg(),
  play_env_cfg=k1_rev1_umr_stand_up_ready_handoff_env_cfg(play=True),
  rl_cfg=k1_rev1_mimic_ppo_runner_cfg(),
  runner_cls=ReadyHandoffRunner,
)


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
  task_id="Cyclo-Mimic-K1-Rev1-UMR-LieDownBack",
  env_cfg=k1_rev1_umr_lie_down_back_env_cfg(),
  play_env_cfg=k1_rev1_umr_lie_down_back_env_cfg(play=True),
  rl_cfg=k1_rev1_mimic_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Cyclo-Mimic-K1-Rev1-UMR-StandUpBack",
  env_cfg=k1_rev1_umr_stand_up_back_env_cfg(),
  play_env_cfg=k1_rev1_umr_stand_up_back_env_cfg(play=True),
  rl_cfg=k1_rev1_mimic_ppo_runner_cfg(),
  runner_cls=StandUpCurriculumRunner,
)

register_mjlab_task(
  task_id="Cyclo-Mimic-K1-Rev1-UMR-LieDownBack-SoftContact",
  env_cfg=k1_rev1_umr_lie_down_back_soft_contact_env_cfg(),
  play_env_cfg=k1_rev1_umr_lie_down_back_soft_contact_env_cfg(play=True),
  rl_cfg=k1_rev1_mimic_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Cyclo-Mimic-K1-Rev1-UMR-LieDownBack-ToeStable",
  env_cfg=k1_rev1_umr_lie_down_back_toe_stable_env_cfg(),
  play_env_cfg=k1_rev1_umr_lie_down_back_toe_stable_env_cfg(play=True),
  rl_cfg=k1_rev1_mimic_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Cyclo-Mimic-K1-Rev1-Koryo",
  env_cfg=k1_rev1_koryo_env_cfg(),
  play_env_cfg=k1_rev1_koryo_env_cfg(play=True),
  rl_cfg=k1_rev1_mimic_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id='Cyclo-Mimic-K1-Rev1-Koryo-Performance',
  env_cfg=k1_rev1_koryo_performance_env_cfg(),
  play_env_cfg=k1_rev1_koryo_performance_env_cfg(play=True),
  rl_cfg=k1_rev1_koryo_performance_ppo_cfg(),
  runner_cls=KoryoPerformanceRunner,
)

register_mjlab_task(
  task_id='Cyclo-Mimic-K1-Rev1-Koryo-ContactV2',
  env_cfg=k1_rev1_koryo_contact_v2_env_cfg(),
  play_env_cfg=k1_rev1_koryo_contact_v2_env_cfg(play=True),
  rl_cfg=k1_rev1_koryo_contact_v2_ppo_cfg(),
  runner_cls=KoryoContactV2Runner,
)

register_mjlab_task(
  task_id='Cyclo-Mimic-K1-Rev1-Koryo-ContactV2-Curriculum',
  env_cfg=k1_rev1_koryo_contact_curriculum_env_cfg(),
  play_env_cfg=k1_rev1_koryo_contact_curriculum_env_cfg(play=True),
  rl_cfg=k1_rev1_koryo_contact_v2_ppo_cfg(),
  runner_cls=KoryoContactCurriculumRunner,
)

register_mjlab_task(
  task_id='Cyclo-Mimic-K1-Rev1-Koryo-ContactV2-Max',
  env_cfg=k1_rev1_koryo_contact_max_env_cfg(),
  play_env_cfg=k1_rev1_koryo_contact_max_env_cfg(play=True),
  rl_cfg=k1_rev1_koryo_contact_max_ppo_cfg(),
  runner_cls=KoryoContactMaxRunner,
)

register_mjlab_task(
  task_id='Cyclo-Mimic-K1-Rev1-Geumgang',
  env_cfg=k1_rev1_geumgang_env_cfg(),
  play_env_cfg=k1_rev1_geumgang_env_cfg(play=True),
  rl_cfg=k1_rev1_geumgang_ppo_cfg(),
)

register_mjlab_task(
  task_id='Cyclo-Mimic-K1-Rev1-Geumgang-Curriculum',
  env_cfg=k1_rev1_geumgang_curriculum_env_cfg(),
  play_env_cfg=k1_rev1_geumgang_curriculum_env_cfg(play=True),
  rl_cfg=k1_rev1_geumgang_ppo_cfg(),
  runner_cls=GeumgangCurriculumRunner,
)

register_mjlab_task(
  task_id='Cyclo-Mimic-K1-Rev1-Geumgang-ContactV2-Max',
  env_cfg=k1_rev1_geumgang_contact_max_env_cfg(),
  play_env_cfg=k1_rev1_geumgang_contact_max_env_cfg(play=True),
  rl_cfg=k1_rev1_geumgang_contact_max_ppo_cfg(),
  runner_cls=GeumgangContactMaxRunner,
)

register_mjlab_task(
  task_id='Cyclo-Mimic-K1-Rev1-Koryo-SupportV3-Max',
  env_cfg=k1_rev1_koryo_support_max_env_cfg(),
  play_env_cfg=k1_rev1_koryo_support_max_env_cfg(play=True),
  rl_cfg=k1_rev1_koryo_support_max_ppo_cfg(),
  runner_cls=KoryoSupportMaxRunner,
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

register_mjlab_task(
  task_id="Cyclo-Mimic-K1-Rev1-Taeguek01-Curriculum",
  env_cfg=k1_rev1_taeguek01_curriculum_env_cfg(),
  play_env_cfg=k1_rev1_taeguek01_curriculum_env_cfg(play=True),
  rl_cfg=k1_rev1_mimic_ppo_runner_cfg(),
  runner_cls=Taeguek01CurriculumRunner,
)

register_mjlab_task(
  task_id="Cyclo-Mimic-K1-Rev1-Taeguek02",
  env_cfg=k1_rev1_taeguek02_env_cfg(),
  play_env_cfg=k1_rev1_taeguek02_env_cfg(play=True),
  rl_cfg=k1_rev1_mimic_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Cyclo-Mimic-K1-Rev1-Taeguek02-Curriculum",
  env_cfg=k1_rev1_taeguek02_curriculum_env_cfg(),
  play_env_cfg=k1_rev1_taeguek02_curriculum_env_cfg(play=True),
  rl_cfg=k1_rev1_mimic_ppo_runner_cfg(),
  runner_cls=Taeguek02CurriculumRunner,
)


register_taeguek_collision_only_tasks()

# Standalone deployment-matched fine-tune; existing tasks retain their behavior.
from .umr_stand_up_ai_sapiens_env_cfg import (
  k1_rev1_umr_stand_up_ai_sapiens_env_cfg,
  k1_rev1_umr_stand_up_ai_sapiens_ppo_cfg,
)
from source.tasks.mimic.ai_sapiens_runner import AiSapiensRunner
register_mjlab_task(
  task_id="Cyclo-Mimic-K1-Rev1-UMR-StandUp-AiSapiens",
  env_cfg=k1_rev1_umr_stand_up_ai_sapiens_env_cfg(),
  play_env_cfg=k1_rev1_umr_stand_up_ai_sapiens_env_cfg(play=True),
  rl_cfg=k1_rev1_umr_stand_up_ai_sapiens_ppo_cfg(),
  runner_cls=AiSapiensRunner,
)

# Settled reference and early body dynamics fine-tune, isolated from prior tasks.
from .umr_stand_up_soft_entry_env_cfg import soft_entry_env_cfg, soft_entry_ppo_cfg
from source.tasks.mimic.soft_entry_runner import SoftEntryRunner
register_mjlab_task(
  task_id="Cyclo-Mimic-K1-Rev1-UMR-StandUp-AiSapiens-SoftEntry",
  env_cfg=soft_entry_env_cfg(), play_env_cfg=soft_entry_env_cfg(play=True),
  rl_cfg=soft_entry_ppo_cfg(), runner_cls=SoftEntryRunner,
)


# SoftEntry with restored original StandUp domain randomization.
from .umr_stand_up_soft_entry_dr_env_cfg import soft_entry_dr_env_cfg, soft_entry_dr_ppo_cfg
from source.tasks.mimic.soft_entry_dr_runner import SoftEntryDRRunner
register_mjlab_task(
  task_id="Cyclo-Mimic-K1-Rev1-UMR-StandUp-AiSapiens-SoftEntry-DR",
  env_cfg=soft_entry_dr_env_cfg(), play_env_cfg=soft_entry_dr_env_cfg(play=True),
  rl_cfg=soft_entry_dr_ppo_cfg(), runner_cls=SoftEntryDRRunner,
)

for chapter in range(3, 9):
  register_mjlab_task(
    task_id=f"Cyclo-Mimic-K1-Rev1-Taeguek{chapter:02d}",
    env_cfg=k1_rev1_taeguek_env_cfg(chapter),
    play_env_cfg=k1_rev1_taeguek_env_cfg(chapter, play=True),
    rl_cfg=k1_rev1_mimic_ppo_runner_cfg(),
  )

# Separate scratch tasks for phase-aware support refinement, chapters 2–8.
from .taeguek_support_env_cfg import k1_rev1_taeguek_support_env_cfg, k1_rev1_taeguek_support_ppo_cfg
for _support_chapter in range(2,9):
  register_mjlab_task(
    task_id=f"Cyclo-Mimic-K1-Rev1-Taeguek{_support_chapter:02d}-SupportV1",
    env_cfg=k1_rev1_taeguek_support_env_cfg(_support_chapter),
    play_env_cfg=k1_rev1_taeguek_support_env_cfg(_support_chapter,play=True),
    rl_cfg=k1_rev1_taeguek_support_ppo_cfg(),
  )


# Isolated refinement tasks: V1 configurations and running jobs stay unchanged.
from .taeguek_support_v2_env_cfg import k1_rev1_taeguek_support_v2_env_cfg, k1_rev1_taeguek_support_v2_ppo_cfg
from source.tasks.mimic.taeguek_support_v2_runner import TaegeukSupportV2Runner
for _v2_chapter in (4, 5):
  register_mjlab_task(
    task_id=f'Cyclo-Mimic-K1-Rev1-Taeguek{_v2_chapter:02d}-SupportV2',
    env_cfg=k1_rev1_taeguek_support_v2_env_cfg(_v2_chapter),
    play_env_cfg=k1_rev1_taeguek_support_v2_env_cfg(_v2_chapter, play=True),
    rl_cfg=k1_rev1_taeguek_support_v2_ppo_cfg(),
    runner_cls=TaegeukSupportV2Runner,
  )

# Limbo remains isolated from the existing UMR/Taekwondo tasks.
from . import limbo_tasks as _limbo_tasks  # noqa: F401
