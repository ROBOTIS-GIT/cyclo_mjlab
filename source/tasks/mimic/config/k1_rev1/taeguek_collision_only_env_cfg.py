# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Versioned Taegeuk 1–8 references with the Koryo ContactV2-Max settings."""
from mjlab.tasks.registry import register_mjlab_task
from source.tasks.mimic.motion_curriculum import MOTIONS, TAEGUEK_COLLISION_ONLY_NAMES
from source.tasks.mimic.performance_runner import KoryoContactMaxRunner
from source.tasks.mimic.curriculum_runner import KoryoCurriculumRunner
from .agents.rsl_rl_ppo_cfg import k1_rev1_mimic_ppo_runner_cfg
from .koryo_env_cfg import k1_rev1_koryo_env_cfg
from .koryo_curriculum_env_cfg import apply_taekwondo_curriculum
from .koryo_contact_max_env_cfg import k1_rev1_koryo_contact_max_env_cfg, k1_rev1_koryo_contact_max_ppo_cfg


def motion_spec(chapter):
  if chapter not in range(1, 9):
    raise ValueError('Taegeuk chapter must be between 1 and 8.')
  return MOTIONS[TAEGUEK_COLLISION_ONLY_NAMES[chapter - 1]]


def taeguek_collision_only_env_cfg(chapter, play=False, contact_max=False, curriculum=False):
  if contact_max and curriculum:
    raise ValueError('Select either fixed maximum disturbances or the ordinary curriculum.')
  cfg = (k1_rev1_koryo_contact_max_env_cfg if contact_max else k1_rev1_koryo_env_cfg)(play=play)
  if curriculum and not play:
    cfg = apply_taekwondo_curriculum(cfg)
  cfg.commands['reference_trajectory'].trajectory_file = str(motion_spec(chapter).trajectory)
  return cfg


def taeguek_collision_only_ppo_cfg(chapter, contact_max=False):
  cfg = (k1_rev1_koryo_contact_max_ppo_cfg if contact_max else k1_rev1_mimic_ppo_runner_cfg)()
  cfg.run_name = motion_spec(chapter).name + ('_contact_max_from_scratch' if contact_max else '')
  return cfg


class Taeguek01CollisionOnlyContactMaxRunner(KoryoContactMaxRunner):
  motion_name = "taeguek01_collision_only"


class Taeguek01CollisionOnlyCurriculumRunner(KoryoCurriculumRunner):
  motion_name = "taeguek01_collision_only"


class Taeguek02CollisionOnlyContactMaxRunner(KoryoContactMaxRunner):
  motion_name = "taeguek02_collision_only"


class Taeguek02CollisionOnlyCurriculumRunner(KoryoCurriculumRunner):
  motion_name = "taeguek02_collision_only"


class Taeguek03CollisionOnlyContactMaxRunner(KoryoContactMaxRunner):
  motion_name = "taeguek03_collision_only"


class Taeguek03CollisionOnlyCurriculumRunner(KoryoCurriculumRunner):
  motion_name = "taeguek03_collision_only"


class Taeguek04CollisionOnlyContactMaxRunner(KoryoContactMaxRunner):
  motion_name = "taeguek04_collision_only"


class Taeguek04CollisionOnlyCurriculumRunner(KoryoCurriculumRunner):
  motion_name = "taeguek04_collision_only"


class Taeguek05CollisionOnlyContactMaxRunner(KoryoContactMaxRunner):
  motion_name = "taeguek05_collision_only"


class Taeguek05CollisionOnlyCurriculumRunner(KoryoCurriculumRunner):
  motion_name = "taeguek05_collision_only"


class Taeguek06CollisionOnlyContactMaxRunner(KoryoContactMaxRunner):
  motion_name = "taeguek06_collision_only"


class Taeguek06CollisionOnlyCurriculumRunner(KoryoCurriculumRunner):
  motion_name = "taeguek06_collision_only"


class Taeguek07CollisionOnlyContactMaxRunner(KoryoContactMaxRunner):
  motion_name = "taeguek07_collision_only"


class Taeguek07CollisionOnlyCurriculumRunner(KoryoCurriculumRunner):
  motion_name = "taeguek07_collision_only"


class Taeguek08CollisionOnlyContactMaxRunner(KoryoContactMaxRunner):
  motion_name = "taeguek08_collision_only"


class Taeguek08CollisionOnlyCurriculumRunner(KoryoCurriculumRunner):
  motion_name = "taeguek08_collision_only"


RUNNERS = {
  1: (Taeguek01CollisionOnlyCurriculumRunner, Taeguek01CollisionOnlyContactMaxRunner),
  2: (Taeguek02CollisionOnlyCurriculumRunner, Taeguek02CollisionOnlyContactMaxRunner),
  3: (Taeguek03CollisionOnlyCurriculumRunner, Taeguek03CollisionOnlyContactMaxRunner),
  4: (Taeguek04CollisionOnlyCurriculumRunner, Taeguek04CollisionOnlyContactMaxRunner),
  5: (Taeguek05CollisionOnlyCurriculumRunner, Taeguek05CollisionOnlyContactMaxRunner),
  6: (Taeguek06CollisionOnlyCurriculumRunner, Taeguek06CollisionOnlyContactMaxRunner),
  7: (Taeguek07CollisionOnlyCurriculumRunner, Taeguek07CollisionOnlyContactMaxRunner),
  8: (Taeguek08CollisionOnlyCurriculumRunner, Taeguek08CollisionOnlyContactMaxRunner),
}


def register_taeguek_collision_only_tasks():
  for chapter, (curriculum_runner, contact_runner) in RUNNERS.items():
    spec = motion_spec(chapter)
    register_mjlab_task(
      task_id=spec.task,
      env_cfg=taeguek_collision_only_env_cfg(chapter),
      play_env_cfg=taeguek_collision_only_env_cfg(chapter, play=True),
      rl_cfg=taeguek_collision_only_ppo_cfg(chapter),
    )
    register_mjlab_task(
      task_id=spec.training_task,
      env_cfg=taeguek_collision_only_env_cfg(chapter, curriculum=True),
      play_env_cfg=taeguek_collision_only_env_cfg(chapter, play=True, curriculum=True),
      rl_cfg=taeguek_collision_only_ppo_cfg(chapter),
      runner_cls=curriculum_runner,
    )
    register_mjlab_task(
      task_id=spec.task + '-ContactV2-Max',
      env_cfg=taeguek_collision_only_env_cfg(chapter, contact_max=True),
      play_env_cfg=taeguek_collision_only_env_cfg(chapter, play=True, contact_max=True),
      rl_cfg=taeguek_collision_only_ppo_cfg(chapter, contact_max=True),
      runner_cls=contact_runner,
    )
