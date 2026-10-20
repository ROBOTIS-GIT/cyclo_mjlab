# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Select a new Taegeuk 1–8 reference for ContactV2-Max training."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import source.tasks  # noqa: F401
from scripts.reinforcement_learning.train import TrainConfig, launch_training
from source.tasks.mimic.motion_curriculum import MOTIONS, TAEGUEK_COLLISION_ONLY_NAMES, motion_identity


def build_config(chapter, num_envs=2048, max_iterations=60000, seed=42, checkpoint=None):
  if chapter not in range(1, 9) or num_envs < 10 or max_iterations < 1:
    raise ValueError('Choose chapter 1–8, at least 10 environments and a positive iteration count.')
  name = TAEGUEK_COLLISION_ONLY_NAMES[chapter - 1]
  task = MOTIONS[name].task + '-ContactV2-Max'
  cfg = TrainConfig.from_task(task)
  cfg.env.scene.num_envs = num_envs
  cfg.agent.max_iterations = max_iterations
  cfg.agent.seed = seed
  cfg.agent.upload_model = False
  if checkpoint is not None:
    cfg = replace(cfg, checkpoint=str(checkpoint))
  return task, cfg


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--chapter', type=int, choices=range(1, 9), required=True)
  parser.add_argument('--num-envs', type=int, default=2048)
  parser.add_argument('--max-iterations', type=int, default=60000,
                      help='Number of updates to run (additional updates when resuming).')
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--checkpoint', type=Path)
  parser.add_argument('--dry-run', action='store_true')
  args = parser.parse_args()
  if args.checkpoint is not None and not args.checkpoint.is_file():
    parser.error('Checkpoint file does not exist.')
  try:
    task, cfg = build_config(args.chapter, args.num_envs, args.max_iterations, args.seed, args.checkpoint)
  except ValueError as error:
    parser.error(str(error))
  if args.dry_run:
    print(json.dumps({'task':task, 'motion':motion_identity(TAEGUEK_COLLISION_ONLY_NAMES[args.chapter-1]),
      'run_name':cfg.agent.run_name, 'num_envs':cfg.env.scene.num_envs,
      'max_iterations':cfg.agent.max_iterations, 'checkpoint':cfg.checkpoint,
      'learning_rate':cfg.agent.algorithm.learning_rate, 'schedule':cfg.agent.algorithm.schedule,
      'entropy_coef':cfg.agent.algorithm.entropy_coef, 'noise_scale':1., 'push_scale':1.}, indent=2))
    return
  launch_training(task, cfg)


if __name__ == '__main__':
  main()
