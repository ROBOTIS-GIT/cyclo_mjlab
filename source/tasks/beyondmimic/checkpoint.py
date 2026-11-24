# Copyright 2026 ROBOTIS CO., LTD.
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
# Author: Kiwoong Park

"""Find a task's checkpoints in the shared BeyondMimic experiment directory."""

import re
from pathlib import Path

import yaml

from source import SRC_PATH


def _matches_task(run, task_id, default_motion):
  # New runs are named <timestamp>_<task_id>[_<run_name>], even with a motion override.
  tagged = re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(Cyclo-BeyondMimic-[^_]+)(?:_|$)", run.name)
  if tagged:
    return tagged[1] == task_id

  # Older runs only identify their motion in env.yaml. Host and container
  # prefixes may differ; retain the full path within the motion library.
  try:
    with (run / "params/env.yaml").open() as stream:
      env = yaml.load(stream, Loader=yaml.BaseLoader)
    saved = Path(env["commands"]["reference_trajectory"]["trajectory_file"])
  except (FileNotFoundError, KeyError, TypeError, yaml.YAMLError):
    return False
  if saved == default_motion:
    return True
  if default_motion.is_relative_to(SRC_PATH / "assets/motions"):
    suffix = default_motion.relative_to(SRC_PATH.parent).parts
    return saved.parts[-len(suffix):] == suffix
  return False


def tracking_checkpoint_path(root, task_id, default_motion, run_pattern, checkpoint_pattern):
  from mjlab.utils.os import get_checkpoint_path

  # A specific run is an explicit choice, including for cross-task transfer.
  explicit = root / run_pattern
  if explicit.is_dir():
    return get_checkpoint_path(explicit.parent, re.escape(explicit.name) + "$", checkpoint_pattern)

  for run in sorted(root.glob("*"), reverse=True):
    if not run.is_dir() or run.name == "wandb_checkpoints" or not re.match(run_pattern, run.name):
      continue
    if _matches_task(run, task_id, Path(default_motion)):
      try:
        return get_checkpoint_path(root, re.escape(run.name) + "$", checkpoint_pattern)
      except ValueError:
        continue  # A matching training run may not have saved its first checkpoint yet.
  raise ValueError(f"No matching checkpoint for {task_id} in {root}. Select an existing run or checkpoint explicitly.")
