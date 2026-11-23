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

"""Script to list mjlab environments."""

import tyro
from prettytable import PrettyTable

import mjlab
import mjlab.tasks  # noqa: F401
import source.tasks
from mjlab.tasks.registry import list_tasks
from source.tasks.blinkmimic.config import TASKS as BLINKMIMIC_TASKS


def list_environments(keyword: str | None = None):
  """List all registered environments.

  Args:
    keyword: Optional filter to only show environments containing this keyword.
  """
  table = PrettyTable(["#", "Task ID"])
  table.title = "Available Environments in mjlab"
  table.align["Task ID"] = "l"

  tasks = {name for name in list_tasks() if name.startswith("Cyclo-")}
  tasks.update(BLINKMIMIC_TASKS)
  matches = sorted(name for name in tasks if not keyword or keyword.lower() in name.lower())
  for index, task_id in enumerate(matches, start=1):
    table.add_row([index, task_id])

  idx = len(matches)
  print(table)
  if idx == 0:
    msg = "[INFO] No tasks matched"
    if keyword:
      msg += f" keyword '{keyword}'"
    print(msg)
  return idx


def main():
  return tyro.cli(list_environments, config=mjlab.TYRO_FLAGS)


if __name__ == "__main__":
  main()
