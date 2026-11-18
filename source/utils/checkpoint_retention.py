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

"""Retention for numbered model checkpoints shared by training adapters."""

import re
from pathlib import Path


def prune_model_checkpoints(directory):
    """Keep the newest five snapshots and every 10,000-iteration milestone."""
    snapshots = []
    for path in Path(directory).glob("model_*.pt"):
        match = re.fullmatch(r"model_([0-9]+)\.pt", path.name)
        if match and path.is_file():
            snapshots.append((int(match[1]), path))
    for step, path in sorted(snapshots, reverse=True)[5:]:
        if step % 10000 != 0:
            path.unlink()
