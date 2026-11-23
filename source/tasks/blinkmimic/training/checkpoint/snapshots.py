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

"""
Bridge the upstream trainer's synchronous save calls in its isolated process.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass
class _SavedCritic:
    directory: Path
    iteration: int
    state: dict


@contextmanager
def include_critic_in_snapshots():
    """
    Bridge last.pt -> snapshot saves in the isolated, synchronous trainer.

    Upstream has no checkpoint callback. Restrict interception to its save
    paths;
    reject mismatched iterations and restore torch.save even if training fails.
    The worker rejects asynchronous --ckpt_light saves before entering this
    scope.
    """
    import torch

    original_save = torch.save
    latest = None

    def save(value, file, *args, **kwargs):
        nonlocal latest
        path = Path(file) if isinstance(file, (str, Path)) else None
        if path is not None and re.fullmatch(
            r"ckpt_[0-9]+\.pt\.tmp", path.name
        ):
            if (
                latest is None
                or latest.directory != path.parent
                or latest.iteration != value.get("iteration")
            ):
                raise RuntimeError(
                    "BlinkMimic snapshot has no matching latest critic"
                )
            value = dict(value, critic=latest.state)
        result = original_save(value, file, *args, **kwargs)
        if path is not None and path.name == "last.pt.tmp":
            latest = _SavedCritic(
                path.parent, value["iteration"], value["critic"]
            )
        return result

    torch.save = save
    try:
        yield
    finally:
        torch.save = original_save
