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
Augment training snapshots while preserving upstream policy-only saves.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass
class _SnapshotExtras:
    directory: Path
    iteration: int
    state: dict
    control: dict | None


@contextmanager
def include_critic_in_snapshots(history=None, clips=None):
    """
    Include critic and sampler state in numbered snapshots, and carry
    inactive bin history in both full and numbered checkpoints.

    Upstream has no checkpoint callback. Restrict interception to its save
    paths;
    reject mismatched iterations and restore torch.save even if training fails.
    Policy-only --ckpt_light snapshots pass through unchanged, including
    background writes. The final full checkpoint still carries resume state.
    """
    import torch

    original_save = torch.save
    latest = None

    def save(value, file, *args, **kwargs):
        nonlocal latest
        path = Path(file) if isinstance(file, (str, Path)) else None
        if path is None:
            return original_save(value, file, *args, **kwargs)
        if path.name == "last.pt.tmp":
            control = value.get("ctrl")
            if control is not None and clips is not None:
                control = dict(control, clips=clips)
                value = dict(value, ctrl=control)
            latest = _SnapshotExtras(path.parent, value["iteration"], value["critic"], control)
        elif re.fullmatch(r"ckpt_[0-9]+\.pt\.tmp", path.name):
            if "pi" in value and "wm" not in value:
                return original_save(value, file, *args, **kwargs)
            if (
                latest is None
                or latest.directory != path.parent
                or latest.iteration != value.get("iteration")
            ):
                raise RuntimeError(
                    "BlinkMimic snapshot has no matching latest critic"
                )
            value = dict(value, critic=latest.state, ctrl=latest.control)
        else:
            return original_save(value, file, *args, **kwargs)
        if history:
            value = dict(value, inactive_sampler=history)
        return original_save(value, file, *args, **kwargs)

    torch.save = save
    try:
        yield
    finally:
        torch.save = original_save
