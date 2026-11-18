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
Restore native training arguments; bridge model-only snapshots in the worker.
"""

from contextlib import contextmanager
from pathlib import Path

import torch


def read_resume_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    required = (
        "args",
        "pi",
        "wm",
        "critic",
        "iteration",
        "wall_time",
        "sim_steps",
    )
    missing = [key for key in required if key not in saved]
    if missing:
        raise ValueError(
            f"Checkpoint cannot resume training; missing: {', '.join(missing)}"
        )
    if type(saved["iteration"]) is not int or saved["iteration"] < 0:
        raise ValueError("Invalid checkpoint iteration")
    optimizer_keys = {"opt_pi", "opt_wm", "opt_v"}
    present = optimizer_keys.intersection(saved)
    if present and present != optimizer_keys:
        raise ValueError("Incomplete optimizer state in BlinkMimic checkpoint")
    return saved


def training_arguments(saved, robot_name):
    settings = saved["args"]
    if settings.get("robot") != robot_name:
        raise ValueError("Resume checkpoint belongs to a different robot")
    ignored = {"resume", "log_dir", "motion_dir", "robot", "ckpt_light"}
    arguments = []
    for name, value in settings.items():
        if name in ignored or value is None or value is False:
            continue
        arguments.append("--" + name)
        if value is not True:
            arguments.append(str(value))
    return arguments


@contextmanager
def allow_model_only_resume(checkpoint):
    """
    Scope the upstream optimizer fallback to this worker and this input file.

    Native resume handles full last.pt files unchanged. Numbered snapshots omit
    optimizer state: their explicit marker keeps the newly created optimizers.
    No module globals or training frames are inspected to obtain live objects.
    """
    if checkpoint is None:
        yield
        return
    from blinkmimic.optim import MuonAdam

    marker = object()
    original_load = torch.load
    optimizer_loaders = [
        (cls, cls.load_state_dict) for cls in (torch.optim.Adam, MuonAdam)
    ]

    def load(file, *args, **kwargs):
        saved = original_load(file, *args, **kwargs)
        if (
            isinstance(file, (str, Path))
            and Path(file).resolve() == Path(checkpoint).resolve()
        ):
            keys = {"opt_pi", "opt_wm", "opt_v"}
            if not keys.intersection(saved):
                saved = dict(saved, **{key: marker for key in keys})
                print(
                    "[WARN] Model-only resume: optimizers restart; saved"
                    " models and iteration are restored.",
                    flush=True,
                )
            elif not keys.issubset(saved):
                raise ValueError(
                    "Incomplete optimizer state in BlinkMimic checkpoint"
                )
        return saved

    def loader(original):
        def load_state(optimizer, state):
            if state is not marker:
                return original(optimizer, state)

        return load_state

    torch.load = load
    for cls, original in optimizer_loaders:
        cls.load_state_dict = loader(original)
    try:
        yield
    finally:
        torch.load = original_load
        for cls, original in optimizer_loaders:
            cls.load_state_dict = original
