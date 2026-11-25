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
from unittest.mock import patch

import torch


def check_resume_models(saved):
    if "pi" in saved and "wm" not in saved:
        raise ValueError(
            "Cannot resume from a policy-only (--ckpt_light) checkpoint: "
            "world model, critic and optimizer states are absent. "
            "Use the full last.pt saved at the end of training. "
            "Light checkpoints can still be used for play and export."
        )


def read_resume_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    check_resume_models(saved)
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


def training_arguments(saved):
    settings = saved["args"]
    ignored = {"resume", "log_dir", "motion_dir", "num_envs", "robot", "ckpt_light"}
    arguments = []
    for name, value in settings.items():
        if name in ignored or value is None or value is False:
            continue
        arguments.append("--" + name)
        if value is not True:
            arguments.append(str(value))
    return arguments


def policy_state_for_robot(state, robot, *, prefix=""):
    """Use the target's reference-action transform, retaining the learned correction."""
    result = state
    for name, values in (("ref_q0", robot.default_joint_pos_list()), ("ref_alpha", robot.action_scale_list())):
        key = prefix + name
        buffer = state.get(key)
        if not isinstance(buffer, torch.Tensor) or buffer.shape != (len(values),):
            raise ValueError(f"Checkpoint {key} shape differs from the target robot")
        target = buffer.new_tensor(values)
        if not torch.equal(buffer, target):
            if result is state:
                result = state.copy()
            result[key] = target
    return result


def prepare_resume_state(saved, target_clips, target_robot, history):
    """Adapt reference buffers and sampler state before native model loading."""
    saved = saved.copy()
    if target_robot is not None and saved.get("args", {}).get("pi_ref_skip"):
        for key, prefix in (("pi", ""), ("pi_live", ""), ("pi_avg", "module.")):
            if saved.get(key) is not None:
                saved[key] = policy_state_for_robot(saved[key], target_robot, prefix=prefix)
    for clip, bins in saved.pop("inactive_sampler", {}).items():
        history[clip] = {key: value.cpu() for key, value in bins.items()}
    control = saved.get("ctrl")
    if control is not None and target_clips is not None:
        from .sampler import transfer_sampler

        saved["ctrl"] = transfer_sampler(
            control, control.get("clips"), target_clips, history,
        )
    return saved


@contextmanager
def native_training_resume(
    checkpoint, target_clips=None, *, target_robot=None
):
    """
    Connect checkpoint adaptation to the native loader in this worker only.

    Native resume restores the learned parameters. Numbered snapshots omit
    optimizer state: their explicit marker keeps the newly created optimizers.
    The load hook applies only to the selected checkpoint; all hooks are restored
    on exit. No training frames are inspected to obtain live objects.
    """
    history = {}
    if checkpoint is None:
        yield history
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
            check_resume_models(saved)
            saved = prepare_resume_state(saved, target_clips, target_robot, history)
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
        yield history
    finally:
        torch.load = original_load
        for cls, original in optimizer_loaders:
            cls.load_state_dict = original


@contextmanager
def initialize_world_model(checkpoint):
    """Seed the native world model before optimizer construction."""
    if checkpoint is None:
        yield
        return
    from blinkmimic import networks

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    settings = saved.get("args", {})
    if "wm" not in saved:
        raise ValueError(
            "Expected a BlinkMimic world model for the target robot"
        )
    weights = saved["wm"]
    del saved
    native_init = networks.WorldModel.__init__

    def initialize(model, *args, **kwargs):
        native_init(model, *args, **kwargs)
        if model.structured != settings.get("wm_structured", False):
            raise ValueError(
                "World model representation differs from the checkpoint"
            )
        model.load_state_dict(weights, strict=True)
        print(f"[INFO] Initialized world model from {checkpoint}", flush=True)

    with patch.object(networks.WorldModel, "__init__", initialize):
        yield
