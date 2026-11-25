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
Local model and optimizer snapshots; simulation and replay state are not saved.
"""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

from source.utils.checkpoint_retention import prune_model_checkpoints


@dataclass
class ModelCheckpoint:
    source: Path
    manifest: dict
    actor: dict
    world_model: dict
    critic: dict | None = None
    step: int | None = None
    optimizers: dict | None = None
    sampler: dict | None = None

    @property
    def export_directory(self):
        if self.source.is_dir():
            return self.source / "exported"
        parent = self.source.parent
        if parent.name == "checkpoints":
            parent = parent.parent
        return parent / "exported" / self.source.stem


def load_checkpoint(source):
    """
    Prefer last.pt in a run directory; also read split bundles and snapshots.
    """
    if isinstance(source, ModelCheckpoint):
        return source
    source = Path(source).expanduser()
    if source.is_dir() and (source / "last.pt").is_file():
        source = source / "last.pt"
    if source.is_dir():
        return _load_split_checkpoint(source)
    return _archive_checkpoint(source, torch.load(source, map_location="cpu", weights_only=True))


def _load_split_checkpoint(source):
    for name in ("manifest.json", "actor.pt", "world_model.pt"):
        if not (source / name).is_file():
            raise ValueError(f"Incomplete checkpoint bundle: missing {name}")
    return ModelCheckpoint(
        source=source,
        manifest=json.loads((source / "manifest.json").read_text()),
        actor=torch.load(
            source / "actor.pt", map_location="cpu", weights_only=True
        ),
        world_model=torch.load(
            source / "world_model.pt", map_location="cpu", weights_only=True
        ),
    )


def _archive_checkpoint(source, saved):
    if (
        not isinstance(saved, dict)
        or saved.get("format") != "cyclo.supermimic.checkpoint.v1"
    ):
        raise ValueError("Expected a combined SuperMimic checkpoint")
    for key in ("manifest", "actor", "world_model"):
        if not isinstance(saved.get(key), dict):
            raise ValueError(f"Invalid checkpoint: missing {key} dictionary")
    step = saved.get("step")
    if type(step) is not int or step < 0:
        raise ValueError("Invalid checkpoint step")
    if saved.get("critic") is not None and not isinstance(
        saved["critic"], dict
    ):
        raise ValueError("Invalid checkpoint critic")
    if saved.get("optimizers") is not None and not isinstance(
        saved["optimizers"], dict
    ):
        raise ValueError("Invalid checkpoint optimizers")
    return ModelCheckpoint(
        source=source,
        manifest=saved["manifest"],
        actor=saved["actor"],
        world_model=saved["world_model"],
        critic=saved.get("critic"),
        step=step,
        optimizers=saved.get("optimizers"),
        sampler=saved.get("sampler"),
    )


def load_world_model(path):
    """Read world-model weights without depending on manifest or policy metadata."""
    if path.is_dir():
        path = path / ("last.pt" if (path / "last.pt").is_file() else "world_model.pt")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(saved, dict) and saved.get("format") == "cyclo.supermimic.checkpoint.v1":
        weights = saved.get("world_model")
    else:
        weights = saved

    if (
        not isinstance(weights, dict)
        or not all(isinstance(weights.get(key), dict) for key in ("dynamics", "sim_context"))
    ):
        raise ValueError("Expected a SuperMimic world-model bundle")
    return weights


def check_motion_contract(old, new):
    if old["anchor_body_name"] != new["anchor_body_name"]:
        raise ValueError("Checkpoint motion anchor_body_name differs from the target")
    if old["body_names"] != new["body_names"]:
        raise ValueError("Checkpoint motion body_names differs from the target")


def check_training_contract(source, target):
    """Check representation and motion semantics for full training resume."""
    if source.get("schema_version") not in (1, 2):
        raise ValueError("Unsupported checkpoint manifest version")
    if source["representation"] != target["representation"]:
        raise ValueError("Checkpoint representation differs from the target")

    old, new = source["environment"], target["environment"]
    if "commands" not in old:
        raise ValueError("Missing checkpoint motion configuration")
    check_motion_contract(old["commands"]["motion"], new["commands"]["motion"])


def initialize_world_model(model, weights):
    """Restore learned weights and normalization with the current action mapping."""
    state = dict(weights)
    state["dynamics"] = dict(weights["dynamics"])
    # Match the simulator's commands, while preserving the network's learned input basis.
    for name in ("action_scale", "action_offset"):
        target = getattr(model.dynamics, name)
        if state["dynamics"][name].shape != target.shape:
            raise ValueError(f"World model dynamics.{name} shape differs from the target")
        state["dynamics"][name] = target
    model.load_bundle(state)


def save_checkpoint(
    directory,
    *,
    step,
    manifest,
    actor,
    world_model,
    critic=None,
    optimizers=None,
    sampler=None,
):
    """Save model history and latest optimizer state, then apply retention."""
    directory = Path(directory)
    target = directory / f"model_{step}.pt"
    saved = dict(
        format="cyclo.supermimic.checkpoint.v1",
        step=step,
        manifest=manifest,
        actor=actor,
        world_model=world_model,
        critic=critic,
    )
    if sampler is not None:
        saved["sampler"] = sampler
    _save_atomic(target, saved)
    _save_atomic(directory / "last.pt", dict(saved, optimizers=optimizers))
    prune_model_checkpoints(target.parent)
    return target


def _save_atomic(target, saved):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            torch.save(saved, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
