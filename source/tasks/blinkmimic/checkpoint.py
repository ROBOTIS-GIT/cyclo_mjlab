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

"""Checkpoint naming, selection, and retention for the BlinkMimic adapter."""

import re

from source.utils.checkpoint_retention import prune_model_checkpoints


def checkpoint_recipe(flags):
    """Use full latest checkpoints and iteration-based historical snapshots."""
    result = []
    flags = iter(flags)
    for flag in flags:
        if flag in ("--ckpt_every", "--ckpt_minutes"):
            next(flags)
        elif flag != "--ckpt_light":
            result.append(flag)
    return [*result, "--ckpt_every", "1000", "--ckpt_minutes", "0"]


def publish_completed_snapshots(run_dir):
    """
    Normalize only atomically published upstream snapshots, never partial
    writes.
    """
    published = []
    for path in run_dir.glob("ckpt_*.pt"):
        match = re.fullmatch(r"ckpt_([0-9]+)\.pt", path.name)
        if match and path.is_file():
            snapshot = run_dir / f"model_{int(match[1])}.pt"
            path.rename(snapshot)
            published.append(snapshot)
    prune_model_checkpoints(run_dir)
    return sorted(
        (p for p in published if p.exists()), key=lambda p: int(p.stem[6:])
    )


def select_checkpoint(checkpoint, checkpoint_dir):
    if checkpoint is None:
        if checkpoint_dir is None or not checkpoint_dir.is_dir():
            raise ValueError(
                "play requires --checkpoint or an existing --checkpoint-dir"
            )
        candidates = [
            path
            for path in checkpoint_dir.iterdir()
            if path.is_file()
            and re.fullmatch(r"(?:model_|ckpt_)[0-9]+\.pt", path.name)
        ]
        if not candidates:
            raise ValueError(
                f"No model_<iteration>.pt checkpoints in {checkpoint_dir}"
            )
        checkpoint = max(
            candidates, key=lambda path: int(path.stem.split("_", 1)[1])
        )
    if not checkpoint.is_file() or not re.fullmatch(
        r"(?:model_|ckpt_)[0-9]+\.pt", checkpoint.name
    ):
        raise ValueError(
            "Expected an existing model_<iteration>.pt checkpoint:"
            f" {checkpoint}"
        )
    return checkpoint.resolve()


def link_evaluation_checkpoint(checkpoint, directory):
    """The upstream directory evaluator discovers only ckpt_*.pt filenames."""
    iteration = int(checkpoint.stem.split("_", 1)[1])
    (directory / f"ckpt_{iteration:06d}.pt").symlink_to(checkpoint.resolve())
