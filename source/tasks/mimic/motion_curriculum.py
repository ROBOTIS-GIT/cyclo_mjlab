# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Motion identities for the shared taekwondo curriculum and evaluation driver."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from source import SRC_PATH


@dataclass(frozen=True)
class MotionCurriculum:
  name: str
  task: str
  trajectory: Path

  @property
  def training_task(self):
    return self.task + "-Curriculum"

  @property
  def state_key(self):
    return self.name + "_curriculum"

  @property
  def sampler_key(self):
    return self.name + "_start_sampler"

  @property
  def best_key(self):
    return self.name + "_best_full"


MOTIONS = {
  "koryo": MotionCurriculum("koryo", "Cyclo-Mimic-K1-Rev1-Koryo",
    SRC_PATH / "assets/motions/K1_rev1/koryo/k1_koryo_converted.npz"),
  "taeguek01": MotionCurriculum("taeguek01", "Cyclo-Mimic-K1-Rev1-Taeguek01",
    SRC_PATH / "assets/motions/K1_rev1/taeguek_01/k1_taeguek01_converted.npz"),
  "taeguek02": MotionCurriculum("taeguek02", "Cyclo-Mimic-K1-Rev1-Taeguek02",
    SRC_PATH / "assets/motions/K1_rev1/taeguek_02/k1_taeguek02_converted.npz"),
}


def motion_identity(name, trajectory=None):
  path = Path(trajectory) if trajectory is not None else MOTIONS[name].trajectory
  with np.load(path) as motion:
    fps = float(motion["fps"].item())
    frames = len(motion["joint_pos"])
  if not np.isfinite(fps) or fps <= 0 or frames < 2:
    raise ValueError(f"Invalid motion length or FPS: {path}")
  return {"name": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
          "frames": frames, "fps": fps, "duration_s": (frames - 1) / fps}


def validate_checkpoint_motion(infos, name, expected=None):
  """Keep old Koryo checkpoints usable; never import their state into Taeguek."""
  infos = infos or {}
  identity = infos.get("motion_identity")
  if identity is not None:
    if identity != (expected or motion_identity(name)):
      raise ValueError("Checkpoint motion differs from the selected task/NPZ. "
                       "Omit --checkpoint to start a new motion from scratch.")
  elif name != "koryo":
    raise ValueError("This checkpoint has no matching motion identity. "
                     f"Start {name} without --checkpoint; use its resume.pt for later resumes.")
  for other_name, spec in MOTIONS.items():
    if other_name != name and spec.state_key in infos:
      raise ValueError(f"Cannot resume {other_name} curriculum as {name}.")
