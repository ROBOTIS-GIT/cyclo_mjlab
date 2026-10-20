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


TAEGUEK_TRAJECTORIES = {
  chapter: SRC_PATH / "assets" / "motions" / "K1_rev1" / f"taeguek_{chapter:02d}"
  / f"k1_taeguek{chapter:02d}_collision_only_50hz_converted.npz"
  for chapter in range(1, 9)
}
# User-selected arm de-jitter/collision correction V1; unchanged 50 Hz timing.
TAEGUEK_TRAJECTORIES[7] = (
  SRC_PATH / "assets/motions/K1_rev1/taeguek_07"
  / "k1_taeguek07_arm_corrected_v1_50hz_converted.npz"
)


MOTIONS = {
  "geumgang": MotionCurriculum("geumgang", "Cyclo-Mimic-K1-Rev1-Geumgang",
    SRC_PATH / "assets/motions/K1_rev1/geumgang/tkd_geumgang_converted.npz"),
  "koryo": MotionCurriculum("koryo", "Cyclo-Mimic-K1-Rev1-Koryo",
    SRC_PATH / "assets/motions/K1_rev1/koryo/k1_koryo_converted.npz"),
  "taeguek01": MotionCurriculum("taeguek01", "Cyclo-Mimic-K1-Rev1-Taeguek01",
    TAEGUEK_TRAJECTORIES[1]),
  "taeguek02": MotionCurriculum("taeguek02", "Cyclo-Mimic-K1-Rev1-Taeguek02",
    TAEGUEK_TRAJECTORIES[2]),
}

# Versioned identities preserve the original Taegeuk 1/2 tasks and checkpoints.
TAEGUEK_COLLISION_ONLY_NAMES = tuple(f'taeguek{i:02d}_collision_only' for i in range(1, 9))
for _chapter, _name in enumerate(TAEGUEK_COLLISION_ONLY_NAMES, start=1):
  MOTIONS[_name] = MotionCurriculum(
    _name, f'Cyclo-Mimic-K1-Rev1-Taeguek{_chapter:02d}-CollisionOnly',
    SRC_PATH / f'assets/motions/K1_rev1/taeguek_{_chapter:02d}/k1_taeguek{_chapter:02d}_collision_only_laptop_50hz.npz',
  )


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
