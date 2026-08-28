#!/usr/bin/env python3
"""Visualize robot meshes and collision geoms with a consistent color scheme."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

# Running a file under scripts/ puts scripts/ (not the repository root) on
# sys.path. Prefer this checkout's source package over any installed copy.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import mujoco.viewer

from mjlab.entity import Entity, EntityCfg
from source.assets.robots.robotis_k1.k1_rev1 import get_k1_rev1_cfg


ROBOT_BUILDERS: dict[str, Callable[[], EntityCfg]] = {
  "k1": get_k1_rev1_cfg,
}


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("robot", choices=ROBOT_BUILDERS)
  parser.add_argument(
    "--mode",
    choices=("both", "visual", "collision"),
    default="both",
    help="Geometry groups to show (default: both).",
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  entity = Entity(ROBOT_BUILDERS[args.robot]())
  model = entity.spec.compile()
  data = mujoco.MjData(model)
  if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)

  # Color collision group 3 red in this viewer only. Asset colors are unchanged.
  for geom_id in range(model.ngeom):
    if model.geom_group[geom_id] == 3:
      model.geom_rgba[geom_id] = (1.0, 0.15, 0.1, 0.55)

  with mujoco.viewer.launch_passive(model, data) as viewer:
    if args.mode == "visual":
      viewer.opt.geomgroup[2] = 1
      viewer.opt.geomgroup[3] = 0
    elif args.mode == "collision":
      viewer.opt.geomgroup[:] = 0
      viewer.opt.geomgroup[3] = 1
      for geom_id in range(model.ngeom):
        if model.geom_group[geom_id] == 3:
          model.geom_rgba[geom_id, 3] = 1.0
    else:
      viewer.opt.geomgroup[2] = 1
      viewer.opt.geomgroup[3] = 1

    while viewer.is_running():
      viewer.sync()
      time.sleep(0.01)


if __name__ == "__main__":
  main()
