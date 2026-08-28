#!/usr/bin/env python3
"""Convert the vendored ROBOTIS K1 Rev.1 URDF to an MJLab-ready MJCF."""

from __future__ import annotations

import shutil
from pathlib import Path

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DESCRIPTION = (
  PROJECT_ROOT / "third_party" / "ai_sapiens" / "ai_sapiens_description"
)
SOURCE_URDF = SOURCE_DESCRIPTION / "urdf" / "k1_rev1" / "k1.urdf"
SOURCE_MESH_DIR = SOURCE_DESCRIPTION / "meshes" / "k1_rev1"

TARGET_DIR = PROJECT_ROOT / "source" / "assets" / "robots" / "robotis_k1"
TARGET_XML = TARGET_DIR / "xmls" / "k1.xml"
TARGET_MESH_DIR = TARGET_XML.parent / "assets"

FOOT_CAPSULE_RADIUS = 0.01
FOOT_CAPSULE_Y_POSITIONS = (-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03)
# Longitudinal center-line endpoints fitted to the K1 outsole contour at the
# capsule center height. The shorter outer rows follow the rounded corners.
FOOT_CAPSULE_X_RANGES = (
  (0.0154, 0.0827),
  (-0.0414, 0.1105),
  (-0.0494, 0.1146),
  (-0.0486, 0.1142),
  (-0.0494, 0.1146),
  (-0.0414, 0.1105),
  (0.0154, 0.0827),
)
FOOT_CAPSULE_Z = -0.0535
FOOT_SITE_X = 0.0325


def _load_urdf() -> mujoco.MjSpec:
  if not SOURCE_URDF.is_file():
    raise FileNotFoundError(
      f"K1 URDF not found: {SOURCE_URDF}. Initialize the ai_sapiens submodule first."
    )

  urdf = SOURCE_URDF.read_text()
  package_uri = "package://ai_sapiens_description/meshes/k1_rev1"
  urdf = urdf.replace(package_uri, str(SOURCE_MESH_DIR.resolve()))
  urdf = urdf.replace(
    '<compiler discardvisual="false"',
    '<compiler fusestatic="false" discardvisual="false"',
  )
  return mujoco.MjSpec.from_string(urdf)


def _prepare_spec(spec: mujoco.MjSpec) -> None:
  pelvis = spec.body("pelvis")
  if pelvis is None:
    raise ValueError("The K1 URDF does not contain the expected root body 'pelvis'.")
  if len(pelvis.joints) != 0:
    raise ValueError("Expected the imported K1 pelvis to have no joint.")
  pelvis.add_freejoint(name="floating_base_joint")

  # The actuator model and effort limits live in k1_rev1.py. Remove the
  # URDF-derived joint-side actuator force ranges to avoid two independent clamps.
  for joint in spec.joints:
    joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_FALSE
    joint.actfrcrange[:] = (0.0, 0.0)
    joint.armature = 0.0

  # URDF collision elements do not have names. MJLab collision configuration and
  # contact rewards address geoms by name, so assign deterministic names here.
  for body in spec.bodies:
    collision_index = 0
    for geom in body.geoms:
      is_collision = geom.contype != 0 or geom.conaffinity != 0
      if not is_collision:
        geom.group = 2
        continue

      if body.name == "left_ankle_roll_link":
        base_name = "left_foot_collision"
      elif body.name == "right_ankle_roll_link":
        base_name = "right_foot_collision"
      else:
        base_name = f"{body.name}_collision"

      geom.name = (
        base_name if collision_index == 0 else f"{base_name}_{collision_index + 1}"
      )
      geom.group = 3
      collision_index += 1


def _replace_foot_collisions(spec: mujoco.MjSpec) -> None:
  """Replace each set of URDF foot collisions with seven longitudinal capsules."""
  for side in ("left", "right"):
    foot_body = spec.body(f"{side}_ankle_roll_link")
    if foot_body is None:
      raise ValueError(f"Missing expected K1 {side} ankle roll body.")

    collision_geoms = [
      geom
      for geom in foot_body.geoms
      if geom.contype != 0 or geom.conaffinity != 0
    ]
    if not collision_geoms:
      raise ValueError(f"Expected at least one {side} foot collision geom.")
    for geom in collision_geoms:
      spec.delete(geom)

    for index, (y_pos, x_range) in enumerate(
      zip(FOOT_CAPSULE_Y_POSITIONS, FOOT_CAPSULE_X_RANGES, strict=True),
      start=1,
    ):
      foot_body.add_geom(
        name=f"{side}_foot{index}_collision",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=(FOOT_CAPSULE_RADIUS, 0.0, 0.0),
        fromto=(
          x_range[0],
          y_pos,
          FOOT_CAPSULE_Z,
          x_range[1],
          y_pos,
          FOOT_CAPSULE_Z,
        ),
        group=3,
        condim=3,
        priority=1,
        friction=(0.6, 0.005, 0.0001),
      )

    foot_body.add_site(
      name=f"{side}_foot",
      pos=(
        FOOT_SITE_X,
        0.0,
        FOOT_CAPSULE_Z - FOOT_CAPSULE_RADIUS,
      ),
      size=(0.01, 0.005, 0.005),
      rgba=(1.0, 0.0, 0.0, 1.0),
      group=5,
    )


def _add_sensors(spec: mujoco.MjSpec) -> None:
  """Add the built-in sensors required by MJLab velocity tasks."""
  pelvis = spec.body("pelvis")
  if pelvis is None:
    raise ValueError("Missing expected K1 pelvis body.")

  # The source description does not provide an IMU extrinsic. Use the pelvis
  # frame origin until a measured extrinsic exists.
  pelvis.add_site(
    name="imu",
    pos=(0.0, 0.0, 0.0),
    size=(0.01, 0.005, 0.005),
    group=5,
  )
  for name, sensor_type in (
    ("imu_ang_vel", mujoco.mjtSensor.mjSENS_GYRO),
    ("imu_lin_vel", mujoco.mjtSensor.mjSENS_VELOCIMETER),
    ("imu_lin_acc", mujoco.mjtSensor.mjSENS_ACCELEROMETER),
  ):
    spec.add_sensor(
      name=name,
      type=sensor_type,
      objtype=mujoco.mjtObj.mjOBJ_SITE,
      objname="imu",
    )
  spec.add_sensor(
    name="root_angmom",
    type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    objname="pelvis",
  )


def _write_assets(spec: mujoco.MjSpec) -> None:
  TARGET_MESH_DIR.mkdir(parents=True, exist_ok=True)
  for mesh in SOURCE_MESH_DIR.glob("*.stl"):
    shutil.copy2(mesh, TARGET_MESH_DIR / mesh.name)

  # Keep the generated MJCF relocatable. Mesh paths are resolved from the XML.
  spec.modelfiledir = str(TARGET_XML.parent)
  spec.meshdir = "assets"
  xml = spec.to_xml()
  lines = xml.splitlines()
  lines.insert(
    1,
    (
      "  <!-- Generated from third_party/ai_sapiens/"
      "ai_sapiens_description/urdf/k1_rev1/k1.urdf. -->"
    ),
  )
  lines.insert(2, "  <!-- Regenerate with scripts/convert_k1_urdf_to_mjcf.py. -->")
  TARGET_XML.parent.mkdir(parents=True, exist_ok=True)
  TARGET_XML.write_text("\n".join(lines) + "\n")

  shutil.copy2(
    PROJECT_ROOT / "third_party" / "ai_sapiens" / "LICENSE",
    TARGET_DIR / "LICENSE.ai_sapiens",
  )


def main() -> None:
  spec = _load_urdf()
  _prepare_spec(spec)
  _replace_foot_collisions(spec)
  _add_sensors(spec)
  spec.compile()
  _write_assets(spec)
  print(f"Wrote {TARGET_XML}")
  print(f"Copied meshes to {TARGET_MESH_DIR}")


if __name__ == "__main__":
  main()
