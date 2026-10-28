# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""K1 Rev.2 asset, with the supplied Rev.2 geometry and inertial properties."""
from pathlib import Path
import mujoco
from source import SRC_PATH
from .k1_rev1 import get_k1_rev1_cfg, K1_REV1_ACTION_SCALE

K1_REV2_XML_PATH: Path = (SRC_PATH / 'assets/robots/k1_rev2/mujoco/k1_rev2/k1.xml')
# Both revisions specify the same QC060/QC080 motor limits and armatures.
K1_REV2_ACTION_SCALE = dict(K1_REV1_ACTION_SCALE)


def _load_k1_rev2_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(K1_REV2_XML_PATH))
    for actuator in tuple(spec.actuators):
        spec.delete(actuator)
    for side in ('left', 'right'):
        spec.body(f'{side}_ankle_roll_link').add_site(
            name=f'{side}_foot', pos=(0.0325, 0., -0.0635),
            size=(.01, .01, .01), group=5)
    for name, kind in [('imu_ang_vel', mujoco.mjtSensor.mjSENS_GYRO),
                       ('imu_lin_vel', mujoco.mjtSensor.mjSENS_VELOCIMETER),
                       ('imu_lin_acc', mujoco.mjtSensor.mjSENS_ACCELEROMETER)]:
        spec.add_sensor(name=name, type=kind, objtype=mujoco.mjtObj.mjOBJ_SITE,
                        objname='imu')
    spec.add_sensor(name='root_angmom', type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
                    objtype=mujoco.mjtObj.mjOBJ_BODY, objname='pelvis')
    return spec


def get_k1_rev2_cfg():
    """Reuse verified motor/PD/contact settings, load only the Rev.2 model."""
    cfg = get_k1_rev1_cfg()
    cfg.spec_fn = _load_k1_rev2_spec
    return cfg
