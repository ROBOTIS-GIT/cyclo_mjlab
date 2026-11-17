# ruff: noqa: E402  # Bootstrap the vendored namespace before source imports.
"""Portable exact-state library tests using synthetic references, no private data."""

import json
import shutil

import numpy as np
import pytest
import torch

from source.tasks.blinkmimic.runtime import prepare

prepare()
from blinkmimic.kinematics import Kinematics

from source.tasks.blinkmimic.motion import (
    ExactMotionLib,
    digest,
    load_exact,
    verify_manifest,
)
from source.tasks.blinkmimic.prepare import prepare_library
from source.tasks.blinkmimic.config.k1_rev1.robot_spec import make_robot


def reference(path, names, frames=4, reverse=False, **changes):
    q = np.tile(np.arange(len(names), dtype=np.float32), (frames, 1)) * 0.001
    data = dict(
        fps=np.array(50),
        joint_names=np.array(names),
        backend=np.array("mujoco"),
        quaternion_order=np.array("wxyz"),
        velocity_frame=np.array("world"),
        velocity_point=np.array("link_origin"),
        root_pos_w=np.zeros((frames, 3)),
        root_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1)),
        root_lin_vel_w=np.tile([0.7, 0.1, -0.2], (frames, 1)),
        root_ang_vel_w=np.tile([0.1, -0.2, 0.3], (frames, 1)),
        joint_pos=q,
        joint_vel=q + 0.2,
    )
    if reverse:
        data.update(
            joint_names=np.array(names[::-1]),
            joint_pos=q[:, ::-1],
            joint_vel=(q + 0.2)[:, ::-1],
        )
    data.update(changes)
    np.savez(path, **data)
    return data


def test_preparation_and_relocation_preserve_order_states_and_boundaries(tmp_path):
    robot = make_robot()
    a, b = tmp_path / "second.npz", tmp_path / "first.npz"
    original = reference(a, robot.joint_names, reverse=True)
    reference(b, robot.joint_names, frames=7)
    library = tmp_path / "library"
    prepare_library([a, b], library, robot.joint_names)
    saved = json.loads((library / "manifest.json").read_text())
    assert [r["song"] for r in saved["clips"]] == ["second", "first"]
    assert saved["clips"][0]["derived"] == "0000_second.npz"
    assert digest(library / "0000_second.npz") == digest(a)
    with pytest.raises(FileExistsError):
        prepare_library([a, b], library, robot.joint_names)
    relocated = tmp_path / "relocated"
    shutil.move(library, relocated)
    a.unlink()
    b.unlink()  # Source files are not needed for the portable derived library.
    motion = ExactMotionLib.from_dir(relocated, robot, Kinematics(robot, "cpu"), "cpu")
    assert motion.names == ["second", "first"] and motion.num_frames == 11
    assert motion.clip_start.tolist() == [0, 4]
    assert motion.clip_end.tolist() == [3, 10]
    np.testing.assert_array_equal(motion.q[:4].numpy(), original["joint_pos"][:, ::-1])
    np.testing.assert_array_equal(
        motion.root_v[:4].numpy(), original["root_lin_vel_w"].astype(np.float32)
    )
    assert motion.future_frames(torch.tensor([2, 3, 9, 10]), 2).tolist() == [
        [3, 3],
        [3, 3],
        [10, 10],
        [10, 10],
    ]
    with pytest.raises(FileNotFoundError):
        verify_manifest(relocated / "manifest.json", verify_sources=True)


@pytest.mark.parametrize("value", [1e100, float("inf"), float("nan"), 1 + 1j])
def test_rejects_values_invalid_in_float32(tmp_path, value):
    names = [f"joint_{i}" for i in range(23)]
    path = tmp_path / "invalid.npz"
    reference(path, names, joint_pos=np.full((4, 23), value))
    with pytest.raises(ValueError, match="Invalid"):
        load_exact(path, names)


def test_empty_manifest_duplicate_identity_and_changed_bytes_rejected(tmp_path):
    names = [f"joint_{i}" for i in range(23)]
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    reference(a, names)
    reference(b, names)
    out = tmp_path / "data"
    path = prepare_library([a, b], out, names)
    data = json.loads(path.read_text())
    path.write_text(json.dumps({"clips": []}))
    with pytest.raises(ValueError, match="nonempty"):
        verify_manifest(path)
    data["clips"][1]["song"] = "a"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Duplicate clip identifier"):
        verify_manifest(path)
    data["clips"][1]["song"] = "b"
    path.write_text(json.dumps(data))
    (out / data["clips"][0]["derived"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_manifest(path)


def test_extra_npz_and_incorrect_frame_count_rejected(tmp_path):
    robot = make_robot()
    p = tmp_path / "a.npz"
    reference(p, robot.joint_names)
    out = tmp_path / "data"
    path = prepare_library([p], out, robot.joint_names)
    extra = out / "extra.npz"
    extra.write_bytes(p.read_bytes())
    with pytest.raises(ValueError, match="exactly match"):
        ExactMotionLib.from_dir(out, robot, Kinematics(robot, "cpu"), "cpu")
    extra.unlink()
    data = json.loads(path.read_text())
    data["clips"][0]["frames"] = 5
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="frame lengths"):
        ExactMotionLib.from_dir(out, robot, Kinematics(robot, "cpu"), "cpu")
