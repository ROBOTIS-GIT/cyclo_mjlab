import numpy as np
import pytest
from cyclo_supermimic.motion import convert_motion, validate_motion


def clip():
    return dict(
        fps=50,
        joint_names=np.array(["b", "a"]),
        joint_pos=np.array([[1, 2], [3, 4]]),
        joint_vel=np.zeros((2, 2)),
        root_pos_w=np.zeros((2, 3)),
        root_quat_w=np.tile([1, 0, 0, 0], (2, 1)),
        root_lin_vel_w=np.zeros((2, 3)),
        root_ang_vel_w=np.zeros((2, 3)),
    )


def test_named_joint_order():
    out = validate_motion(clip(), ("a", "b"))
    np.testing.assert_array_equal(out["joint_pos"], [[2, 1], [4, 3]])


@pytest.mark.parametrize(
    "key,value",
    [
        ("fps", 30),
        ("joint_names", ["a", "a"]),
        ("joint_vel", np.zeros((3, 2))),
        ("root_quat_w", np.zeros((2, 4))),
        ("root_pos_w", np.full((2, 3), np.nan)),
    ],
)
def test_reject_invalid_motion(key, value):
    data = clip()
    data[key] = value
    with pytest.raises(ValueError):
        validate_motion(data, ("a", "b"))


def test_legacy_root_and_source_preservation(tmp_path):
    data = clip()
    for root, body in [
        ("root_pos_w", "body_pos_w"),
        ("root_quat_w", "body_quat_w"),
        ("root_lin_vel_w", "body_lin_vel_w"),
        ("root_ang_vel_w", "body_ang_vel_w"),
    ]:
        data[body] = data.pop(root)[:, None, :]
    data["body_names"] = ["pelvis"]
    src = tmp_path / "legacy.npz"
    dst = tmp_path / "research.npz"
    np.savez(src, **data)
    original = src.read_bytes()
    convert_motion(src, dst, ("a", "b"))
    with np.load(dst) as converted:
        np.testing.assert_array_equal(converted["root_quat_w"], [[1, 0, 0, 0]] * 2)
    assert src.read_bytes() == original
    with pytest.raises(FileExistsError):
        convert_motion(src, dst, ("a", "b"))
    with pytest.raises(ValueError):
        convert_motion(src, src, ("a", "b"))
