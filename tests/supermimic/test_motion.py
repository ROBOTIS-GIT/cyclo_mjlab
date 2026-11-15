import numpy as np
import pytest
from cyclo_supermimic.motion import ROOT_CONTRACT, convert_motion, validate_motion


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
        ("fps", "50"),
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


def test_explicit_root_contract_and_source_preservation(tmp_path):
    data = clip()
    data.update(ROOT_CONTRACT)
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


def test_finite_float64_cannot_overflow_loader_float32():
    data = clip()
    data["joint_vel"] = np.full((2, 2), 1e40)
    with pytest.raises(ValueError, match="joint_vel.*finite float32"):
        validate_motion(data, ("a", "b"))
