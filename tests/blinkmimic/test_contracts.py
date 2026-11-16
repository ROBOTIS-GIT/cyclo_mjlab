# ruff: noqa: E402  # Vendored path bootstrap must precede upstream imports.
"""Recipe, exact-input and numerical K1 adaptation regression checks."""

import json

import mujoco
import numpy as np
import pytest
import torch

from source.tasks.blinkmimic.runtime import VENDOR, prepare

prepare()
from blinkmimic.dr import XiLayout
from blinkmimic.kinematics import Kinematics
from blinkmimic.mirror import Mirror
from blinkmimic.rotations import quat_to_mat

from source.tasks.blinkmimic.cli import execution_recipe, recipe
from source.tasks.blinkmimic.motion import digest, load_exact
from source.tasks.blinkmimic.robot import make_robot


def test_recipe_and_vendor_identity():
    flags = recipe()
    for flag, value in {
        "--pi_batch": "2048",
        "--wm_hidden": "1024,1024,1024",
        "--pi_lr_mult": "4",
        "--wm_only_iters": "0",
    }.items():
        assert flags[flags.index(flag) + 1] == value
    upstream = json.loads((VENDOR / "UPSTREAM.json").read_text())
    for name, sha in upstream["original_sha256"].items():
        if name not in upstream["modifications"]:
            assert digest(VENDOR / name) == sha
    assert "--enc_hidden" not in flags  # upstream full-size default
    assert "--dec_hidden" not in flags


def test_systems_memory_alternative_preserves_all_learning_options():
    official, removed = execution_recipe("official")
    assert official == recipe() and not removed
    alternative, removed = execution_recipe("compiled_no_torch_graphs")
    restore = iter(official)
    expected = []
    for flag in restore:
        if flag == "--eprobe_async":
            assert next(restore) == "20"
        elif flag not in {
            "--cuda_graph",
            "--graph_early",
            "--collect_graph",
            "--reset_masked",
            "--eprobe_graph",
        }:
            expected.append(flag)
    assert alternative == expected
    assert "--compile_more" in alternative
    assert "--pi_batch" in alternative and "--wm_hidden" in alternative
    assert "--pi_opt" in alternative and "--wm_opt" in alternative


def test_nominal_pose_does_not_modify_shared_asset():
    from source.assets.robots.robotis_k1.k1_rev1 import get_k1_rev1_cfg

    r = make_robot()
    assert r.default_joint_pos[".*_elbow_joint"] == 0.6
    assert get_k1_rev1_cfg().init_state.joint_pos[".*_elbow_joint"] == 0.95
    assert len(r.joint_names) == 23


def test_fk_and_twists_match_mujoco():
    r = make_robot()
    kin = Kinematics(r, "cpu")
    m = mujoco.MjModel.from_xml_path(r.mjcf_path)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(123)
    for _ in range(4):
        d.qpos[:] = 0
        d.qpos[:3] = rng.normal(size=3)
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        d.qpos[3:7] = quat
        d.qpos[7:] = rng.uniform(-0.3, 0.3, 23)
        d.qvel[:] = rng.normal(size=29)
        mujoco.mj_forward(m, d)
        R = quat_to_mat(torch.tensor(quat, dtype=torch.float32)[None])

        def tensor(a):
            return torch.tensor(a, dtype=torch.float32)[None]

        w = (R @ tensor(d.qvel[3:6])[..., None]).squeeze(-1)
        out = kin.forward(
            tensor(d.qpos[:3]),
            R,
            tensor(d.qvel[:3]),
            w,
            tensor(d.qpos[7:]),
            tensor(d.qvel[6:]),
        )
        for i, name in enumerate(r.tracked_bodies):
            bid = m.body(name).id
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, vel, 0)
            # MuJoCo body velocity is at COM; transform to body origin.
            origin_v = vel[3:] - np.cross(vel[:3], d.xipos[bid] - d.xpos[bid])
            for actual, expected in zip(
                (out[0][0, i], out[1][0, i], out[2][0, i], out[3][0, i]),
                (d.xpos[bid], d.xmat[bid].reshape(3, 3), origin_v, vel[:3]),
                strict=True,
            ):
                np.testing.assert_allclose(actual.numpy(), expected, atol=2e-6)


def test_mirror_is_physical_and_involutive():
    r = make_robot()
    kin = Kinematics(r, "cpu")
    mirror = Mirror(r, XiLayout(r, "cpu").slices, "cpu")
    torch.manual_seed(5)
    S = (
        torch.randn(3, 3),
        torch.eye(3).expand(3, 3, 3),
        torch.randn(3, 3),
        torch.randn(3, 3),
        torch.randn(3, 23) * 0.2,
        torch.randn(3, 23),
    )
    for a, b in zip(S, mirror.state(mirror.state(S)), strict=True):
        torch.testing.assert_close(a, b)
    bp, bR, bv, bw = kin.forward(*S)
    mp, mR, mv, mw = kin.forward(*mirror.state(S))
    names = r.tracked_bodies
    perm = [
        names.index(
            n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")
        )
        for n in names
    ]
    torch.testing.assert_close(mp, bp[:, perm] @ mirror.M, atol=2e-6, rtol=0)
    torch.testing.assert_close(mv, bv[:, perm] @ mirror.M, atol=2e-6, rtol=0)
    torch.testing.assert_close(mw, -(bw[:, perm] @ mirror.M), atol=2e-6, rtol=0)
    torch.testing.assert_close(mR, mirror.M @ bR[:, perm] @ mirror.M, atol=2e-6, rtol=0)


def test_invalid_motion_rejected(tmp_path):
    r = make_robot()
    p = tmp_path / "bad.npz"
    np.savez(p, backend="isaac")
    with pytest.raises(ValueError, match="backend"):
        load_exact(p, r.joint_names)


def test_task_launcher_keeps_training_and_evaluation_isolated(tmp_path, monkeypatch):
    from source.tasks.blinkmimic import cli

    commands = []

    class Child:
        pid = 123456

        def __init__(self, command, **kwargs):
            commands.append(command)
            assert kwargs["env"]["WANDB_MODE"] == "disabled"

        def wait(self):
            return 0

    monkeypatch.setattr(cli.subprocess, "check_output", lambda *a, **k: "test-source")
    monkeypatch.setattr(cli.subprocess, "Popen", Child)
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text("{}")
    train = tmp_path / "train"
    with pytest.raises(SystemExit) as result:
        cli.main(
            "train",
            [
                cli.TASK,
                "--motion-dir",
                str(data),
                "--run-dir",
                str(train),
                "--num_envs",
                "2048",
            ],
        )
    assert result.value.code == 0
    assert commands[-1][4] == "train"
    assert commands[-1][commands[-1].index("--num_envs") + 1] == "2048"
    assert "--enc_hidden" not in commands[-1] and "--dec_hidden" not in commands[-1]
    ckpt = train / "ckpt_000100.pt"
    ckpt.write_bytes(b"test-only; subprocess mocked")
    evaluation = tmp_path / "eval"
    with pytest.raises(SystemExit) as result:
        cli.main(
            "play",
            [
                cli.TASK,
                "--motion-dir",
                str(data),
                "--run-dir",
                str(evaluation),
                "--checkpoint",
                str(ckpt),
            ],
        )
    assert result.value.code == 0
    assert commands[-1][4] == "evaluate"
    assert "--ckpt" not in commands[-1]  # directory mode writes per-clip JSON
    assert (evaluation / ckpt.name).resolve() == ckpt
    assert ckpt.read_bytes() == b"test-only; subprocess mocked"
