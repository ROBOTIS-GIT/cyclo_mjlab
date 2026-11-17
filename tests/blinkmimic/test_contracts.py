# ruff: noqa: E402  # Vendored path bootstrap must precede upstream imports.
"""Recipe, exact-input and numerical K1 adaptation regression checks."""

import datetime
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
from blinkmimic.networks import Policy, RunningNorm, policy_kwargs
from blinkmimic.rotations import quat_to_mat

from source.tasks.blinkmimic.cli import execution_recipe, recipe
from source.tasks.blinkmimic.motion import digest, load_exact
from source.tasks.blinkmimic.config.k1_rev1.robot_spec import make_robot


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


def test_normalization_and_raw_goal_reference_skip_contract():
    norm = RunningNorm(2)
    norm.mean.copy_(torch.tensor([2.0, -3.0]))
    norm.var.copy_(torch.tensor([0.0, 4.0]))
    x = torch.tensor([[2.01, 1.0]])
    torch.testing.assert_close(norm(x), (x - norm.mean) / torch.sqrt(norm.var + 0.0001))
    robot = make_robot()
    policy = Policy(81, 610, 23, **policy_kwargs({"pi_ref_skip": True}, robot))
    with torch.no_grad():
        policy.dec[-1].weight.zero_()
        policy.dec[-1].bias.fill_(0.25)
        policy.goal_norm.mean.fill_(10.0)
        policy.goal_norm.var.fill_(0.25)
    goal = torch.zeros(1, 610)
    goal[:, 15:38] = 0.1
    with torch.no_grad():
        action = policy(torch.zeros(1, 10, 81), goal)
    expected = 0.25 + (goal[:, 15:38] - policy.ref_q0) / policy.ref_alpha
    torch.testing.assert_close(action, expected)
    target = policy.ref_q0 + policy.ref_alpha * action
    torch.testing.assert_close(target, goal[:, 15:38] + policy.ref_alpha * 0.25)


def test_task_launcher_keeps_training_and_evaluation_isolated(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from source.tasks.blinkmimic import cli

    from dataclasses import replace

    selected_robot = SimpleNamespace(name="task_selected_robot")
    monkeypatch.setitem(cli.TASKS, cli.TASK, replace(
        cli.TASKS[cli.TASK], robot_factory=lambda: selected_robot,
    ))
    commands = []
    config_exports = []

    def export_config(child, run_dir, robot):
        assert robot is selected_robot
        config_exports.append(run_dir)
        return run_dir / "params/sim2real.yaml"

    class Child:
        pid = 123456
        returncode = 0

        def __init__(self, command, **kwargs):
            commands.append(command)
            assert kwargs["env"]["WANDB_MODE"] == "disabled"

        def poll(self):
            return 0

        def wait(self):
            return 0

    monkeypatch.setattr(cli.subprocess, "check_output", lambda *a, **k: "test-source")
    monkeypatch.setattr(cli.subprocess, "Popen", Child)
    monkeypatch.setattr(cli, "export_training_config_when_ready", export_config)
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text("{}")
    train = tmp_path / "logs" / cli.TASK / "train"
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
    assert commands[-1][commands[-1].index("--task") + 1] == cli.TASK
    assert commands[-1][commands[-1].index("--robot") + 1] == selected_robot.name
    assert config_exports == [train]
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
                "--viewer",
                "headless",
            ],
        )
    assert result.value.code == 0
    assert commands[-1][4] == "evaluate"
    assert commands[-1][commands[-1].index("--robot") + 1] == selected_robot.name
    assert config_exports == [train]
    assert "--ckpt" not in commands[-1]  # directory mode writes per-clip JSON
    assert (evaluation / ckpt.name).resolve() == ckpt
    assert ckpt.read_bytes() == b"test-only; subprocess mocked"

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    def run_gui(command, **kwargs):
        commands.append(command)
        assert "stdout" not in kwargs and "stderr" not in kwargs
        assert "XDG_CACHE_HOME" not in kwargs["env"]
        return SimpleNamespace(returncode=0)

    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(cli.subprocess, "run", run_gui)
    record = (train / "run-record.json").read_bytes()
    with pytest.raises(SystemExit) as result:
        cli.main("play", [cli.TASK])
    assert result.value.code == 0
    assert commands[-1][4] == "play"
    assert commands[-1][commands[-1].index("--task") + 1] == cli.TASK
    assert commands[-1][commands[-1].index("--checkpoint") + 1] == str(ckpt)
    assert commands[-1][commands[-1].index("--motion-dir") + 1] == str(data)
    assert config_exports == [train]
    assert not (train / "play").exists()
    assert (train / "run-record.json").read_bytes() == record

    prepared = []
    def prepare_task(task, output):
        assert task in cli.TASKS
        prepared.append((task, output))
        output.mkdir()
        (output / "manifest.json").write_text("{}")
        return output

    monkeypatch.setattr(cli, "prepare_task_motions", prepare_task)
    default_run = tmp_path / "task-defaults"
    with pytest.raises(SystemExit) as result:
        cli.main("train", [cli.TASK, "--run-dir", str(default_run)])
    assert result.value.code == 0
    assert prepared == [(cli.TASK, default_run / "motions")]
    assert commands[-1][commands[-1].index("--motion_dir") + 1] == str(default_run / "motions")
    saved = json.loads((default_run / "run-record.json").read_text())
    assert saved["task"] == cli.TASK
    assert saved["manifest"] == str(default_run / "motions/manifest.json")

    with pytest.raises(SystemExit) as result:
        cli.main("train", [cli.TASK])
    assert result.value.code == 0
    automatic = prepared[-1][1].parent
    assert automatic.parent == tmp_path / "logs" / cli.TASK
    datetime.datetime.strptime(automatic.name, "%Y-%m-%d_%H-%M-%S")
    (automatic / "ckpt_000200.pt").touch()
    with pytest.raises(SystemExit) as result:
        cli.main("play", [cli.TASK])
    assert result.value.code == 0
    assert commands[-1][commands[-1].index("--checkpoint") + 1] == str(automatic / "ckpt_000200.pt")



def test_task_motion_paths_support_files_directories_and_globs(tmp_path):
    from source.tasks.blinkmimic.prepare import resolve_motion_files

    nested = tmp_path / "library/nested"
    nested.mkdir(parents=True)
    files = [tmp_path / "single.npz", nested / "one.npz", nested / "two.npz"]
    for path in files:
        path.touch()
    paths = (files[0], tmp_path / "library", nested / "*.npz")
    assert resolve_motion_files(paths) == files
    with pytest.raises(ValueError, match="No motion files matched"):
        resolve_motion_files((tmp_path / "missing/*.npz",))
    with pytest.raises(ValueError, match="at least one motion"):
        resolve_motion_files(())


def test_latest_training_run_uses_task_and_start_time(tmp_path):
    from source.tasks.blinkmimic.cli import TASK, latest_training_checkpoint

    def run(name, started_at, *, task=TASK, action="train", iteration=100):
        directory = tmp_path / TASK / name
        directory.mkdir(parents=True)
        (directory / "run-record.json").write_text(json.dumps(dict(
            task=task, action=action, started_at=started_at,
        )))
        checkpoint = directory / f"ckpt_{iteration:06d}.pt"
        if iteration:
            checkpoint.touch()
        return checkpoint

    # Run names, modification times, and iteration counts do not define run recency.
    expected = run("a_new", "2026-11-17T07:00:00+00:00")
    run("z_old", "2026-11-17T06:00:00+00:00", iteration=900)
    run("starting", "2026-11-17T08:00:00+00:00", iteration=0)
    run("different_task", "2026-11-17T09:00:00+00:00", task="other")
    run("evaluation", "2026-11-17T09:00:00+00:00", action="play")
    run("a_new/play/viewer", "2026-11-17T09:00:00+00:00", action="play")
    incomplete = tmp_path / TASK / "incomplete"
    incomplete.mkdir()
    (incomplete / "run-record.json").write_text('{"task":')
    assert latest_training_checkpoint(tmp_path, TASK) == expected
    with pytest.raises(ValueError, match="No saved training checkpoints"):
        latest_training_checkpoint(tmp_path / "missing", TASK)


def test_training_motion_directory_supports_host_and_container_paths(tmp_path, monkeypatch):
    from source.tasks.blinkmimic import cli

    checkpoint = tmp_path / "ckpt_000100.pt"
    record_path = tmp_path / "run-record.json"
    data = tmp_path / ".blinkmimic-data/dances"
    data.mkdir(parents=True)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    for root in (str(tmp_path), "/different/container/checkout"):
        record_path.write_text(json.dumps(dict(
            cwd=root, manifest=f"{root}/.blinkmimic-data/dances/manifest.json",
        )))
        assert cli.training_motion_directory(checkpoint) == data
    record_path.unlink()
    with pytest.raises(ValueError, match="use --motion-dir"):
        cli.training_motion_directory(checkpoint)


def test_checkpoint_selection_uses_iteration_and_explicit_override(tmp_path):
    from source.tasks.blinkmimic.cli import select_checkpoint

    newest = tmp_path / "ckpt_000300.pt"
    newest.write_bytes(b"newer iteration")
    older = tmp_path / "ckpt_000100.pt"
    older.write_bytes(b"later file modification")
    (tmp_path / "ckpt_000400.pt.tmp").write_bytes(b"save in progress")
    (tmp_path / "ckpt_000500.pt").mkdir()
    (tmp_path / "last.pt").write_bytes(b"not an iteration checkpoint")
    assert select_checkpoint(None, tmp_path) == newest
    assert select_checkpoint(older, tmp_path) == older
    assert select_checkpoint(older, tmp_path / "missing") == older


def test_checkpoint_selection_reports_missing_or_invalid_inputs(tmp_path):
    from source.tasks.blinkmimic.cli import select_checkpoint

    with pytest.raises(ValueError, match="No ckpt"):
        select_checkpoint(None, tmp_path)
    with pytest.raises(ValueError, match="requires"):
        select_checkpoint(None, None)
    invalid = tmp_path / "last.pt"
    invalid.touch()
    with pytest.raises(ValueError, match="Expected"):
        select_checkpoint(invalid, tmp_path)


@pytest.mark.parametrize("mode", ["train", "evaluate", "play"])
def test_worker_uses_task_robot_for_each_entry_point(monkeypatch, mode):
    from dataclasses import replace
    import sys

    import blinkmimic.motion
    from blinkmimic.robot import get_robot
    from source.tasks.blinkmimic import play, worker
    from source.tasks.blinkmimic.config import BlinkMimicTaskCfg

    robot = replace(make_robot(), name="task_robot")
    task = "Cyclo-BlinkMimic-Test"
    monkeypatch.setitem(worker.TASKS, task, BlinkMimicTaskCfg(
        robot_factory=lambda: robot, motion_paths=(),
    ))
    # Restore the process-local upstream registration after this test.
    monkeypatch.setitem(sys.modules, "blinkmimic.task_robot", None)
    monkeypatch.setitem(sys.modules, "blinkmimic.task_robot.task_robot_constants", None)
    monkeypatch.setattr(blinkmimic.motion, "MotionLib", blinkmimic.motion.MotionLib)
    monkeypatch.setattr(sys, "argv", ["worker", mode, "--task", task, "--test-option"])
    calls = []
    monkeypatch.setattr(worker.runpy, "run_module", lambda module, **kw: calls.append(module))
    monkeypatch.setattr(play, "main", lambda **kw: calls.append(kw["robot"]))

    worker.main()

    assert get_robot(robot.name) is robot
    assert calls == ([robot] if mode == "play" else [f"blinkmimic.{mode}"])
    assert sys.argv == ["worker", "--test-option"]
