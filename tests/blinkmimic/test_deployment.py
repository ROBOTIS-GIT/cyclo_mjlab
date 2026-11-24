"""Deployment settings reflect the resolved training arguments and robot configuration."""

import os
import subprocess
import sys

import pytest
import yaml

from source.tasks.blinkmimic.runtime import ROOT, prepare

prepare()
from source.tasks.blinkmimic.cli import export_training_config_when_ready
from source.tasks.blinkmimic.config.k1_rev1.robot_spec import make_robot


@pytest.mark.parametrize(
    "flags,history,future,reference_skip",
    [([], 10, 10, False), (["--hist", "3", "--goal_frames", "2", "--pi_ref_skip"], 3, 2, True)],
)
def test_launcher_exports_resolved_training_config(
    tmp_path, flags, history, future, reference_skip
):
    # Execute the real trainer's argument parser and startup in an isolated process.
    # Stop at its first GPU-backed object; the launcher consumes the saved args.json.
    script = """
from unittest.mock import patch
from source.tasks.blinkmimic.runtime import prepare
from source.tasks.blinkmimic.worker import main
prepare()
class ReachedGpuInitialization(Exception):
    pass
with patch('blinkmimic.kinematics.Kinematics', side_effect=ReachedGpuInitialization):
    try:
        main()
    except ReachedGpuInitialization:
        pass
    else:
        raise AssertionError('Trainer did not reach GPU initialization')
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, "train", "--task", "Cyclo-BlinkMimic-K1-Rev1-001", "--robot", "k1",
         "--log_dir", str(tmp_path), *flags],
        cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        config_path = export_training_config_when_ready(child, tmp_path, make_robot())
        stdout, stderr = child.communicate(timeout=60)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    assert child.returncode == 0, stdout + stderr
    env_cfg = yaml.safe_load((tmp_path / "params/env.yaml").read_text())
    assert env_cfg["robot"]["joint_names"] == list(make_robot().joint_names)
    assert env_cfg["robot"]["robot_cfg"]["spec_fn"].endswith("._load_k1_rev1_spec")
    assert config_path == tmp_path / "params/sim2real.yaml"
    cfg = yaml.safe_load((tmp_path / "params/sim2real.yaml").read_text())
    robot = make_robot()
    assert cfg["policy_joints"] == list(robot.joint_names)
    assert cfg["step_dt"] == .02
    assert cfg["actions"]["joint_pos"]["scale"] == [float(f"{value:.3g}") for value in robot.action_scale_list()]
    observations = cfg["observations"]["blinkmimic"]
    assert observations["history"]["layout"] == "frame_major"
    assert list(observations["history"]["terms"]) == [
        "joint_pos_rel", "joint_vel", "base_ang_vel", "projected_gravity",
        "last_action", "reference_orientation",
    ]
    assert all(term["history_length"] == history for term in observations["history"]["terms"].values())
    assert observations["motion_goal"]["frames"] == future
    assert observations["motion_goal"]["groups"][0]["terms"]["future_joint_vel"]["scale"] == [0.1] * robot.num_joints
    assert observations["reference_skip_in_onnx"] is reference_skip


@pytest.mark.parametrize("contents", [None, '{"hist":'])
def test_launcher_rejects_missing_or_incomplete_settings_after_trainer_exit(tmp_path, contents):
    if contents is not None:
        (tmp_path / "args.json").write_text(contents)
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=10)
    with pytest.raises(RuntimeError, match="before saving complete args.json"):
        export_training_config_when_ready(child, tmp_path, make_robot())
    assert not (tmp_path / "params/sim2real.yaml").exists()
