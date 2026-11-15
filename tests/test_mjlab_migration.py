"""Regression tests for the mjlab 1.6.1 API migration; no GPU required."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mjlab.entity import Entity

from source.assets.robots.robotis_k1.k1_rev1 import (
    FEET_ONLY_COLLISION,
    FULL_COLLISION,
    FULL_COLLISION_WITHOUT_SELF,
    K1_REV1_FOOT_COLLISION_GEOM_PATTERN,
    get_k1_rev1_cfg,
)
from source.tasks.mimic.mdp.commands import ReferenceTrajectoryCommand


@pytest.mark.parametrize(
    "collision,masks",
    [
        (FULL_COLLISION, (1, 1)),
        (FULL_COLLISION_WITHOUT_SELF, (0, 1)),
        (FEET_ONLY_COLLISION, (0, 1)),
    ],
)
def test_compiled_collision_and_actuator_contract(collision, masks):
    import re

    cfg = get_k1_rev1_cfg()
    cfg.collisions = (collision,)
    model = Entity(cfg).spec.compile()
    assert model.nu == 23
    for i in range(model.ngeom):
        name = model.geom(i).name
        if "_collision" not in name:
            continue
        foot = bool(re.fullmatch(K1_REV1_FOOT_COLLISION_GEOM_PATTERN, name))
        if collision is FEET_ONLY_COLLISION and not foot:
            assert model.geom_contype[i] == model.geom_conaffinity[i] == 0
            continue
        assert (model.geom_contype[i], model.geom_conaffinity[i]) == masks
        assert model.geom_condim[i] == (3 if foot else 1)
        assert model.geom_priority[i] == (1 if foot else 0)
        if foot:
            assert model.geom_friction[i, 0] == pytest.approx(0.6)
    assert np.isfinite(model.actuator_gainprm).all()
    assert np.all(model.actuator_gainprm[:, 0] > 0)
    assert np.all(model.actuator_biasprm[:, 1:3] < 0)
    assert np.all(model.actuator_forcerange[:, 1] > 0)


def command_fixture():
    command = object.__new__(ReferenceTrajectoryCommand)
    command.cfg = SimpleNamespace(
        body_names=("root", "tip"),
        start_from_zero=False,
        failure_sampling_ema_alpha=0.1,
        failure_sampling_uniform_ratio=0.1,
        failure_sampling_kernel_size=1,
    )
    command._env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        scene=SimpleNamespace(env_origins=torch.zeros(3, 3)),
        termination_manager=SimpleNamespace(
            terminated=torch.tensor([True, False, True])
        ),
    )
    command._pending_forward = False
    command.frame_ids = torch.tensor([10, 40, 70])
    command.reference_anchor_body_id = command.robot_anchor_body_id = 0
    command.tracked_body_ids = torch.tensor([0, 1])
    quats = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(100, 2, 1)
    command.reference = SimpleNamespace(
        num_frames=100,
        body_position_w=torch.arange(600).reshape(100, 2, 3).float() / 100,
        body_orientation_w=quats,
        body_linear_velocity_w=torch.ones(100, 2, 3),
        body_angular_velocity_w=torch.zeros(100, 2, 3),
    )
    command.robot = SimpleNamespace(
        data=SimpleNamespace(
            body_link_pos_w=torch.zeros(3, 2, 3),
            body_link_quat_w=quats[:3],
            body_com_lin_vel_w=torch.zeros(3, 2, 3),
            body_link_ang_vel_w=torch.zeros(3, 2, 3),
        )
    )
    command.aligned_body_position_w = torch.full((3, 2, 3), -7.0)
    command.aligned_body_orientation_w = torch.full((3, 2, 4), -7.0)
    command.aligned_body_linear_velocity_w = torch.full((3, 2, 3), -7.0)
    command.aligned_body_angular_velocity_w = torch.full((3, 2, 3), -7.0)
    command.start_bin_count = 2
    command.failure_score_by_bin = torch.tensor([2.0, 4.0])
    command._episode_failures_by_bin = torch.tensor([3.0, 5.0])
    command.failure_smoothing_kernel = torch.ones(1)
    command.metrics = {
        name: torch.full((3,), -1.0)
        for name in (
            "start_sampling_entropy",
            "start_sampling_top1_prob",
            "start_sampling_top1_bin",
        )
    }
    return command


def test_partial_reset_advances_only_selected_targets_without_consuming_global_state():
    cmd = command_fixture()
    frames, scores, pending = (
        t.clone()
        for t in (cmd.frame_ids, cmd.failure_score_by_bin, cmd._episode_failures_by_bin)
    )
    cmd._resample_command = lambda ids: (
        pytest.fail("unexpected wrap") if len(ids) else None
    )
    cmd._update_command(torch.tensor([1]))
    frames[1] += 1
    assert torch.equal(cmd.frame_ids, frames)
    assert torch.equal(cmd.failure_score_by_bin, scores)
    assert torch.equal(cmd._episode_failures_by_bin, pending)
    for name in ("position", "orientation", "linear_velocity", "angular_velocity"):
        values = getattr(cmd, f"aligned_body_{name}_w")
        assert torch.all(values[[0, 2]] == -7)
        assert not torch.all(values[1] == -7)


def test_step_advances_once_and_consumes_accumulated_failures():
    cmd = command_fixture()
    cmd._resample_command = lambda ids: None
    cmd._update_command()
    assert torch.equal(cmd.frame_ids, torch.tensor([11, 41, 71]))
    torch.testing.assert_close(cmd.failure_score_by_bin, torch.tensor([2.1, 4.1]))
    assert torch.count_nonzero(cmd._episode_failures_by_bin) == 0


def test_partial_failure_sampling_accumulates_and_scopes_metrics():
    cmd = command_fixture()
    cmd._sample_start_frames(torch.tensor([0]))
    assert torch.equal(cmd._episode_failures_by_bin, torch.tensor([4.0, 5.0]))
    cmd._sample_start_frames(torch.tensor([2]))
    assert torch.equal(cmd._episode_failures_by_bin, torch.tensor([4.0, 6.0]))
    assert all(value[1] == -1 for value in cmd.metrics.values())


@pytest.mark.parametrize("trigger", ["timer", "wrap"])
def test_resample_forwards_before_aligning_targets(trigger):
    cmd = command_fixture()
    calls = []

    def forward():
        calls.append("forward")
        cmd.robot.data.body_link_pos_w[:, 0, 0] = 17.0

    cmd._env.sim = SimpleNamespace(forward=forward)

    def resample(ids):
        if len(ids):
            cmd.frame_ids[ids] = 0
            cmd._pending_forward = True

    cmd._resample_command = resample
    if trigger == "wrap":
        cmd.frame_ids[1] = 99
    else:
        cmd._pending_forward = True
    cmd._update_command()
    assert calls == ["forward"]
    torch.testing.assert_close(
        cmd.aligned_body_position_w[:, 0, 0], torch.full((3,), 17.0)
    )
    assert not cmd._pending_forward


def test_flat_ground_foot_observation_and_clearance_contract():
    from source.tasks.velocity.mdp import feet_clearance, foot_height

    sites = SimpleNamespace(site_ids=[2, 0], name="robot")
    positions = torch.zeros(4, 3, 3)
    positions[:, :, 2] = torch.tensor([0.2, 999.0, 0.4])
    velocities = torch.zeros_like(positions)
    velocities[:, 0, 0] = 3.0
    velocities[:, 2, 1] = 4.0
    velocities[0] = 0.0
    commands = torch.tensor(
        [[1.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.051, 0.0, 0.0], [0.0, 0.0, 0.051]]
    )
    env = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(site_pos_w=positions, site_lin_vel_w=velocities)
            )
        },
        command_manager=SimpleNamespace(get_command=lambda _: commands),
    )
    heights = foot_height(env, sites)
    assert heights.shape == (4, 2)
    torch.testing.assert_close(heights, torch.tensor([[0.4, 0.2]]).repeat(4, 1))
    cost = feet_clearance(env, 0.1, "base_velocity", 0.05, sites)
    torch.testing.assert_close(cost, torch.tensor([0.0, 0.0, 1.5, 1.5]))


def test_upright_preserves_exponential_body_tilt_reward():
    from mjlab.managers import RewardTermCfg
    from mjlab.utils.lab_api.math import quat_from_euler_xyz

    from source.tasks.velocity.mdp import upright

    angles = torch.tensor([0.0, 0.3, -0.7])
    quats = quat_from_euler_xyz(angles, torch.zeros(3), torch.zeros(3))
    asset = SimpleNamespace(name="robot", body_ids=[0])
    env = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    body_link_quat_w=quats[:, None],
                    gravity_vec_w=torch.tensor([0.0, 0.0, -1.0]).repeat(3, 1),
                )
            )
        }
    )
    term = upright(
        RewardTermCfg(func=upright, weight=1.0, params={"asset_cfg": asset}), env
    )
    torch.testing.assert_close(
        term(env, std=0.4, asset_cfg=asset), torch.exp(-angles.sin().square() / 0.4**2)
    )


def test_velocity_reward_and_observation_configuration_contract():
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Cyclo-Velocity-Flat-K1-Rev1-v0")
    assert cfg.rewards["upright"].weight == 1.0
    assert cfg.rewards["upright"].params["asset_cfg"].body_names == ("torso_link",)
    assert cfg.rewards["foot_clearance"].weight == -2.0
    assert cfg.rewards["foot_clearance"].params["command_threshold"] == 0.05
    assert (
        len(
            cfg.observations["critic"]
            .terms["foot_height"]
            .params["asset_cfg"]
            .site_names
        )
        == 2
    )
    assert "default_stand_joint_deviation" not in cfg.rewards


def test_all_reward_names_and_weights_match_previous_baseline():
    import json
    from pathlib import Path

    from mjlab.tasks.registry import load_env_cfg

    contracts = json.loads(
        (Path(__file__).parent / "fixtures/mjlab_1_2_contracts.json").read_text()
    )
    for task, contract in contracts["tasks"].items():
        assert {
            name: term.weight for name, term in load_env_cfg(task).rewards.items()
        } == contract["rewards"]
