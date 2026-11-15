"""Opt-in, bounded CPU simulation and PPO regression for the locked runtime.

Run with CYCLO_RUN_NATIVE_TESTS=1; no viewer, rendering or GPU is required.
Set CYCLO_TEST_DEVICE=cuda:0 explicitly for the same bounded GPU checks.
"""

import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import source.tasks  # noqa: F401

pytestmark = pytest.mark.skipif(
    os.environ.get("CYCLO_RUN_NATIVE_TESTS") != "1",
    reason="Set CYCLO_RUN_NATIVE_TESTS=1 for bounded native CPU regression",
)


@pytest.mark.parametrize(
    "task",
    [
        "Cyclo-Velocity-Flat-K1-Rev1-v0",
        "Cyclo-Mimic-K1-Rev1-Dance1",
        "Cyclo-Mimic-K1-Rev1-Dance2",
    ],
)
def test_native_ppo_reset_and_checkpoint(task, tmp_path):
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = 3
    cfg.seed = 7
    env = ManagerBasedRlEnv(cfg=cfg, device=os.environ.get("CYCLO_TEST_DEVICE", "cpu"))
    try:
        obs, _ = env.reset()
        for value in obs.values():
            assert torch.isfinite(value).all()
        contracts = json.loads(
            (Path(__file__).parent / "fixtures/mjlab_1_2_contracts.json").read_text()
        )["tasks"][task]
        assert obs["actor"].shape[-1] == contracts["actor"]
        assert obs["critic"].shape[-1] == contracts["critic"]
        assert env.action_manager.total_action_dim == 23
        assert int(env.sim.mj_model.opt.integrator) == 3  # implicitfast
        assert env.step_dt == pytest.approx(0.02)
        if "reference_trajectory" in cfg.commands:
            cmd = env.command_manager.get_term("reference_trajectory")
            untouched = torch.tensor([0, 2], device=env.device)
            frames = cmd.frame_ids[untouched].clone()
            aligned = cmd.aligned_body_position_w[untouched].clone()
            scores = cmd.failure_score_by_bin.clone()
            cmd._episode_failures_by_bin.fill_(2.0)
            pending = cmd._episode_failures_by_bin.clone()
            qpos = env.sim.data.qpos[untouched].clone()
            env.reset(env_ids=torch.tensor([1], device=env.device))
            assert torch.equal(cmd.frame_ids[untouched], frames)
            assert torch.equal(cmd.aligned_body_position_w[untouched], aligned)
            assert torch.equal(cmd.failure_score_by_bin, scores)
            assert torch.equal(cmd._episode_failures_by_bin, pending)
            torch.testing.assert_close(env.sim.data.qpos[untouched], qpos)
            # Clip wrap must refresh derived state before anchor alignment.
            cmd.frame_ids[1] = cmd.reference.num_frames - 1
            cmd._update_command()
            assert cmd.frame_ids[1] < cmd.reference.num_frames
            assert not cmd._pending_forward
            anchor = cmd.robot_anchor_pos_w[1]
            reference_anchor = cmd.aligned_body_position_w[
                1, cmd.reference_anchor_body_id
            ]
            torch.testing.assert_close(reference_anchor[:2], anchor[:2])

        agent_cfg = load_rl_cfg(task)
        agent_cfg.num_steps_per_env = 4
        agent_cfg.algorithm.num_learning_epochs = 1
        agent_cfg.algorithm.num_mini_batches = 1
        agent_cfg.logger = "tensorboard"
        agent_cfg.upload_model = False
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(agent_cfg), str(tmp_path), env.device)
        initial = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
        checkpoint = tmp_path / "model_0.pt"
        saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
        final = runner.alg.actor.state_dict()
        assert any(not torch.equal(initial[k], v) for k, v in final.items())
        for name, value in final.items():
            torch.testing.assert_close(
                saved["actor_state_dict"][name], value.detach().cpu(), rtol=0, atol=0
            )
        # Reload into a fresh model, as the play/resume entry points do.
        runner = runner_cls(wrapped, asdict(agent_cfg), None, env.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.device)
        with torch.inference_mode():
            for _ in range(3):
                actions = policy(wrapped.get_observations())
                assert actions.shape == (3, 23)
                assert torch.isfinite(actions).all()
                _, rewards, _, _ = wrapped.step(actions)
                assert torch.isfinite(rewards).all()
    finally:
        env.close()


@pytest.mark.parametrize(
    "task", ["Cyclo-Mimic-K1-Rev1-Dance1", "Cyclo-Mimic-K1-Rev1-Dance2"]
)
def test_explicit_and_automatic_reset_target_alignment(task):
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = 3
    cfg.seed = 19
    cfg.terminations = {"time_out": cfg.terminations["time_out"]}
    motion = cfg.commands["reference_trajectory"]
    motion.start_from_zero = True
    motion.reset_pose_noise = {}
    motion.reset_velocity_noise = {}
    motion.reset_joint_position_noise = (0.0, 0.0)
    env = ManagerBasedRlEnv(cfg=cfg, device=os.environ.get("CYCLO_TEST_DEVICE", "cpu"))
    try:
        env.reset()
        cmd = env.command_manager.get_term("reference_trajectory")
        selected = torch.tensor([1], device=env.device)
        untouched = torch.tensor([0, 2], device=env.device)
        frames = cmd.frame_ids[untouched].clone()
        qpos = env.sim.data.qpos[untouched].clone()
        scores = cmd.failure_score_by_bin.clone()
        cmd._episode_failures_by_bin.fill_(2.0)
        env.reset(env_ids=selected)
        assert (
            cmd.frame_ids[1] == 1
        )  # Physical state is frame zero, target is next frame.
        explicit_qpos = env.sim.data.qpos[1].clone()
        explicit_target = cmd.aligned_body_position_w[1].clone()
        torch.testing.assert_close(env.sim.data.qpos[untouched], qpos, rtol=0, atol=0)
        assert torch.equal(cmd.frame_ids[untouched], frames)
        assert torch.equal(cmd.failure_score_by_bin, scores)
        assert torch.all(cmd._episode_failures_by_bin == 2.0)
        env.episode_length_buf[1] = env.max_episode_length
        _, _, _, truncated, _ = env.step(torch.zeros(3, 23, device=env.device))
        assert truncated[1]
        assert cmd.frame_ids[1] == 1
        torch.testing.assert_close(
            env.sim.data.qpos[1], explicit_qpos, rtol=0, atol=1e-6
        )
        torch.testing.assert_close(
            cmd.aligned_body_position_w[1], explicit_target, rtol=0, atol=1e-6
        )
        assert torch.equal(cmd.frame_ids[untouched], frames + 1)
        alpha = cmd.cfg.failure_sampling_ema_alpha
        torch.testing.assert_close(
            cmd.failure_score_by_bin, (1 - alpha) * scores + alpha * 2.0
        )
        assert torch.count_nonzero(cmd._episode_failures_by_bin) == 0
    finally:
        env.close()
