"""Bounded opt-in checks for the stable-backend motion/replay adapter."""

import os

import pytest
import torch
from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg

from source.tasks.supermimic.checkpoint import LocalBundleRunner
from supermimic.scripts.train_utils import create_env

from source.tasks.supermimic import REPO_ROOT
from source.tasks.supermimic.export import nominal_action_contract
from source.tasks.supermimic.motion import prepare_motion_files
from source.tasks.supermimic.config.k1_rev1.base_env_cfg import make_env_cfg

pytestmark = pytest.mark.skipif(
    os.environ.get("CYCLO_RUN_NATIVE_TESTS") != "1",
    reason="Set CYCLO_RUN_NATIVE_TESTS=1 for bounded native CPU regression",
)


@pytest.mark.parametrize(
    "representation, velocity_only",
    [("gencoord", False), ("maxcoord", False), ("gencoord", True)],
)
def test_native_partial_reset_and_multiclip_replay(representation, velocity_only, tmp_path):
    cfg = make_env_cfg(representation)
    source = REPO_ROOT / "source/assets/motions/K1_rev1"
    paths = [source / name / f"{name}.npz" for name in ("dance1", "dance2")]
    originals = [p.read_bytes() for p in paths]
    motion = cfg.commands["motion"]
    motion.motion_file = prepare_motion_files(
        ",".join(map(str, paths)),
        cfg.scene.entities["robot"].spec_fn().compile(),
        tmp_path,
    )
    motion.pose_range, motion.velocity_range = {}, {}
    motion.joint_position_range = (0.0, 0.0)
    motion.sampling_mode = "start"
    cfg.terminations = {
        name: cfg.terminations[name]
        for name in ("motion_end", "time_out", "anchor_pos", "anchor_ori")
    }
    cfg.terminations["anchor_pos"].params["threshold"] = 100.0
    cfg.terminations["anchor_ori"].params["threshold"] = 10.0
    env = create_env(cfg, 3, "cpu")
    try:
        native = env.unwrapped
        assert int(native.sim.mj_model.opt.integrator) == 3  # implicitfast
        env.reset()
        cmd = native.command_manager.get_term("motion")
        selected, others = torch.tensor([1]), torch.tensor([0, 2])
        frames = cmd.time_steps[others].clone()
        poses = native.sim.data.qpos[others].clone()
        aligned = cmd.body_pos_relative_w[others].clone()
        scores = cmd.bin_failed_count.clone()
        cmd._current_bin_failed.fill_(2)
        native.reset(env_ids=selected)
        assert cmd.time_steps[1] == 1  # physical frame zero, next-frame target
        assert torch.equal(cmd.time_steps[others], frames)
        assert torch.equal(cmd.bin_failed_count, scores)
        assert torch.all(cmd._current_bin_failed == 2)
        torch.testing.assert_close(native.sim.data.qpos[others], poses, rtol=0, atol=0)
        torch.testing.assert_close(
            cmd.body_pos_relative_w[others], aligned, rtol=0, atol=0
        )
        assert not cmd._pending_forward
        # A partial reset must not consume the global adaptive failure EMA.
        cmd.cfg.sampling_mode = "adaptive"
        native.step(torch.zeros(3, 23))
        torch.testing.assert_close(
            cmd.bin_failed_count,
            (1 - cmd.cfg.adaptive_alpha) * scores + cmd.cfg.adaptive_alpha * 2,
        )
        assert torch.count_nonzero(cmd._current_bin_failed) == 0
        cmd.cfg.sampling_mode = "start"
        cfg_runner = SuperMimicRunnerCfg(
            representation=representation,
            sampling_mode="start",
            buffer_size=16,
            dynamics_hidden_dims=(16, 16),
            actor_hidden_dims=(16, 16),
            critic_hidden_dims=(16, 16),
        )
        if velocity_only:
            cfg_runner.slot_mix = {"motion": 0.0, "velocity": 1.0}
        runner = LocalBundleRunner(
            manifest={}, env=env, cfg=cfg_runner,
            task_id=f"Cyclo-SuperMimic-{representation}-K1-Rev1-001", device="cpu",
        )
        if velocity_only:
            assert [s.name for s in runner._active_task_specs] == ["velocity"]
            assert [s.name for s in runner.wm_env.task_specs] == ["motion", "velocity"]
        raw = torch.linspace(-1.7, 1.4, 23).repeat(3, 1)
        action = native.action_manager.get_term("joint_pos")
        exported = nominal_action_contract(cfg, list(native.scene["robot"].joint_names))
        for field, actual in (("scale", action.scale), ("offset", action._offset)):
            torch.testing.assert_close(
                torch.tensor(exported[field]).expand_as(actual), actual, rtol=0, atol=0
            )
        action.process_actions(raw)
        torch.testing.assert_close(action.raw_action, raw, rtol=0, atol=0)
        torch.testing.assert_close(
            action._processed_actions, runner._dynamics.proc(raw), rtol=0, atol=0
        )
        end = int(cmd.motion.motion_ends[0])
        cmd.reset_to_frame(torch.tensor([0]), end - 2)
        cmd.set_time_steps(torch.tensor([0]), torch.tensor([end - 1]))
        native.sim.forward()
        cmd.update_relative_body_poses()
        native.episode_length_buf[1] = native.max_episode_length
        runner.step()
        storage = runner.replay_buffer._storage
        assert storage["motion_time"][0, 0] == end - 1
        assert storage["truncated"][0, 0]
        assert not storage["terminated"][0, 0]
        assert storage["truncated"][1, 0] and not storage["terminated"][1, 0]
        assert cmd.time_steps[1] == 1
        assert native.termination_manager.get_term("motion_end").tolist() == [
            True,
            False,
            False,
        ]
        assert cmd.time_steps[0] == 1  # reset, never silently crosses into clip 2
        for _ in range(3):
            runner.step()
        batch = runner.replay_buffer.sample(2, 64)
        assert not batch["truncated"][:, :-1].any()
        assert not batch["terminated"][:, :-1].any()
        torch.testing.assert_close(
            batch["motion_time"][:, 1] - batch["motion_time"][:, 0],
            torch.ones(64, dtype=torch.long),
        )
        causal, _ = runner.replay_buffer.sample_causal(4, 64)
        after_reset = causal["motion_time"][:, -1] != end - 1
        assert not (causal["motion_time"][after_reset] == end - 1).any()
        probabilities = torch.arange(1, cmd.bin_count + 1, dtype=torch.float)
        cmd.set_sampling_distribution(probabilities)
        torch.testing.assert_close(
            cmd._external_probs, probabilities / probabilities.sum()
        )
    finally:
        env.close()
    assert [p.read_bytes() for p in paths] == originals
