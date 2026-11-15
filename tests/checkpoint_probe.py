"""Fresh-process CPU checkpoint playback/export probe (no viewer or hardware)."""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnx
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from onnx.reference import ReferenceEvaluator

import source.tasks  # noqa: F401
from source.utils import export_sim2real_cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-inference", type=Path)
    args = parser.parse_args()
    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        agent = load_rl_cfg(args.task)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = (load_runner_cls(args.task) or MjlabOnPolicyRunner)(
            wrapped, asdict(agent), None, "cpu"
        )
        runner.load(str(args.checkpoint), map_location="cpu")
        policy = runner.get_inference_policy(device="cpu")
        with torch.inference_mode():
            if args.expected_inference:
                expected = torch.load(
                    args.expected_inference, weights_only=False, map_location="cpu"
                )
                torch.testing.assert_close(
                    policy(expected["obs"]), expected["actions"], rtol=1e-6, atol=1e-6
                )
            for _ in range(3):
                actions = policy(wrapped.get_observations())
                assert actions.shape == (1, 23) and torch.isfinite(actions).all()
                _, rewards, _, _ = wrapped.step(actions)
                assert torch.isfinite(rewards).all()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        runner.export_policy_to_jit(str(args.output_dir), "policy.pt")
        runner.export_policy_to_onnx(str(args.output_dir), "policy.onnx")
        metadata = get_base_metadata(env, str(args.checkpoint))
        assert len(metadata["joint_names"]) == 23
        attach_metadata_to_onnx(args.output_dir / "policy.onnx", metadata)
        export_sim2real_cfg(env, args.output_dir)
        model = onnx.load(args.output_dir / "policy.onnx")
        onnx.checker.check_model(model)
        observations = wrapped.get_observations()
        obs = observations["actor"].detach().cpu()
        with torch.inference_mode():
            live_actions = policy(observations).detach().cpu().numpy()
        scripted = torch.jit.load(str(args.output_dir / "policy.pt"))
        with torch.inference_mode():
            jit_actions = scripted(obs).numpy()
        onnx_actions = ReferenceEvaluator(model).run(
            None, {model.graph.input[0].name: obs.numpy()}
        )[0]
        assert jit_actions.shape == onnx_actions.shape == (1, 23)
        assert np.isfinite(onnx_actions).all()
        np.testing.assert_allclose(jit_actions, live_actions, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(onnx_actions, live_actions, rtol=1e-5, atol=1e-5)
        import yaml

        sim2real = yaml.safe_load(
            (args.output_dir / "params/sim2real.yaml").read_text()
        )
        assert sim2real["policy_joints"] == metadata["joint_names"]
        print("CHECKPOINT_PLAY_EXPORT_OK", args.task, args.checkpoint, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
