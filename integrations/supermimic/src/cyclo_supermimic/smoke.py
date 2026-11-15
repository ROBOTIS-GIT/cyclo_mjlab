"""Bounded actual simulator -> world model -> differentiable policy smoke."""

import argparse
import tempfile
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--representation", choices=("gencoord", "maxcoord"), default="gencoord"
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    import numpy as np
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from supermimic.runners.supermimic_runner import (
        SuperMimicRunnerCfg,
        build_actor,
        build_world_model,
    )
    from supermimic.tasks.task_spec import build_actor_obs_td

    from .motion import model_joint_names
    from .tasks import k1_env_cfg

    torch.manual_seed(7)
    with tempfile.TemporaryDirectory(prefix="cyclo-supermimic-") as tmp:
        cfg = k1_env_cfg(args.representation, play=True)
        model = cfg.scene.entities["robot"].spec_fn().compile()
        names = model_joint_names(model)
        robot_cfg = cfg.scene.entities["robot"]
        import re

        joints = np.array(
            [
                next(
                    (
                        v
                        for pat, v in robot_cfg.init_state.joint_pos.items()
                        if re.fullmatch(pat, n)
                    ),
                    0.0,
                )
                for n in names
            ],
            dtype=np.float32,
        )
        frames = 100
        file = Path(tmp) / "standing.npz"
        np.savez(
            file,
            fps=50,
            joint_names=names,
            joint_pos=np.tile(joints, (frames, 1)),
            joint_vel=np.zeros((frames, len(names))),
            root_pos_w=np.tile([0, 0, 0.78], (frames, 1)),
            root_quat_w=np.tile([1, 0, 0, 0], (frames, 1)),
            root_lin_vel_w=np.zeros((frames, 3)),
            root_ang_vel_w=np.zeros((frames, 3)),
        )
        cfg.commands["motion"].motion_file = str(file)
        cfg.scene.num_envs = 2
        env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=args.device))
        try:
            rcfg = SuperMimicRunnerCfg(
                representation=args.representation,
                dynamics_hidden_dims=(32, 32),
                actor_hidden_dims=(32, 32),
            )
            wm = build_world_model(rcfg, env, args.device)
            actor = build_actor(rcfg, wm, args.device)
            obs, _ = env.reset()
            state = obs["full_state"]
            rep = wm.representation.represent(state)
            aux = wm.representation.extract_aux_state(state)
            context = wm.sim_context.read(env.unwrapped)
            action = torch.zeros((2, env.num_actions), device=args.device)
            # Real simulator transition supervises the untouched upstream Dynamics.
            next_obs, _, _, _ = env.step(action)
            target = wm.representation.represent(next_obs["full_state"])
            opt = torch.optim.Adam(wm.dynamics.parameters(), lr=1e-4)
            _, prediction = wm.dynamics(rep, action, aux, context)
            loss = (prediction - target.detach()).square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            # Validate gradients through the learned dynamics back to actual actor.
            motion = env.unwrapped.command_manager.get_term("motion")
            _, goal = wm.representation.compute_wm_goal(
                motion.motion, motion.time_steps, aux
            )
            sensor = wm.representation.compute_actor_sensor(
                rep, context, aux_state=aux, last_action=action
            )
            actor_obs = build_actor_obs_td(
                {"motion": goal, "velocity": torch.zeros(2, 3, device=args.device)},
                sensor.unsqueeze(1),
            )
            policy_action = actor(actor_obs)
            _, imagined = wm.dynamics(rep.detach(), policy_action, aux, context)
            policy_loss = imagined.square().mean()
            actor.zero_grad()
            policy_loss.backward()
            gradients = [p.grad for p in actor.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            assert sum(float(g.abs().sum()) for g in gradients) > 0
            assert torch.isfinite(loss) and torch.isfinite(policy_loss)
            print(
                f"PASS {args.representation}: simulator step, WM update loss={loss.item():.6g}, actor gradient through WM; actions={env.num_actions}"
            )
        finally:
            env.close()
