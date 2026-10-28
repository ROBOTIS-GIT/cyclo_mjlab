"""Shared reset-bank setup for limbo evaluation; no training at import."""
import torch
from source.tasks.mimic.mdp import limbo_prepare_v2 as v2

def reset_bank(env, wrap, seed):
    n = env.num_envs
    env.reset(seed=seed)
    c = v2.command(env)
    robot = env.scene['robot']
    ids = torch.arange(n, device=env.device)
    indices = torch.arange(64, -1, -1, device=env.device)
    roots = c.reset_roots[indices].clone()
    roots[:, :3] += env.scene.env_origins
    robot.write_joint_state_to_sim(c.reset_joints[indices], torch.zeros_like(c.reset_joints[indices]), env_ids=ids)
    robot.write_root_link_pose_to_sim(roots, env_ids=ids)
    robot.write_root_link_velocity_to_sim(torch.zeros((n, 6), device=env.device), env_ids=ids)
    robot.clear_state(env_ids=ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.sim.sense()
    c.frame_ids[:] = 0
    for name, value in [('preparing', True), ('prep_origin', True), ('prepared', False),
                        ('prep_timed_out', False), ('prep_elapsed', 0), ('prep_stable', 0),
                        ('prep_steps', 0), ('stable_steps', 0), ('prep_release_pulse', 0),
                        ('progress_valid', False), ('started_at_zero', True),
                        ('episode_seen', False), ('episode_reached', False), ('episode_success', False)]:
        getattr(c, name)[:] = value
    c.prefix_ok[:] = False
    c.prefix_reached_pulse[:] = 0
    c.prefix_failed_pulse[:] = 0
    c.reset_alpha[:] = indices.float()/64
    env.observation_manager.compute(update_history=True)
    return c, wrap.get_observations()
