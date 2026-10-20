"""Behavioral regression checks, independent of MuJoCo rollout dynamics."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from refinement import ReferenceResidualSmoothness, moving_clearance, arm_pose_tracking, env_cfg


def fixture():
    zeros = lambda *shape: torch.zeros(*shape)
    cmd = NS(robot=NS(joint_names=[f'joint_shoulder_{i}' for i in range(10)] + ['hip']),
        cfg=NS(body_names=['torso_link']), frame_ids=torch.zeros(2, dtype=torch.long),
        robot_joint_vel=zeros(2, 11), joint_vel=zeros(2, 11),
        robot_joint_pos=zeros(2, 11), joint_pos=zeros(2, 11),
        robot_body_ang_vel_w=zeros(2, 1, 3), body_ang_vel_w=zeros(2, 1, 3), metrics={})
    env = NS(command_manager=NS(get_term=lambda name: cmd), num_envs=2, device='cpu',
        episode_length_buf=torch.full((2,), 10), step_dt=.02, common_step_counter=24000)
    return env, cmd


class RewardsTest(unittest.TestCase):
    def test_exact_dynamic_reference_is_free(self):
        env, cmd = fixture()
        reward = ReferenceResidualSmoothness(None, env)
        for i in range(20):
            cmd.joint_vel.fill_(10 * (-1)**i)
            cmd.robot_joint_vel.copy_(cmd.joint_vel)
            cmd.body_ang_vel_w.fill_(5 * (-1)**i)
            cmd.robot_body_ang_vel_w.copy_(cmd.body_ang_vel_w)
            self.assertTrue(torch.equal(reward(env), torch.zeros(2)))
            env.common_step_counter += 1
            cmd.frame_ids += 1
        self.assertTrue(torch.equal(arm_pose_tracking(env), torch.ones(2)))

    def test_oscillation_cost_cache_jump_and_partial_reset(self):
        env, cmd = fixture()
        reward = ReferenceResidualSmoothness(None, env)
        for i in range(20):
            cmd.robot_joint_vel.fill_(2 * (-1)**i)
            cmd.robot_body_ang_vel_w[:, :, :2] = .8 * (-1)**i
            result = reward(env).clone()
            old_age = reward.age.clone()
            self.assertTrue(torch.equal(result, reward(env)))
            self.assertTrue(torch.equal(old_age, reward.age))
            env.common_step_counter += 1
            cmd.frame_ids += 1
        self.assertTrue((result > 0).all())
        cmd.frame_ids[0] += 100
        result = reward(env)
        self.assertEqual(float(result[0]), 0.)
        self.assertGreater(float(result[1]), 0.)
        reward.reset(torch.tensor([1]))
        env.common_step_counter += 1
        cmd.frame_ids += 1
        self.assertEqual(float(reward(env)[1]), 0.)

    def test_turn_and_foot_gating(self):
        env, cmd = fixture()
        reward = ReferenceResidualSmoothness(None, env)
        for i in range(20):
            cmd.robot_body_ang_vel_w[:, :, 2] = 5.
            self.assertTrue(torch.equal(reward(env), torch.zeros(2)))
            cmd.frame_ids += 1
            env.common_step_counter += 1
        actual = torch.tensor([.015, .015, .30, .003])
        target = torch.tensor([.002, .002, .30, .002])
        speed = torch.tensor([.3, 2., .1, .3])
        eligible = torch.tensor([True, True, False, True])
        result = moving_clearance(actual, target, speed, eligible)
        self.assertGreater(float(result[0]), 0.)
        self.assertTrue(torch.equal(result[1:], torch.zeros(3)))

    def test_isolation_and_signs(self):
        from source.tasks.mimic.config.k1_rev1.taeguek_support_v2_env_cfg import k1_rev1_taeguek_support_v2_env_cfg
        original = k1_rev1_taeguek_support_v2_env_cfg(5)
        tuned = env_cfg()
        self.assertNotIn('reference_residual_smoothness', original.rewards)
        self.assertEqual(tuned.rewards['reference_residual_smoothness'].weight, -1.)
        self.assertEqual(tuned.rewards['action_rate_l2'].weight, original.rewards['action_rate_l2'].weight)
        self.assertEqual(tuned.commands['reference_trajectory'].trajectory_file,
                         original.commands['reference_trajectory'].trajectory_file)
        self.assertEqual(tuned.decimation, original.decimation)
        self.assertEqual(tuned.sim.mujoco.timestep, original.sim.mujoco.timestep)


if __name__ == '__main__':
    unittest.main(verbosity=2)
