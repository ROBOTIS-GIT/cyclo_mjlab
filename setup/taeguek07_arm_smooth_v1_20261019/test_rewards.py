"""Behavioral tests for reference preservation, reset safety and task isolation."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from refinement import ArmStability, arm_pose_tracking, hold_gate, env_cfg
import targeted_base


def fixture():
    zeros = lambda *shape: torch.zeros(*shape)
    cmd = NS(robot=NS(joint_names=[f'shoulder_{i}' for i in range(10)] + ['hip']),
        frame_ids=torch.zeros(2, dtype=torch.long),
        robot_joint_vel=zeros(2, 11), joint_vel=zeros(2, 11),
        robot_joint_pos=zeros(2, 11), joint_pos=zeros(2, 11),
        reference=NS(joint_velocity=zeros(200, 11), fps=50.), metrics={})
    env = NS(command_manager=NS(get_term=lambda name: cmd), num_envs=2, device='cpu',
        episode_length_buf=torch.full((2,), 10), step_dt=.02, common_step_counter=24000)
    return env, cmd


class RewardsTest(unittest.TestCase):
    def test_exact_fast_motion_has_no_penalty(self):
        env, cmd = fixture(); term = ArmStability(None, env)
        for i in range(20):
            cmd.joint_vel.fill_(10 * (-1)**i)
            cmd.robot_joint_vel.copy_(cmd.joint_vel)
            self.assertTrue(torch.equal(term(env), torch.zeros(2)))
            env.common_step_counter += 1; cmd.frame_ids += 1
        self.assertTrue(torch.equal(arm_pose_tracking(env), torch.ones(2)))

    def test_oscillation_penalized_with_partial_reset_and_frame_jumps(self):
        env, cmd = fixture(); term = ArmStability(None, env)
        for i in range(20):
            cmd.robot_joint_vel.fill_(2 * (-1)**i)
            result = term(env).clone(); age = term.age.clone()
            self.assertTrue(torch.equal(result, term(env)))
            self.assertTrue(torch.equal(age, term.age))
            env.common_step_counter += 1; cmd.frame_ids += 1
        self.assertTrue((result > 0).all())
        cmd.frame_ids[0] += 50
        self.assertEqual(float(term(env)[0]), 0.)
        other = term.cached[1].clone(); age = term.age[1].clone()
        term.reset(torch.tensor([0]))
        self.assertEqual(float(term(env)[0]), 0.)
        self.assertEqual(float(term.cached[1]), float(other))
        self.assertEqual(float(term.age[1]), float(age))
        env.common_step_counter += 1; cmd.frame_ids += 1
        self.assertEqual(float(term(env)[0]), 0.)
        self.assertGreater(float(term.cached[1]), 0.)

    def test_gate_protects_punches_and_does_not_depend_on_robot_speed(self):
        ref = torch.zeros(100, 10); ref[50] = 5
        gate = hold_gate(ref, 50)
        self.assertTrue((gate[45:56] < .001).all())
        self.assertTrue((gate[:45] == 1).all())
        env, cmd = fixture(); term = ArmStability(None, env)
        saved = term.hold.clone(); cmd.robot_joint_vel.fill_(100)
        term(env)
        self.assertTrue(torch.equal(saved, term.hold))

    def test_fixed_wrong_pose_is_not_rewarded_as_correct(self):
        env, cmd = fixture()
        perfect = arm_pose_tracking(env)
        cmd.robot_joint_pos[:, :10] = .3
        self.assertTrue((arm_pose_tracking(env) < perfect).all())

    def test_legs_not_penalized_and_configs_isolated(self):
        env, cmd = fixture(); term = ArmStability(None, env)
        for i in range(20):
            cmd.robot_joint_vel[:, -1] = 10 * (-1)**i
            self.assertTrue(torch.equal(term(env), torch.zeros(2)))
            env.common_step_counter += 1; cmd.frame_ids += 1
        old = targeted_base.env_cfg(7); new = env_cfg()
        self.assertNotIn('arm_stability', old.rewards)
        self.assertEqual(new.rewards['arm_stability'].weight, -1.)
        self.assertEqual(new.rewards['taeguek_support_quality'].func, old.rewards['taeguek_support_quality'].func)
        self.assertEqual(new.rewards['taeguek_support_quality'].params, {'ramp_steps': 1})
        for name in old.rewards:
            self.assertEqual(old.rewards[name].weight, new.rewards[name].weight)
        self.assertEqual(old.commands['reference_trajectory'], new.commands['reference_trajectory'])
        self.assertEqual(old.actions, new.actions)
        self.assertEqual(old.observations, new.observations)
        self.assertEqual(old.decimation, new.decimation)


if __name__ == '__main__':
    unittest.main(verbosity=2)
