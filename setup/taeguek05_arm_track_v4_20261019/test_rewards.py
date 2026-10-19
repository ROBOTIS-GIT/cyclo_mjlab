"""Behavioral checks: large-error recovery, reference holds and reset safety."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_mul
from refinement import ArmRecovery, env_cfg, robust_cost, hold_gate
import smooth_v3_base as base


def fixture():
    z = lambda *shape: torch.zeros(*shape)
    q = z(2, 5, 4); q[..., 0] = 1
    names = ['torso_link', 'left_elbow_link', 'right_elbow_link',
             'left_wrist_roll_rubber_hand', 'right_wrist_roll_rubber_hand']
    cmd = NS(robot=NS(joint_names=[f'shoulder_{i}' for i in range(10)] + ['hip']),
        cfg=NS(body_names=names), frame_ids=torch.zeros(2, dtype=torch.long),
        robot_joint_vel=z(2, 11), joint_vel=z(2, 11), robot_joint_pos=z(2, 11), joint_pos=z(2, 11),
        body_pos_w=z(2, 5, 3), robot_body_pos_w=z(2, 5, 3), body_quat_w=q.clone(),
        robot_body_quat_w=q.clone(), reference=NS(joint_velocity=z(200, 11), fps=50.), metrics={})
    env = NS(command_manager=NS(get_term=lambda _: cmd), num_envs=2, device='cpu',
        episode_length_buf=torch.full((2,), 10), step_dt=.02, common_step_counter=24000)
    return env, cmd


class RewardsTest(unittest.TestCase):
    def test_large_error_recovery_is_dense(self):
        errors = torch.tensor([.2, 1., 2., 3.], requires_grad=True)
        costs = robust_cost(errors / .5)
        self.assertTrue((costs[1:] > costs[:-1]).all())
        costs.sum().backward()
        self.assertTrue((errors.grad > 0).all())
        env, cmd = fixture(); cmd.robot_joint_pos[:, :10] = 2.
        wrong = ArmRecovery(None, env)(env)
        cmd.robot_joint_pos[:, :10] = 1.
        self.assertTrue((ArmRecovery(None, env)(env) < wrong).all())

    def test_exact_fast_motion_is_not_penalized(self):
        env, cmd = fixture(); term = ArmRecovery(None, env)
        for i in range(20):
            cmd.joint_vel.fill_(10 * (-1)**i); cmd.robot_joint_vel.copy_(cmd.joint_vel)
            self.assertTrue(torch.equal(term(env), torch.zeros(2)))
            env.common_step_counter += 1; cmd.frame_ids += 1

    def test_body_error_and_rigid_transform_invariance(self):
        env, cmd = fixture()
        cmd.robot_body_pos_w[:, 3, 0] = .12
        before = ArmRecovery(None, env)(env).clone()
        self.assertTrue((before > 0).all())
        # Apply independent rigid transformations to actual and reference poses.
        for pos, rot, angle in ((cmd.robot_body_pos_w, cmd.robot_body_quat_w, .7),
                                (cmd.body_pos_w, cmd.body_quat_w, -.4)):
            q = torch.tensor([torch.cos(torch.tensor(angle/2)), 0, 0,
                              torch.sin(torch.tensor(angle/2))]).expand_as(rot)
            pos.copy_(quat_apply(q, pos) + torch.tensor([5., -3., 2.]))
            rot.copy_(quat_mul(q, rot))
        torch.testing.assert_close(ArmRecovery(None, env)(env), before)

    def test_holds_reset_frame_jump_and_repeated_calls(self):
        env, cmd = fixture(); term = ArmRecovery(None, env)
        for i in range(20):
            cmd.robot_joint_vel[:, :10] = 2 * (-1)**i
            value = term(env).clone(); age = term.age.clone()
            torch.testing.assert_close(term(env), value)
            torch.testing.assert_close(term.age, age)
            env.common_step_counter += 1; cmd.frame_ids += 1
        self.assertTrue((value > 0).all())
        cmd.frame_ids[0] += 50
        self.assertEqual(float(term(env)[0]), 0.)
        other = term.cached[1].clone(); age = term.age[1].clone()
        term.reset(torch.tensor([0])); self.assertEqual(float(term(env)[0]), 0.)
        torch.testing.assert_close(term.cached[1], other)
        torch.testing.assert_close(term.age[1], age)

    def test_fast_reference_gate_and_pose_recovery(self):
        ref = torch.zeros(100, 10); ref[50] = 5
        gate = hold_gate(ref, 50)
        self.assertTrue((gate[45:56] < .001).all())
        self.assertTrue((gate[:45] == 1).all())
        env, cmd = fixture(); term = ArmRecovery(None, env)
        for _ in range(20):
            cmd.robot_joint_vel[:, :10] = 2.
            term(env); env.common_step_counter += 1; cmd.frame_ids += 1
        hold_near = cmd.metrics['arm_v4_hold_velocity'].clone()
        cmd.robot_joint_pos[:, :10] = 2.; term(env)
        self.assertTrue((cmd.metrics['arm_v4_hold_velocity'] < hold_near).all())
        self.assertTrue((cmd.metrics['arm_v4_joint'] > 0).all())

    def test_legs_and_source_task_unchanged(self):
        env, cmd = fixture(); term = ArmRecovery(None, env)
        for i in range(20):
            cmd.robot_joint_vel[:, -1] = 30 * (-1)**i
            cmd.robot_joint_pos[:, -1] = 2.
            self.assertTrue(torch.equal(term(env), torch.zeros(2)))
            env.common_step_counter += 1; cmd.frame_ids += 1
        old, new = base.env_cfg(), env_cfg()
        self.assertNotIn('arm_recovery', old.rewards)
        for name in old.rewards:
            self.assertEqual(old.rewards[name].weight, new.rewards[name].weight)
        for name in ('commands', 'actions', 'observations', 'events', 'terminations'):
            self.assertEqual(getattr(old, name), getattr(new, name))


if __name__ == '__main__':
    unittest.main(verbosity=2)
