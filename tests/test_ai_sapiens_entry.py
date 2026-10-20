"""Deployment entry contract regression tests (CPU, no viewer/GPU required)."""
import unittest
from types import SimpleNamespace as S
from unittest.mock import patch
import torch
from source.tasks.mimic.mdp.ai_sapiens_entry import AiSapiensAction, AiSapiensCommand
from source.tasks.mimic.mdp.commands import ReferenceTrajectoryCommand
from mjlab.envs.mdp.actions import JointPositionAction
from source.tasks.mimic.config.k1_rev1.umr_stand_up_ai_sapiens_env_cfg import k1_rev1_umr_stand_up_ai_sapiens_env_cfg

class EntryContract(unittest.TestCase):
  def test_continuous_blend_preserves_raw_history_and_bypasses_partial_starts(self):
    a = object.__new__(AiSapiensAction)
    a._env = S(physics_dt=.002, sim=S(model=S(actuator_gainprm=torch.zeros(2,1,3), actuator_biasprm=torch.zeros(2,1,3))))
    a.command = S(enabled=torch.tensor([True,False]), elapsed=torch.zeros(2), just_reset=torch.zeros(2,dtype=torch.bool),
      cfg=S(entry_penalty_seconds=1.5), metrics={'handoff_alpha':torch.zeros(2)})
    a._raw_actions = torch.zeros(2,1)
    a._scale = 1.; a._offset = 0.
    a.ready_target = torch.tensor([.36])
    a.applied_target = torch.full((2,1),.36)
    a.previous_target = a.applied_target.clone()
    a.actor_target = a.applied_target.clone()
    a.target_delta = torch.zeros_like(a.applied_target)
    a._processed_actions = a.applied_target.clone()
    a.ready_kp = torch.tensor([150.]); a.policy_kp = torch.tensor([76.45])
    a.ready_kd = torch.tensor([4.]); a.policy_kd = torch.tensor([1.42])
    a.ctrl_ids = torch.tensor([0])
    a.process_actions(torch.full((2,1),.56))
    with patch.object(JointPositionAction, 'apply_actions'):
      for i in range(151):
        a.apply_actions()
        s=min(i*.002/.3,1); alpha=s*s*(3-2*s)
        self.assertAlmostEqual(float(a.applied_target[0,0]), .36+.2*alpha, places=5)
        self.assertAlmostEqual(float(a.applied_target[1,0]), .56, places=6)
        self.assertAlmostEqual(float(a._env.sim.model.actuator_gainprm[0,0,0]),150+(76.45-150)*alpha,places=3)
        torch.testing.assert_close(a._raw_actions,torch.full((2,1),.56))
    a.process_actions(torch.full((2,1),.7))
    with patch.object(JointPositionAction, 'apply_actions'):
      a.apply_actions()
    torch.testing.assert_close(a.applied_target,torch.full((2,1),.7))

  def test_new_reset_exposes_frame_zero_then_advances(self):
    c=object.__new__(AiSapiensCommand)
    c.frame_ids=torch.tensor([0,25]); c.just_reset=torch.tensor([True,False])
    with patch.object(ReferenceTrajectoryCommand,'_update_command',lambda obj:obj.frame_ids.add_(1)):
      c._update_command()
      self.assertEqual(c.frame_ids.tolist(),[0,26])
      c._update_command()
      self.assertEqual(c.frame_ids.tolist(),[1,27])
      c.refresh_reference()
      self.assertEqual(c.frame_ids.tolist(),[1,27])

  def test_task_isolated_and_policy_period_preserved(self):
    from source.tasks.mimic.config.k1_rev1.umr_stand_up_ready_handoff_env_cfg import k1_rev1_umr_stand_up_ready_handoff_env_cfg
    old=k1_rev1_umr_stand_up_ready_handoff_env_cfg()
    new=k1_rev1_umr_stand_up_ai_sapiens_env_cfg()
    self.assertAlmostEqual(new.decimation*new.sim.mujoco.timestep,.02)
    self.assertEqual(new.sim.mujoco.iterations,100)
    self.assertIn('ready_timeout',old.terminations)
    self.assertNotIn('ready_timeout',new.terminations)
    self.assertEqual(len(new.actions['joint_pos'].ready_kp),23)
    self.assertEqual(new.actions['joint_pos'].ready_kd['left_wrist_roll_joint'],10.)
    self.assertNotEqual(old.scene.entities['robot'].spec_fn,new.scene.entities['robot'].spec_fn)

if __name__=='__main__': unittest.main()
