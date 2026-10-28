import unittest
import torch
from mjlab.utils.lab_api.math import quat_apply
from refinement import sidekick_stability_cost,env_cfg

class SidekickTests(unittest.TestCase):
 def test_reference_and_deadband_have_no_cost(self):
  up=torch.tensor([[0.,0.,1.],[0.,0.,1.]])
  tilt,lift=sidekick_stability_cost(torch.tensor([.001,.003]),torch.zeros(2),up,up,torch.ones(2))
  self.assertTrue(torch.all(tilt==0));self.assertTrue(torch.all(lift==0))
 def test_tilt_and_height_cost_only_in_mask(self):
  up=torch.tensor([[.2,0.,.98],[.2,0.,.98]]);ref=torch.tensor([[0.,0.,1.],[0.,0.,1.]])
  a,b=sidekick_stability_cost(torch.tensor([.02,.02]),torch.zeros(2),up,ref,torch.tensor([1.,0.]))
  self.assertGreater(a[0],0);self.assertGreater(b[0],0);self.assertEqual(a[1],0);self.assertEqual(b[1],0);self.assertLess(a[0],1);self.assertLess(b[0],1)
 def test_pure_yaw_rotation_free(self):
  up=torch.tensor([[0.,0.,1.]]);yaw=torch.tensor([[.70710678,0.,0.,.70710678]])
  rotated=quat_apply(yaw,up);a,b=sidekick_stability_cost(torch.zeros(1),torch.zeros(1),rotated,up,torch.ones(1));self.assertAlmostEqual(a.item(),0,places=6)
 def test_cfg_and_inherited_rewards(self):
  c=env_cfg();self.assertEqual(c.rewards['t4_sidekick_stability'].params['ramp_steps'],72000);self.assertEqual(c.rewards['t6_kick_height'].weight,-.25);self.assertEqual(c.rewards['reference_pelvis_xy'].weight,1);self.assertEqual(c.rewards['t4_grf_extra'].params['ramp_steps'],1);self.assertNotIn('t4_target_left_support',c.rewards)
if __name__=='__main__': unittest.main()
