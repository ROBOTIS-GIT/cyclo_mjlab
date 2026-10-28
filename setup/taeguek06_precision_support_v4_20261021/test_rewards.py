import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse
from kick_support import correction_costs, correction_ramp, kick_support_mask
from refinement import env_cfg, agent_cfg, validate_inputs, SOURCE


class RewardTests(unittest.TestCase):
    def costs(self, bottom=0., load=350., angular=None, expected=True):
        z=torch.full((1,2),bottom);g=torch.tensor([[[0.,0.,1.],[0.,0.,1.]]])
        w=torch.zeros(1,2,3) if angular is None else angular
        return correction_costs(z,g,g,w,torch.zeros_like(w),
            torch.full((1,2,4),load),350.,torch.tensor([[expected,False]]))

    def test_no_cost_for_supported_flat_foot(self):
        self.assertTrue(all(v.sum()==0 for v in self.costs().values()))

    def test_small_hop_and_low_hover_are_not_free(self):
        c=self.costs(bottom=.0039,load=0.)
        self.assertGreater(c['height'].sum(),0)
        self.assertEqual(c['deficit'].sum(),1)
        self.assertEqual(self.costs(bottom=.0005,load=0.)['deficit'].sum(),1)

    def test_intentional_kick_and_noneligible_steps_are_free(self):
        self.assertTrue(all(v.sum()==0 for v in self.costs(bottom=1.,load=0.,expected=False).values()))

    def test_pivot_yaw_is_free(self):
        w=torch.tensor([[[0.,0.,8.],[0.,0.,0.]]])
        self.assertEqual(self.costs(angular=w)['rocking'].sum(),0)
        up=torch.tensor([[[0.,0.,1.]]]);q=torch.tensor([[[.70710678,0.,0.,.70710678]]])
        torch.testing.assert_close(quat_apply_inverse(q,up),up)

    def test_geometry_gates_do_not_use_actual_flight(self):
        b=torch.zeros(100,2);b[20:70,1]=.8
        mask=kick_support_mask(b,50)
        self.assertTrue(mask[25:65,0].all())
        self.assertFalse(mask[:,1].any())
        self.assertFalse(mask[:20].any())
        self.assertFalse(mask[20,0])  # trim entry boundary
        b[40:50,0]=.1
        self.assertFalse(kick_support_mask(b,50)[40:50].any())

    def test_existing_rewards_stay_mature(self):
        cfg=env_cfg();validate_inputs()
        for key in ['t6_arm_recovery','t6_kick_height','t6_soft_contact']:
            self.assertEqual(cfg.rewards[key].params['ramp_steps'],1)
        self.assertEqual(cfg.commands['reference_trajectory'].first_frame_env_fraction,.3)
        self.assertEqual(agent_cfg().algorithm.learning_rate,3e-6)
        self.assertTrue(SOURCE.exists())

    def test_ramp_only_new_correction(self):
        self.assertEqual(correction_ramp(0),0)
        self.assertEqual(correction_ramp(36000),.5)
        self.assertEqual(correction_ramp(72000),1)
        self.assertEqual(correction_ramp(240000),1)

    def test_no_saturation_hides_larger_clearance(self):
        self.assertGreater(self.costs(bottom=.10)['height'].sum(),self.costs(bottom=.03)['height'].sum())

if __name__=='__main__':unittest.main()
