import unittest
import torch
from precision_costs import stable_mask, costs, WEIGHTS
from refinement_costs import total_impact, landing_gates
from costs import phase_drift

class PrecisionTests(unittest.TestCase):
    def cost(self,t=18.37,speed=.01,locked=True,loads=None):
        if loads is None:loads=torch.full((1,2,5),175.)
        return costs(torch.tensor([t]),loads,350.,.005,
                     torch.full((1,2),locked),torch.zeros(1,2,2),torch.full((1,2),speed))

    def test_force_coverage_no_window_gap_or_double_weight(self):
        loads=torch.tensor([[[1500.,350.,350.,350.,350.],[0.,0.,0.,0.,0.]]])
        base=total_impact(loads,350.,.005)
        for t in [6.85,6.9,7.3,8.6,9.575,18.37,37.75,38.08,38.65,38.97,49.]:
            gate=landing_gates(torch.tensor([t])).amax(-1)
            new=self.cost(t=t,loads=loads)
            for name,value in zip(['outside_peak','outside_rate','outside_excess'],base):
                torch.testing.assert_close(value*gate+new[name],value)

    def test_small_slip_detected_only_when_locked(self):
        self.assertGreater(float(self.cost()['locked_speed']),0.)
        self.assertEqual(float(self.cost(speed=.005)['locked_speed']),0.)
        self.assertEqual(float(self.cost(locked=False)['locked_speed']),0.)

    def test_normal_support_free_and_large_load_jitter_detected(self):
        result=self.cost(speed=0.)
        self.assertTrue(all(float(x)==0. for x in result.values()))
        loads=torch.full((1,2,5),175.)
        loads[:,0,::2]+=100.;loads[:,1,::2]-=100.
        self.assertGreater(float(self.cost(loads=loads)['locked_load_change']),0.)
        self.assertEqual(float(self.cost(loads=loads,locked=False)['locked_load_change']),0.)
        self.assertEqual(float(self.cost(loads=loads)['outside_peak']),0.)

    def test_reference_mask_excludes_dynamic_and_short_support(self):
        f=lambda x:torch.full((80,2),x)
        bottom,speed,yaw,tilt,phase,expected=f(.005),f(.0),f(.0),f(.0),f(1),f(True)
        mask=stable_mask(bottom,speed,yaw,tilt,phase,expected,50.)
        self.assertTrue(mask[3:77].all());self.assertFalse(mask[:3].any());self.assertFalse(mask[77:].any())
        for value,replacement in [(bottom,.03),(speed,.1),(yaw,.6),(tilt,.7),(phase,2),(expected,False)]:
            old=value.clone();value.fill_(replacement)
            self.assertFalse(stable_mask(bottom,speed,yaw,tilt,phase,expected,50.).any())
            value.copy_(old)
        expected.zero_();expected[20:30]=True
        self.assertFalse(stable_mask(bottom,speed,yaw,tilt,phase,expected,50.).any())

    def test_world_anchor_does_not_follow_reference_or_actual_contact(self):
        position=torch.tensor([[[1.,2.],[3.,4.]]]);locked=torch.ones(1,2,dtype=torch.bool)
        continuous=torch.ones(1,dtype=torch.bool)
        drift,anchor=phase_drift(position,torch.zeros_like(position),locked,~locked,continuous)
        self.assertTrue((drift==0).all())
        drift,after=phase_drift(position+.02,anchor,locked,locked,continuous)
        torch.testing.assert_close(drift,torch.full_like(drift,.02))
        torch.testing.assert_close(after,anchor)
        drift,_=phase_drift(position+.02,anchor,locked,locked,~continuous)
        self.assertTrue((drift==0).all())

if __name__=='__main__':unittest.main()
