import unittest
import torch
from balance_costs import support_gate,support_terms,recovery_terms
from costs import phase_drift


class BalanceTests(unittest.TestCase):
    def gate(self,phase=2,expected=True,speed=.03,bottom=.005,age=.2):
        f=lambda x:torch.full((1,2),x)
        return support_gate(f(bottom),f(speed),f(phase),f(expected),f(age))

    def test_pivots_included_but_swing_and_relocation_excluded(self):
        self.assertTrue((self.gate()==1).all())
        for kwargs in [dict(phase=0),dict(expected=False),dict(speed=.3),dict(bottom=.03),dict(age=0.)]:
            self.assertFalse(self.gate(**kwargs).any())

    def support(self,speed=.03,loads=None,bottom=.005):
        if loads is None:loads=torch.full((1,2,5),175.)
        g=torch.tensor([[[0.,0.,1.],[0.,0.,1.]]])
        return support_terms(self.gate(),torch.full((1,2),speed),torch.zeros(1,2),
                             loads,350.,torch.full((1,2),bottom),torch.full((1,2),.005),g,g)

    def test_small_slip_has_stronger_signal_without_punishing_zero(self):
        self.assertEqual(float(self.support(speed=0.)['support_slip']),0.)
        self.assertGreater(float(self.support(speed=.008)['support_slip']),0.)
        self.assertGreater(float(self.support()['support_slip']),float(self.support(speed=.008)['support_slip']))

    def test_reference_gate_survives_actual_contact_loss(self):
        normal=self.support(speed=0.)
        self.assertEqual(float(normal['support_gap']),0.)
        self.assertEqual(float(normal['support_height']),0.)
        lost=self.support(loads=torch.zeros(1,2,5),bottom=.03)
        self.assertGreater(float(lost['support_gap']),0.)
        self.assertGreater(float(lost['support_height']),0.)

    def test_light_dual_support_load_transfer_allowed(self):
        loads=torch.full((1,2,5),10.)
        self.assertEqual(float(self.support(loads=loads)['support_gap']),0.)

    def recovery(self,t=7.3,down=2.,height=.04,xy=.3):
        f=lambda v:torch.full((1,2),v)
        return recovery_terms(torch.tensor([t]),f(xy),f(0.),f(height),f(down),
                              f(.04),f(-1.),torch.tensor([-.8]),torch.tensor([-.4]))

    def test_braking_only_first_landing_and_improves_when_slowed(self):
        self.assertGreater(float(self.recovery()['first_foot_braking']),0.)
        self.assertLess(float(self.recovery(down=.2)['first_foot_braking']),float(self.recovery()['first_foot_braking']))
        for t in [3.,8.,38.1]:
            self.assertEqual(float(self.recovery(t=t)['first_foot_braking']),0.)
            self.assertEqual(float(self.recovery(t=t)['first_pelvis_braking']),0.)

    def test_path_cost_remains_informative_far_from_reference(self):
        self.assertEqual(float(self.recovery(xy=0.)['path_recovery']),0.)
        values=[float(self.recovery(xy=x)['path_recovery']) for x in [.1,.3,.6,1.]]
        self.assertTrue(all(b>a for a,b in zip(values,values[1:])))

    def test_reference_relative_anchor_has_no_conflicting_world_target(self):
        expected=torch.ones(1,2,dtype=torch.bool);continuous=torch.ones(1,dtype=torch.bool)
        ref=torch.zeros(1,2,2);actual=ref.clone()
        _,offset=phase_drift(actual-ref,torch.zeros_like(ref),expected,~expected,continuous)
        ref=ref+.012;actual=actual+.012
        drift,_=phase_drift(actual-ref,offset,expected,expected,continuous)
        self.assertTrue((drift==0).all())
        drift,_=phase_drift(actual+.02-ref,offset,expected,expected,continuous)
        torch.testing.assert_close(drift,torch.full_like(drift,.02))

if __name__=='__main__':unittest.main()
