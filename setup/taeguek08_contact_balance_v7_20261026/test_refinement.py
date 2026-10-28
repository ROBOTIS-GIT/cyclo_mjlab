import unittest
import torch
from refinement_costs import total_impact, approach_terms, quiet_terms, landing_gates, WEIGHTS, support_gap
from costs import phase_drift
from guard import update_guard

class RefinementTests(unittest.TestCase):
    def test_force_redistribution_cannot_hide_impact(self):
        a=torch.tensor([[[1400.,800.,400.,300.,300.],[0.,0.,0.,0.,0.]]])
        b=a.clone();b[:,0]*=.6;b[:,1]=a[:,0]*.4
        for x,y in zip(total_impact(a,350.,.005),total_impact(b,350.,.005)):
            torch.testing.assert_close(x,y)

    def test_force_cost_monotonic_and_normal_support_free(self):
        values=[]
        for f in [350.,875.,1400.,2100.,3500.]:
            p,r,e=total_impact(torch.full((1,2,5),f/2),350.,.005)
            values.append(float(p+e));self.assertEqual(float(r),0.)
        self.assertEqual(values[0],0.);self.assertEqual(values[1],0.)
        self.assertTrue(all(b>a for a,b in zip(values[1:],values[2:])))

    def test_newest_first_force_rise(self):
        a=torch.zeros(1,2,5);a[:,0]=torch.tensor([1400.,350.,350.,350.,350.])
        self.assertGreater(float(total_impact(a,350.,.005)[1]),0.)
        a[:,0]=torch.tensor([350.,350.,350.,350.,1400.])
        self.assertEqual(float(total_impact(a,350.,.005)[1]),0.)

    def test_excess_integrates_once(self):
        a=torch.zeros(1,2,5);a[:,0]=torch.tensor([1400.,1750.,700.,1050.,350.])
        self.assertAlmostEqual(float(total_impact(a,350.,.005)[2])*.02,4.5*.005)

    def test_target_windows(self):
        g=landing_gates(torch.tensor([3.,7.3,8.05,20.,38.14,42.]))
        self.assertEqual(g.tolist(),[[0.,0.],[0.,1.],[0.,1.],[0.,0.],[1.,0.],[0.,0.]])

    def approach(self,bottom=.03,down=2.,pz=.7,pvz=-.8,occupancy=1.,expected=False):
        f=lambda x:torch.full((1,2),x);s=lambda x:torch.tensor([x])
        return approach_terms(f(bottom),f(down),f(.05),f(-1.5),s(pz),s(.75),s(pvz),s(-.4),f(1.),f(occupancy),torch.full((1,2),expected))

    def test_braking_reduces_cost(self):
        self.assertLess(float(self.approach(down=.5)['foot_braking']),float(self.approach()['foot_braking']))
        self.assertLess(float(self.approach(pvz=-.4)['pelvis_braking']),float(self.approach()['pelvis_braking']))

    def test_early_landing_and_pelvis_collapse_cost(self):
        self.assertGreater(float(self.approach(bottom=0.)['early_height']),float(self.approach(bottom=.05)['early_height']))
        self.assertGreater(float(self.approach(pz=.65)['pelvis_height']),float(self.approach(pz=.75)['pelvis_height']))

    def test_fast_airborne_reference_not_braked(self):
        f=lambda x:torch.full((1,2),x);s=lambda x:torch.tensor([x])
        a=approach_terms(f(.5),f(3.),f(.5),f(-3.),s(.8),s(.8),s(-1.),s(-1.),f(1.),f(0.),torch.zeros(1,2,dtype=torch.bool))
        self.assertTrue(all(float(v)==0 for v in a.values()))

    def test_support_cannot_evade_by_lifting(self):
        f=lambda x:torch.full((1,2),x);s=lambda x:torch.tensor([x])
        a=approach_terms(f(.1),f(0.),f(.005),f(0.),s(.8),s(.8),s(0.),s(0.),f(1.),f(0.),torch.ones(1,2,dtype=torch.bool))
        self.assertGreater(float(a['late_support']),1.)

    def quiet(self,phase=1,t=42.,speed=.2):
        return quiet_terms(torch.tensor([t]),torch.full((1,2,2),.03),torch.full((1,2),speed),torch.zeros(1,2),torch.ones(1,2,3),torch.zeros(1,2),torch.full((1,2),phase),torch.ones(1,2,dtype=torch.bool))

    def test_pivot_and_swing_not_frozen(self):
        for phase in [0,2,3]:self.assertTrue(all(float(v)==0 for v in self.quiet(phase).values()))

    def test_quiet_speed_and_targeted_emphasis(self):
        self.assertGreater(float(self.quiet()['quiet_speed']),float(self.quiet(speed=.01)['quiet_speed']))
        self.assertGreater(float(self.quiet(t=42.)['quiet_drift']),float(self.quiet(t=20.)['quiet_drift']))

    def test_phase_offset_survives_actual_contact_loss(self):
        err=torch.zeros(1,2,2);offset=err.clone();planted=torch.ones(1,2,dtype=torch.bool)
        _,offset=phase_drift(err,offset,planted,~planted,torch.ones(1,dtype=torch.bool))
        drift,_=phase_drift(err+.03,offset,planted,planted,torch.ones(1,dtype=torch.bool))
        self.assertTrue(torch.all(drift==.03))

    def test_tracking_guard_warns_without_halting(self):
        good=dict(arm=.10,hand=.025,tilt=.065,joint=.63,length=2500.)
        state=update_guard({},250,good)
        bad={**good,'arm':.4}
        state=update_guard(state,1000,bad);self.assertFalse(state['halt'])
        state=update_guard(state,1250,bad);self.assertTrue(state['warning']);self.assertFalse(state['halt'])

    def test_v4_observed_arm_regression_is_warned(self):
        baseline=dict(arm=.0979787,hand=.0227354,tilt=.0641851,joint=.6233693,length=2538.)
        actual=dict(arm=.1232677,hand=.0258224,tilt=.0646789,joint=.6870297,length=2543.)
        s=update_guard({'baseline':baseline},1250,actual)
        self.assertIn('arm',s['reasons'])
        self.assertTrue(update_guard(s,1500,actual)['warning'])
        self.assertFalse(update_guard(s,1500,actual)['halt'])

    def gap(self,loads,phase=1,age=.2,speed=0.):
        f=lambda x:torch.full((1,2),x)
        return support_gap(loads,350.,f(speed),f(0.),f(phase),torch.ones(1,2,dtype=torch.bool),f(age))

    def test_support_gap_detects_substep_loss_but_allows_load_transfer(self):
        a=torch.full((1,2,5),175.)
        self.assertEqual(float(self.gap(a)),0.)
        a[:,0]=20.;a[:,1]=330.
        self.assertEqual(float(self.gap(a)),0.)
        a[:,0,1:3]=0.
        self.assertGreater(float(self.gap(a)),0.)

    def test_support_gap_gates_reference_not_actual_contact(self):
        loads=torch.zeros(1,2,5)
        self.assertGreater(float(self.gap(loads)),0.)
        for phase in [0,2,3]:self.assertEqual(float(self.gap(loads,phase=phase)),0.)
        self.assertEqual(float(self.gap(loads,age=0.)),0.)
        self.assertEqual(float(self.gap(loads,speed=.1)),0.)

if __name__=='__main__':unittest.main()
