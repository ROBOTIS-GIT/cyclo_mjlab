import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import unittest
import torch
from contact import impact_costs,approach_cost,compress,landing_masks,body_support_deficit
from source.tasks.mimic.mdp.support_contacts import SupportTracker
from refinement import env_cfg,guard_update

class ContactBehavior(unittest.TestCase):
    def test_steady_body_weight_free(self):
        peak,rate=impact_costs(torch.full((2,2,5),175.),350.,.005)
        self.assertEqual(float(peak.sum()+rate.sum()),0.)
    def test_substep_peak_survives_policy_average(self):
        loads=torch.tensor([[[0.,0.,1400.,0.,0.],[0.,0.,0.,0.,0.]]])
        peak,rate=impact_costs(loads,350.,.005)
        self.assertGreater(peak.item(),0);self.assertGreater(rate.item(),0)
    def test_unloading_not_loading(self):
        _,rate=impact_costs(torch.tensor([[[0.,0.,1400.,1400.,1400.],[0.,0.,0.,0.,0.]]]),350.,.005)
        self.assertEqual(rate.item(),0)
    def test_body_weight_scaling(self):
        f=torch.rand(5,2,5)*1400
        a=impact_costs(f,350,.005);b=impact_costs(f*2,700,.005)
        for x,y in zip(a,b):torch.testing.assert_close(x,y)
    def test_large_impacts_remain_distinct(self):
        costs=[]
        for force in (1000.,1400.,2000.):
            p,r=impact_costs(torch.tensor([[[force]*5,[0.]*5]]),350,.005);costs.append(compress(.4*p+.04*r).item())
        self.assertTrue(costs[0]<costs[1]<costs[2])
    def test_approach_protects_high_kick_and_departure(self):
        down=torch.tensor([[3.,3.]])
        self.assertEqual(approach_cost(down,torch.tensor([[.3,.3]]),torch.ones(1,2,dtype=torch.bool)).item(),0)
        self.assertEqual(approach_cost(down,torch.zeros(1,2),torch.zeros(1,2,dtype=torch.bool)).item(),0)
        self.assertGreater(approach_cost(down,torch.zeros(1,2),torch.ones(1,2,dtype=torch.bool)).item(),0)
    def test_reference_only_labels_cover_landing_and_clear_next_flight(self):
        z=torch.zeros(100,2);g=torch.ones(100,2,dtype=torch.bool);speed=torch.zeros(100,2)
        z[20:30,0]=.2;g[20:30,0]=False
        z[45:55,0]=.2;g[45:55,0]=False
        a,s=landing_masks(z,g,speed,50.)
        self.assertTrue(a[27,0]);self.assertTrue(s[40,0]);self.assertFalse(s[45,0]);self.assertFalse(s[:,1].any())
        speed[:]=2.;_,fast=landing_masks(z,g,speed,50.);self.assertFalse(fast.any())
    def test_settling_tracks_long_unload_across_steps_and_resets(self):
        t=SupportTracker(2,'cpu',.005);expected=torch.ones(2,2,dtype=torch.bool)
        for _ in range(5):t.update(torch.ones(2,2,4)*175,expected)
        loss=0
        for _ in range(10):
            events,missing,_,_=t.update(torch.zeros(2,2,4),expected);loss+=events.sum().item()
        self.assertEqual(loss,4);self.assertTrue((missing==1).all());self.assertTrue((t.unload_age>.15).all())
        t.reset(torch.tensor([0]));self.assertEqual(t.unload_age[0].sum(),0);self.assertGreater(t.unload_age[1].sum(),0)
    def test_body_support_does_not_penalize_transfer_or_reference_flight(self):
        support=torch.tensor([[True,False]])
        transfer=torch.tensor([[[350.,250.,100.,0.],[0.,100.,250.,350.]]])
        self.assertEqual(body_support_deficit(transfer,350,support).item(),0.)
        self.assertEqual(body_support_deficit(torch.zeros_like(transfer),350,torch.zeros_like(support)).item(),0.)
        self.assertEqual(body_support_deficit(torch.zeros_like(transfer),350,support).item(),1.)
    def test_original_tracking_and_single_support_objective_preserved(self):
        c=env_cfg();self.assertEqual(c.rewards['arm_recovery'].params['ramp_steps'],1)
        self.assertEqual(c.rewards['arm_pose_tracking'].weight,.35)
        self.assertEqual(len([k for k in c.rewards if 'support' in k]),1)
        self.assertEqual(c.sim.mujoco.timestep*c.decimation,.02)
        self.assertGreater(c.episode_length_s,40.32)
    def test_tracking_guard_requires_persistent_regression(self):
        good={'length':1700.,'arm':.1,'hand':.03,'joint':.5};bad={**good,'arm':.5}
        s=guard_update({},500,good);s=guard_update(s,1000,bad);self.assertFalse(s['halt'])
        s=guard_update(s,1500,bad);self.assertTrue(s['halt'])
if __name__=='__main__':unittest.main()
