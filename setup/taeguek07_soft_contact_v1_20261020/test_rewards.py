from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from refinement import impact_costs,approach_cost,env_cfg,arm_base,validate_inputs
from foot_micro import foot_costs,phase_drift

class SoftContactTests(unittest.TestCase):
    def test_standing_weight_is_free(self):
        p,r=impact_costs(torch.full((3,2,5),175.),350.,.005)
        self.assertEqual((p+r).sum(),0)
    def test_impacts_increase_without_hard_cap(self):
        vals=[]
        for n in [1000.,1500.,2000.]:
            p,_=impact_costs(torch.full((1,2,5),n),350.,.005);vals.append(p.item())
        self.assertTrue(vals[0]<vals[1]<vals[2])
    def test_rise_not_unloading(self):
        _,up=impact_costs(torch.tensor([[[1000.,100.],[0.,0.]]]),350.,.005)
        _,down=impact_costs(torch.tensor([[[100.,1000.],[0.,0.]]]),350.,.005)
        self.assertGreater(up.item(),0);self.assertEqual(down.item(),0)
    def test_loading_rate_timestep_scaling(self):
        _,a=impact_costs(torch.tensor([[[500.,100.],[0.,0.]]]),350.,.005)
        _,b=impact_costs(torch.tensor([[[300.,100.],[0.,0.]]]),350.,.0025)
        torch.testing.assert_close(a,b)
    def test_airborne_fast_kick_allowed(self):
        speed=torch.tensor([[4.,4.]])
        self.assertEqual(approach_cost(speed,torch.full((1,2),.5),torch.full((1,2),.5)),0)
        self.assertGreater(approach_cost(speed,torch.full((1,2),.02),torch.full((1,2),.02)),0)
        self.assertEqual(approach_cost(torch.full((1,2),.3),torch.zeros(1,2),torch.zeros(1,2)),0)
    def test_reference_movement_free_and_swing_excluded(self):
        z2=torch.zeros(1,2);z3=torch.zeros(1,2,3);zxy=torch.zeros(1,2,2)
        self.assertEqual(sum(v.sum() for v in foot_costs(zxy,z3,z3,z2,z2,z2,torch.ones(1,2,dtype=torch.bool)).values()),0)
        self.assertEqual(sum(v.sum() for v in foot_costs(zxy+1,z3+4,z3+100,z2+1,z2,z2+2,torch.zeros(1,2,dtype=torch.bool)).values()),0)
    def test_preserve_old_rewards_and_full_clip(self):
        old=arm_base.env_cfg();new=env_cfg()
        for k,v in old.rewards.items():
            if k!='arm_stability':self.assertEqual(v,new.rewards[k])
        self.assertEqual(new.rewards['arm_stability'].params,{'ramp_steps':1})
        self.assertEqual(new.rewards['taeguek_support_quality'].params,{'ramp_steps':1})
        self.assertGreater(new.episode_length_s,46.38)
    def test_frozen_inputs(self):validate_inputs()

if __name__=='__main__':unittest.main(verbosity=2)
