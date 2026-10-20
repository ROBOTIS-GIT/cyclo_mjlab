"""Behavioral checks for direction, support/flight separation and episode state."""
import unittest,math
from types import SimpleNamespace as NS
import torch
from refinement import (DirectionalTurnTracking,turn_components,support_geometry_cost,yaw,env_cfg,CHAPTERS)

def quat(a):
    a=torch.as_tensor(a,dtype=torch.float32)
    return torch.stack((torch.cos(a/2),torch.zeros_like(a),torch.zeros_like(a),torch.sin(a/2)),-1)

class RefinementTests(unittest.TestCase):
    def test_opposite_direction_cost(self):
        good=turn_components(torch.tensor([-.04]),torch.tensor([-.04]),torch.tensor([0.]),.02)
        wrong=turn_components(torch.tensor([.04]),torch.tensor([-.04]),torch.tensor([0.]),.02)
        self.assertEqual(sum(x.item() for x in good),0)
        self.assertGreater(wrong[1].item(),1);self.assertGreater(wrong[2].item(),0)

    def test_full_revolution_is_not_zero(self):
        cost=turn_components(torch.zeros(1),torch.zeros(1),torch.tensor([2*math.pi]),.02)[0]
        self.assertGreater(cost.item(),10)

    def test_quaternion_sign_and_branch(self):
        q=quat(torch.tensor([math.pi-.01,-math.pi+.01]))
        torch.testing.assert_close(yaw(q),yaw(-q))

    def geometry(self,bottom,support,air,vz=0.,target=0.,pelvis=.8):
        b=torch.tensor([[bottom,bottom]]);r=torch.full_like(b,target)
        return support_geometry_cost(b,r,torch.full_like(b,vz),torch.zeros_like(b),
            torch.full_like(b,support,dtype=torch.bool),torch.tensor([pelvis]),torch.tensor([.8]),torch.full_like(b,air,dtype=torch.bool))

    def test_support_cost_increases_beyond_old_cap(self):
        self.assertGreater(self.geometry(.067,True,False)[0].sum(),self.geometry(.020,True,False)[0].sum()*2)

    def test_support_height_tolerance(self):
        self.assertEqual(self.geometry(.002,True,False)[0].sum(),0)

    def test_upward_departure(self):
        self.assertGreater(self.geometry(0.,True,False,vz=.3)[1].sum(),0)
        self.assertEqual(self.geometry(0.,True,False,vz=-.3)[1].sum(),0)

    def test_intended_jump_not_pinned(self):
        h,r,f=self.geometry(.35,False,True,vz=2.,target=.35)
        self.assertEqual(h.sum()+r.sum()+f.sum(),0)

    def test_kick_under_and_overshoot(self):
        for actual in [.15,.55]:
            h,r,f=self.geometry(actual,False,True,target=.35)
            self.assertGreater(f.sum(),0);self.assertEqual(h.sum()+r.sum(),0)

    def test_whole_body_hop_not_free_kick(self):
        # Relative height matches but world foot height overshoots by 10 cm.
        self.assertGreater(self.geometry(.45,False,True,target=.35,pelvis=.9)[2].sum(),0)

    def make_turn(self):
        ref=torch.zeros(2000)
        for start in [370,1110]:
            ref[start:start+61]=torch.linspace(0,-math.pi,61)
        cmd=NS(reference=NS(body_orientation_w=quat(ref)[:,None,:],fps=50.),reference_anchor_body_id=0,
               frame_ids=torch.zeros(2,dtype=torch.long),robot_anchor_quat_w=quat(torch.zeros(2)),metrics={})
        env=NS(num_envs=2,device='cpu',step_dt=.02,common_step_counter=0,episode_length_buf=torch.ones(2,dtype=torch.long)*100,
               command_manager=NS(get_term=lambda _:cmd))
        return env,cmd,DirectionalTurnTracking(None,env)

    def test_both_turns_cache_and_reset(self):
        env,cmd,t=self.make_turn()
        maxima=[]
        for start in [370,1110]:
            for j in range(61):
                cmd.frame_ids[:]=start+j
                cmd.robot_anchor_quat_w=quat(torch.tensor([-math.pi*j/60,math.pi*j/60]))
                env.common_step_counter+=1
                cost=t(env).clone();state=t.progress.clone()
                torch.testing.assert_close(cost,t(env),rtol=0,atol=0)
                torch.testing.assert_close(state,t.progress,rtol=0,atol=0)
            self.assertLess(cost[0].item(),1e-4);self.assertGreater(cost[1].item(),1)
            self.assertAlmostEqual(t.progress[1].item(),2*math.pi,places=4)
            maxima.append(cost.tolist())
            # Leaving the window must remove incentive to unwind later.
            cmd.frame_ids[:]=start+70;env.common_step_counter+=1
            self.assertEqual(t(env).sum().item(),0)
        before=t.age[1].clone();t.reset(torch.tensor([0]))
        self.assertEqual(t.age[0].item(),0);self.assertEqual(t.age[1],before)

    def test_configuration_scope(self):
        with self.assertRaises(ValueError):env_cfg(5)
        for c in CHAPTERS:
            cfg=env_cfg(c);p=env_cfg(c,True)
            self.assertGreater(cfg.episode_length_s,30)
            self.assertTrue(p.commands['reference_trajectory'].start_from_zero)
            cmd=cfg.commands['reference_trajectory']
            self.assertEqual(cmd.first_frame_env_fraction,.3)
            self.assertEqual(cmd.entry_env_fraction,0 if c==3 else .15)
            self.assertEqual('targeted_turn' in cfg.rewards,c==3)

if __name__=='__main__':unittest.main(verbosity=2)
