from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch,numpy as np
from landing import landing_costs,speed_target,phase_gate,WEIGHTS,TargetedLanding,interpolate_height_speed
from config import env_cfg,base_v2,guard_update,validate_inputs,agent_cfg,MOTION
from source.tasks.mimic.mdp.taeguek_support_v2 import sole_bottom_height

def case(**changes):
    x=dict(time=torch.tensor([27.90]),bottom=torch.tensor([.05]),speed_down=torch.tensor([2.]),reference_bottom=torch.tensor([.09]),reference_vz=torch.tensor([-1.]),loads=torch.zeros(1,2,5),weight_n=350.,physics_dt=.005,height_matched_down=torch.tensor([.7]))
    x.update(changes);return landing_costs(**x)

class LandingTests(unittest.TestCase):
    def test_outside_window_zero(self):
        for t in [0.,7.,27.5,28.36,29.5,38.9]:self.assertEqual(sum(v.sum() for v in case(time=torch.tensor([t]),loads=torch.full((1,2,5),3000.)).values()),0)
    def test_gate_boundaries(self):torch.testing.assert_close(phase_gate(torch.tensor([27.70,28.35])),torch.zeros(2))
    def test_fast_reference_allowed(self):
        self.assertEqual(case(speed_down=torch.tensor([1.9]),reference_vz=torch.tensor([-1.9]),height_matched_down=torch.tensor([1.9]),bottom=torch.tensor([.19]),reference_bottom=torch.tensor([.19]))['braking'],0)
    def test_low_foot_brakes_even_if_reference_fast(self):
        self.assertGreater(case(speed_down=torch.tensor([1.9]),reference_vz=torch.tensor([-1.9]),height_matched_down=torch.tensor([.1]),bottom=torch.tensor([0.]))['braking'],0)
    def test_phase_reference_also_limits(self):
        self.assertGreater(case(reference_vz=torch.tensor([-.4]),height_matched_down=torch.tensor([2.]))['braking'],0)
    def test_slow_touchdown_allowed(self):self.assertEqual(case(speed_down=torch.tensor([.5]),height_matched_down=torch.tensor([0.]))['braking'],0)
    def test_ascending_not_braked(self):self.assertEqual(case(reference_vz=torch.tensor([1.]))['braking'],0)
    def test_height_interpolation(self):
        h=torch.tensor([0.,.1,.2]);v=torch.tensor([0.,1.,2.]);out=interpolate_height_speed(torch.tensor([-.1,.05,.3]),h,v);torch.testing.assert_close(out,torch.tensor([0.,.5,2.]))
    def test_reference_following_never_braked(self):
        speeds=torch.linspace(0,3,100);self.assertTrue(torch.all(speed_target(-speeds,speeds)>=speeds))
    def test_left_foot_cannot_hide_impact(self):
        a=torch.zeros(1,2,5);a[:,0]=1400.;b=a.flip(1)
        for k in ['total_peak','total_rate','total_excess']:torch.testing.assert_close(case(loads=a)[k],case(loads=b)[k])
        self.assertGreater(case(loads=a)['total_peak'],0)
    def test_shared_load_same_total(self):
        a=torch.zeros(1,2,5);a[:,0]=1400.;b=torch.full((1,2,5),700.)
        for k in ['total_peak','total_rate','total_excess']:torch.testing.assert_close(case(loads=a)[k],case(loads=b)[k])
    def test_steady_support_free_total_cost(self):
        c=case(loads=torch.full((1,2,5),175.))
        for k in ['total_peak','total_rate','total_excess']:self.assertEqual(c[k],0)
    def test_large_peak_not_saturated(self):
        vals=[case(loads=torch.full((1,2,5),f))['total_peak'] for f in [700.,1400.,2800.]];self.assertTrue(vals[0]<vals[1]<vals[2])
    def test_history_boundary_rate(self):
        loads=torch.tensor([[[700.,700.,700.,700.,0.],[700.,700.,700.,700.,0.]]]);self.assertGreater(case(loads=loads)['total_rate'],0)
        self.assertEqual(case(loads=loads.flip(-1))['total_rate'],0)
    def test_no_previous_sample_peak_double_count(self):
        loads=torch.zeros(1,2,5);loads[:,:,-1]=5000.;c=case(loads=loads)
        self.assertEqual(c['total_peak'],0);self.assertEqual(c['total_excess'],0)
    def test_integral_dt_scaling(self):
        loads=torch.tensor([[[700.,525.,350.,175.,0.],[700.,525.,350.,175.,0.]]]);c=case(loads=loads)
        expected=((loads[:,:,:-1].sum(1)/350.-3).clamp_min(0)*.005).sum(-1)
        torch.testing.assert_close(c['total_excess']*.02,expected)
    def test_bodyweight_scaling(self):
        a=case(loads=torch.full((1,2,5),1400.));b=case(loads=torch.full((1,2,5),2800.),weight_n=700.)
        for k in a:torch.testing.assert_close(a[k],b[k])
    def test_late_hover_not_free(self):
        c=case(time=torch.tensor([28.1]),reference_bottom=torch.tensor([.001]),reference_vz=torch.tensor([0.]),bottom=torch.tensor([.08]),speed_down=torch.tensor([0.]))
        self.assertGreater(c['late_height'],0);self.assertGreater(c['late_unload'],0)
    def test_braking_not_contact_gated(self):
        torch.testing.assert_close(case()['braking'],case(loads=torch.full((1,2,5),300.))['braking'])
    def test_finite_gradient(self):
        x=torch.tensor([2.],requires_grad=True);c=case(speed_down=x);sum(WEIGHTS[k]*v for k,v in c.items()).sum().backward();self.assertTrue(torch.isfinite(x.grad).all());self.assertGreater(x.grad,0)
    def test_shape_rejection(self):
        with self.assertRaises(ValueError):case(loads=torch.zeros(1,5))
    def test_baseline_rewards_preserved(self):
        a=base_v2.env_cfg();b=env_cfg();self.assertEqual(set(b.rewards)-set(a.rewards),{'targeted_right_landing'})
        for k,v in a.rewards.items():
            w=b.rewards[k];self.assertEqual(v.func,w.func);self.assertEqual(v.weight,w.weight)
            self.assertEqual({**v.params,'ramp_steps':1} if k in ['support_micro_motion','soft_ground_contact'] else v.params,w.params)
    def test_interfaces_unchanged(self):
        a=base_v2.env_cfg();b=env_cfg();self.assertEqual(a.sim,b.sim);self.assertEqual(a.observations,b.observations);self.assertEqual(a.actions,b.actions);self.assertEqual(a.episode_length_s,b.episode_length_s)
    def test_sampling(self):
        c=env_cfg().commands['reference_trajectory'];self.assertEqual(c.first_frame_env_fraction,.3);self.assertEqual(c.focus_env_fraction,.3);self.assertEqual(len(c.focus_windows),9);self.assertEqual(c.focus_windows[-6:],((26.7,27.2),)*6)
    def test_optimizer(self):
        a=agent_cfg();self.assertEqual(a.algorithm.learning_rate,5e-6);self.assertEqual(a.algorithm.entropy_coef,.0003);self.assertEqual(a.max_iterations,10000)
    def test_frozen_inputs(self):validate_inputs()

class GuardTests(unittest.TestCase):
    good=dict(arm_rmse=.09,arm_max=.18,joint_error=.60,length=2300.)
    def test_sustained_regression_stops(self):
        s=guard_update({},250,self.good);bad={**self.good,'arm_rmse':.25}
        s=guard_update(s,1000,bad);self.assertFalse(s['halt']);self.assertTrue(guard_update(s,1250,bad)['halt'])
    def test_one_failure_recovers(self):
        s=guard_update({},1000,{**self.good,'joint_error':2.});self.assertFalse(guard_update(s,1250,self.good)['halt'])
    def test_same_checkpoint_not_double_counted(self):
        s=guard_update({},1000,{**self.good,'arm_rmse':.3});self.assertEqual(s,guard_update(s,1000,self.good))
    def test_baseline_cannot_drift(self):
        s=guard_update({},250,self.good);s=guard_update(s,500,{**self.good,'arm_rmse':.2});self.assertEqual(s['baseline'],self.good)
    def test_invalid_metrics_stop(self):
        self.assertTrue(guard_update({},1000,{**self.good,'arm_rmse':float('nan')})['halt']);self.assertTrue(guard_update({},1000,{})['halt'])
    def test_length_collapse(self):
        s=guard_update({},250,self.good);s=guard_update(s,1000,{**self.good,'length':500.});self.assertIn('episode_length',s['reasons'])

if __name__=='__main__':unittest.main(verbosity=2)
