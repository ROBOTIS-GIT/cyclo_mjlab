"""Boundary and substep regressions for the entry correction."""
import unittest
from types import SimpleNamespace as S
import numpy as np
import torch
from scripts.reinforcement_learning.build_ai_sapiens_soft_entry_motion import make_prefix
from source.tasks.mimic.mdp.soft_entry import SoftEntryAction, LIMITS, entry_dynamics

class SoftEntryTests(unittest.TestCase):
  def test_prefix_boundary_and_unchanged_later_motion(self):
    x = np.zeros((560,30)); x[:,2]=.12; x[:,6]=1
    x[:,7:]=np.arange(560)[:,None]*.001
    pose=np.array([0.,0.,.08,1.,0.,0.,0.]); joints=np.full(23,.05)
    y=make_prefix(x,pose,joints)
    np.testing.assert_allclose(y[:16,2],.08)
    np.testing.assert_allclose(y[:16,7:],.05)
    np.testing.assert_array_equal(y[75:],x[75:])
    np.testing.assert_allclose(np.linalg.norm(y[:,3:7],axis=1),1.)
    self.assertLess(abs(y[16,2]-y[15,2]),1e-5)
    # The quintic join has no position step or artificial large velocity.
    self.assertLess(abs((y[75,2]-y[74,2])*50),.001)

  def action(self):
    a=object.__new__(SoftEntryAction)
    a._env=S(physics_dt=.002)
    d=S(body_com_lin_vel_w=torch.zeros(2,2,3),body_link_ang_vel_w=torch.zeros(2,2,3),body_link_pos_w=torch.zeros(2,2,3))
    a._entity=S(data=d); a.body_ids=a.ref_ids=[0,1]
    a.prev_velocity=torch.full((2,2,4),99.)
    a.velocity_valid=torch.zeros(2,dtype=torch.bool)
    a.interval_peaks={k:torch.zeros(2) for k in LIMITS}
    a.command=S(body_lin_vel_w=torch.zeros(2,2,3),body_ang_vel_w=torch.zeros(2,2,3),body_pos_w=torch.zeros(2,2,3),
      cfg=S(entry_penalty_seconds=1.5),elapsed=torch.tensor([.1,.1]),penalty_mask=torch.tensor([True,False]),
      metrics={'soft_entry_peak_'+k:torch.zeros(2) for k in LIMITS})
    a.finalized=False
    return a

  def test_substep_spike_and_reset_history(self):
    a=self.action(); a.sample_dynamics()
    self.assertEqual(float(a.interval_peaks['vertical_acceleration'].max()),0.)
    a._entity.data.body_com_lin_vel_w[:,:,2]=.4
    a.sample_dynamics()
    a._entity.data.body_com_lin_vel_w.zero_(); a.sample_dynamics()
    torch.testing.assert_close(a.interval_peaks['vertical_speed'],torch.full((2,),.4))
    torch.testing.assert_close(a.interval_peaks['vertical_acceleration'],torch.full((2,),200.))
    env=S(action_manager=S(get_term=lambda _:a))
    reward=entry_dynamics(env,'vertical_speed')
    self.assertGreater(float(reward[0]),0.)
    self.assertEqual(float(reward[1]),0.)
    a.command.elapsed[:]=1.5
    self.assertEqual(float(entry_dynamics(env,'vertical_speed').max()),0.)

if __name__=='__main__': unittest.main()
