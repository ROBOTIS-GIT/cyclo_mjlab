"""Behavioral regression tests for phase eligibility and additive task config."""
from unittest import TestCase
import torch
import source.tasks
from mjlab.tasks.registry import load_env_cfg
from source.tasks.mimic.mdp.taeguek_support import support_phase_mask
from source.tasks.mimic.mdp.support_contacts import SupportTracker

class PhaseTests(TestCase):
  def mask(self,bottom=None,speed=None):
    b=torch.zeros(100,2) if bottom is None else bottom
    v=torch.zeros_like(b) if speed is None else speed
    return support_phase_mask(b<.012,b,v,50)

  def test_quiet_stance_retained(self):
    self.assertTrue(self.mask().all())

  def test_low_fast_step_not_mislabelled_support(self):
    v=torch.zeros(100,2);v[20:70,0]=1.5
    m=self.mask(speed=v)
    self.assertFalse(m[20:70,0].any());self.assertTrue(m[:,1].all())

  def test_kick_pivot_keeps_support_and_excludes_kicking_foot(self):
    b=torch.zeros(100,2);b[20:70,1]=.5
    v=torch.ones(100,2)*.5;m=self.mask(b,v)
    self.assertTrue(m[25:65,0].all());self.assertFalse(m[:,1].any())

  def test_boundary_margin_and_short_confidence_flicker(self):
    v=torch.ones(100,2);v[20:60,0]=0;v[70:74,0]=0;m=self.mask(speed=v)
    self.assertFalse(m[20:22,0].any());self.assertTrue(m[22:58,0].all())
    self.assertFalse(m[58:60,0].any());self.assertFalse(m[70:74,0].any())

  def test_short_signal_loss_ignored_but_hop_penalized(self):
    tr=SupportTracker(1,'cpu',.005);expected=torch.tensor([[True,False]])
    for _ in range(20):tr.update(torch.ones(1,2,4)*350,expected)
    counts,missing,_,_=tr.update(torch.tensor([[[350.,0,350,350],[350,350,350,350]]]),expected)
    self.assertEqual(counts.sum(),0)
    counts,missing,_,_=tr.update(torch.zeros(1,2,4),expected)
    self.assertEqual(counts.tolist(),[[1,0]]);self.assertGreater(missing[0,0],0)
    self.assertEqual(missing[0,1],0)

class ConfigTests(TestCase):
  def test_only_sensor_and_reward_added(self):
    for chapter in range(2,6):
      with self.subTest(chapter=chapter):
        base=load_env_cfg(f'Cyclo-Mimic-K1-Rev1-Taeguek{chapter:02d}')
        cfg=load_env_cfg(f'Cyclo-Mimic-K1-Rev1-Taeguek{chapter:02d}-SupportV1')
        for field in ['observations','actions','events','terminations','commands','sim']:
          self.assertEqual(getattr(base,field),getattr(cfg,field),field)
        self.assertEqual(set(cfg.rewards)-set(base.rewards),{'taeguek_support_quality'})
        for k in base.rewards:self.assertEqual(base.rewards[k],cfg.rewards[k])
        self.assertEqual(len(cfg.scene.sensors),len(base.scene.sensors)+1)
        self.assertEqual(cfg.scene.sensors[-1].history_length,cfg.decimation+1)
