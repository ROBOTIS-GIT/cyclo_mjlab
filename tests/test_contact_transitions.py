import unittest
from types import SimpleNamespace
import torch
from source.tasks.mimic.mdp.contact_transitions import ContactTracker, FootContactTransitions, reference_support_mask
from source.tasks.mimic.performance_runner import KoryoPerformanceRunner, KoryoContactV2Runner
from mjlab.tasks.registry import load_env_cfg


class ContactTransitionTests(unittest.TestCase):
  def setUp(self):
    self.t = ContactTracker(2, 'cpu', .005)
    self.expected = torch.ones(2, 2, dtype=torch.bool)

  def feed(self, values, expected=None):
    loads = torch.tensor(values, dtype=torch.float).view(1,1,-1).expand(2,2,-1)
    return self.t.update(loads, self.expected if expected is None else expected)

  def test_single_touchdown_is_free_and_cross_step_recontact_is_counted_once(self):
    self.assertEqual(self.feed([350]*4)[0].sum().item(),0)
    self.feed([350]*4)
    self.feed([0]*4)
    self.feed([0]*4)
    self.assertEqual(self.feed([350]*4)[0].tolist(),[2,2])
    self.assertEqual(self.feed([350]*4)[0].sum().item(),0)

  def test_hysteresis_keeps_state_through_intermediate_loads(self):
    self.feed([350]*4)
    self.feed([12]*4)
    self.assertTrue(self.t.contact.all())
    self.feed([0]*4)
    self.feed([12]*4)
    self.assertFalse(self.t.contact.any())
    self.assertEqual(self.feed([30]*4)[0].tolist(),[2,2])

  def test_single_substep_dropout_is_debounced(self):
    self.feed([350]*4)
    self.assertEqual(self.feed([350,0,350,350])[0].sum().item(),0)
    self.assertTrue(self.t.contact.all())

  def test_intended_reference_flight_does_not_charge_next_touchdown(self):
    self.feed([350]*4)
    self.feed([0]*4, torch.zeros_like(self.expected))
    self.assertEqual(self.feed([350]*4)[0].sum().item(),0)

  def test_long_air_gap_is_not_a_short_recontact(self):
    self.feed([350]*4)
    for _ in range(10):self.feed([0]*4)
    self.assertEqual(self.feed([350]*4)[0].sum().item(),0)

  def test_brief_force_spikes_cannot_satisfy_sustained_support(self):
    for _ in range(10):_,deficit=self.feed([350,0,0,0])
    self.assertEqual(deficit.tolist(),[2,2])
    for _ in range(10):_,deficit=self.feed([350]*4)
    self.assertEqual(deficit.tolist(),[0,0])

  def test_reset_is_per_environment(self):
    self.feed([350]*4);self.feed([0]*4)
    self.t.reset(torch.tensor([0]))
    self.assertEqual(self.feed([350]*4)[0].tolist(),[0,2])

  def test_reference_height_hysteresis_and_intended_liftoff(self):
    bottom=torch.tensor([[0.,0.],[.016,.016],[.026,.010],[.016,.016],[.005,.005]])
    velocity=torch.zeros_like(bottom);velocity[2,1]=.3
    result=reference_support_mask(bottom,velocity)
    self.assertEqual(result[:,0].tolist(),[True,True,False,False,True])
    self.assertFalse(result[2,1])

  def test_version_migration_is_warmstart_not_resume(self):
    old=object.__new__(KoryoPerformanceRunner);new=object.__new__(KoryoContactV2Runner)
    self.assertTrue(old.resumes_performance_state({'version':1}))
    self.assertFalse(new.resumes_performance_state({'version':1}))
    self.assertTrue(new.resumes_performance_state({'version':2}))
    with self.assertRaises(ValueError):old.resumes_performance_state({'version':2})
    with self.assertRaises(ValueError):new.resumes_performance_state({'version':99})

  def test_new_task_preserves_v1_and_policy_inputs(self):
    old=load_env_cfg('Cyclo-Mimic-K1-Rev1-Koryo-Performance')
    new=load_env_cfg('Cyclo-Mimic-K1-Rev1-Koryo-ContactV2')
    self.assertIn('performance_foot_chatter',old.rewards)
    self.assertNotIn('performance_foot_chatter',new.rewards)
    self.assertEqual(old.commands['reference_trajectory'].trajectory_file,new.commands['reference_trajectory'].trajectory_file)
    self.assertEqual(list(old.observations['actor'].terms),list(new.observations['actor'].terms))

  def test_term_deduplicates_reads_and_resets_teleports(self):
    force=torch.zeros(2,2,5,3);force[...,2]=350
    cmd=SimpleNamespace(frame_ids=torch.tensor([0,0]),metrics={})
    env=SimpleNamespace(common_step_counter=1,episode_length_buf=torch.ones(2,dtype=torch.long),
       cfg=SimpleNamespace(decimation=4),step_dt=.02,
       scene={'performance_feet':SimpleNamespace(data=SimpleNamespace(force_history=force))})
    term=object.__new__(FootContactTransitions)
    term.env=env;term.cmd=cmd;term.tracker=self.t;term.reference_support=torch.ones(100,2,dtype=torch.bool)
    term.last_frame=torch.full((2,),-1,dtype=torch.long);term.age=torch.zeros(2)
    term.last_step=-1;term.cached=torch.zeros(2)
    term(env);age=term.age.clone();term(env)
    torch.testing.assert_close(term.age,age)
    for i in range(1,12):
      env.common_step_counter+=1;env.episode_length_buf+=1;cmd.frame_ids+=1;term(env)
    self.assertTrue((term.age>.15).all())
    cmd.frame_ids[0]=70;cmd.frame_ids[1]+=1;env.common_step_counter+=1
    term(env)
    self.assertLess(term.age[0].item(),.03)
    self.assertGreater(term.age[1].item(),.15)
    self.assertEqual(term.cached[0].item(),0)

if __name__=='__main__':unittest.main()
