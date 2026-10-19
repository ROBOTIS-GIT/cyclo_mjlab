"""Behavioral coverage for full-disturbance scratch support training."""
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import torch
import source.tasks
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from source.tasks.mimic.koryo_schedule import initial_state
from source.tasks.mimic.mdp.koryo_curriculum import KoryoCurriculumCommandCfg, KoryoObservationNoise
from source.tasks.mimic.mdp.support_contacts import (
  SupportTracker, FootSupportQuality, sole_horizontal_speed, excess_motion_cost,
)
from source.tasks.mimic.performance_runner import KoryoSupportMaxRunner, KoryoContactMaxRunner

TASK = 'Cyclo-Mimic-K1-Rev1-Koryo-SupportV3-Max'


class SupportTrackerTests(TestCase):
  def setUp(self):
    self.tracker = SupportTracker(2, 'cpu', .005)
    self.expected = torch.ones(2, 2, dtype=torch.bool)

  def feed(self, loads, expected=None):
    loads = torch.tensor(loads, dtype=torch.float).view(1,1,-1).expand(2,2,-1)
    return self.tracker.update(loads, self.expected if expected is None else expected)

  def planted(self):
    for _ in range(10):
      self.feed([350]*4)

  def test_long_unload_does_not_expire_or_need_actual_contact(self):
    self.planted()
    events, _, _, _ = self.feed([0]*4)
    self.assertEqual(events.tolist(), [[1,1],[1,1]])
    for _ in range(30):
      events, missing, duration, occupancy = self.feed([0]*4)
    self.assertEqual(events.sum().item(), 0)
    torch.testing.assert_close(missing, torch.ones(2,2))
    torch.testing.assert_close(duration, torch.full((2,2), 4.))
    self.assertEqual(occupancy.sum().item(), 0)
    self.assertEqual(self.feed([350]*4)[0].sum().item(), 0)
    self.assertEqual(self.feed([350]*4)[1].sum().item(), 0)

  def test_first_touchdown_flight_and_landing_grace(self):
    self.assertEqual(self.feed([350]*4)[0].sum().item(), 0)
    self.planted()
    events, missing, duration, _ = self.feed([0]*4, torch.zeros_like(self.expected))
    self.assertEqual((events+missing+duration).sum().item(), 0)
    self.assertEqual(self.feed([0]*4)[1].sum().item(), 0)
    self.assertEqual(self.feed([350]*4)[0].sum().item(), 0)

  def test_debounce_hysteresis_and_force_spikes(self):
    self.planted()
    self.assertEqual(self.feed([350,0,350,350])[0].sum().item(), 0)
    self.feed([12]*4)
    self.assertTrue(self.tracker.contact.all())
    self.feed([0]*4)
    for _ in range(10):
      _, missing, _, occupancy = self.feed([350,0,0,0])
    self.assertEqual(occupancy.sum().item(), 0)
    self.assertEqual(missing.sum().item(), 4)

  def test_reset_is_local_and_does_not_invent_liftoff(self):
    self.planted()
    self.tracker.reset(torch.tensor([0]))
    events, _, _, _ = self.feed([0]*4)
    self.assertEqual(events.tolist(), [[0,0],[1,1]])
    self.assertLess(self.tracker.expected_age[0,0], .03)
    self.assertGreater(self.tracker.expected_age[1,0], .2)

  def test_contact_point_stationary_despite_moving_com(self):
    q=torch.tensor([[[1.,0,0,0]]])
    v=torch.tensor([[[1.,0,0]]]);w=torch.tensor([[[0.,1,0]]])
    offsets=torch.zeros(1,1,3);radii=torch.ones(1,1);com=torch.zeros(1,3)
    speed=sole_horizontal_speed(q,v,w,offsets,radii,com)
    torch.testing.assert_close(speed,torch.zeros(1,1))
    speed=sole_horizontal_speed(q,v,torch.zeros_like(w),offsets,radii,com)
    torch.testing.assert_close(speed,torch.ones(1,1))

  def test_reference_pivot_allowed_and_recovery_cost_bounded(self):
    speed=torch.tensor([[.4,.3]]);w=torch.tensor([[[1.,0,3.],[0,2.,-4.]]])
    slip,rotation=excess_motion_cost(speed,speed,w,w)
    self.assertEqual((slip+rotation).sum().item(),0)
    slip,rotation=excess_motion_cost(speed*100,speed,w*100,w)
    torch.testing.assert_close(slip,torch.full((1,2),4.))
    torch.testing.assert_close(rotation,torch.full((1,2),4.))

  def test_reward_step_cache_reset_and_no_contact_cost(self):
    force=torch.zeros(2,2,5,3);force[...,2]=350
    zeros=torch.zeros(2,2,3);quats=torch.zeros(2,2,4);quats[...,0]=1
    cmd=SimpleNamespace(frame_ids=torch.zeros(2,dtype=torch.long),metrics={},
      robot_body_quat_w=quats,robot_body_lin_vel_w=zeros,robot_body_ang_vel_w=zeros,body_ang_vel_w=zeros)
    env=SimpleNamespace(common_step_counter=0,episode_length_buf=torch.ones(2,dtype=torch.long),
      cfg=SimpleNamespace(decimation=4),step_dt=.02,
      scene={'performance_feet':SimpleNamespace(data=SimpleNamespace(force_history=force))})
    term=object.__new__(FootSupportQuality)
    term.cmd=cmd;term.feet=[0,1];term.tracker=self.tracker
    term.reference_support=torch.ones(100,2,dtype=torch.bool)
    term.last_frame=torch.full((2,),-1,dtype=torch.long);term.age=torch.zeros(2)
    term.cached=torch.zeros(2);term.last_step=-1
    term.offsets=torch.zeros(2,1,3);term.radii=torch.ones(2,1)*.005;term.com=torch.zeros(2,3)
    term.ref_speed=torch.zeros(100,2)
    def step():
      result=term(env);env.common_step_counter+=1;env.episode_length_buf+=1;cmd.frame_ids+=1
      return result
    term(env);age=term.age.clone();term(env);torch.testing.assert_close(term.age,age)
    for _ in range(12):step()
    force.zero_()
    for _ in range(15):cost=step()
    self.assertTrue((cost>0).all())
    self.assertFalse(term.tracker.contact.any())
    cmd.frame_ids[0]=70
    cost=step()
    self.assertEqual(cost[0].item(),0)
    self.assertGreater(cost[1].item(),0)


class FullDisturbanceScratchTests(TestCase):
  def test_full_stage_first_reset_and_original_ppo(self):
    cfg=load_env_cfg(TASK);old=load_env_cfg('Cyclo-Mimic-K1-Rev1-Koryo-ContactV2-Max')
    cmd=Mock()
    with patch.object(KoryoCurriculumCommandCfg,'build',return_value=cmd):
      cfg.commands['reference_trajectory'].build(Mock())
    cmd.load_curriculum_state_dict.assert_called_once_with(initial_state(8))
    for key in old.events:
      self.assertEqual(cfg.events[key],old.events[key])
    for term in cfg.observations['actor'].terms.values():
      if isinstance(term.noise,KoryoObservationNoise):self.assertEqual(term.noise.curriculum_scale,1)
    self.assertEqual(cfg.rewards['action_rate_l2'].weight,-.1)
    self.assertNotIn('performance_stance_foot_motion',cfg.rewards)
    self.assertNotIn('performance_contact_transitions',cfg.rewards)
    self.assertIn('support_quality',cfg.rewards)
    self.assertEqual(cfg.observations,old.observations)
    self.assertEqual(cfg.actions,old.actions)
    self.assertEqual(cfg.terminations,old.terminations)
    rl=load_rl_cfg(TASK)
    self.assertFalse(rl.resume)
    self.assertEqual((rl.algorithm.learning_rate,rl.algorithm.schedule,rl.algorithm.entropy_coef),(.001,'adaptive',.005))
    self.assertEqual(rl.actor.distribution_cfg['init_std'],1.)
    self.assertIs(load_runner_cls(TASK),KoryoSupportMaxRunner)

  def test_reject_old_checkpoints_before_loading_networks(self):
    runner=object.__new__(KoryoSupportMaxRunner)
    old={'koryo_performance':{'version':2,'fixed_disturbance_stage':8}}
    with patch('torch.load',return_value={'infos':old}),patch.object(MjlabOnPolicyRunner,'load') as load:
      with self.assertRaisesRegex(ValueError,'starts from scratch'):runner.load('old.pt')
      load.assert_not_called()

  def test_resume_and_save_preserve_full_stage_and_reward_ramp(self):
    runner=object.__new__(KoryoSupportMaxRunner)
    state={'version':3,'objective':'support_contact_quality_v3','fixed_disturbance_stage':8}
    infos={'koryo_performance':state,'contact_training_steps':48024,'koryo_curriculum':initial_state(8),
           'performance_start_sampler':torch.arange(3.)}
    cmd=SimpleNamespace(cfg=SimpleNamespace(trajectory_file=None),failure_score_by_bin=torch.zeros(3),device='cpu',
      frame_ids=torch.ones(2,dtype=torch.long),_update_command=Mock(),load_curriculum_state_dict=Mock(),
      curriculum_state_dict=lambda:initial_state(8))
    env=SimpleNamespace(common_step_counter=0,command_manager=SimpleNamespace(get_term=lambda _:cmd))
    runner.env=SimpleNamespace(unwrapped=env,reset=Mock())
    runner.alg=SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{'lr':.0004}]),learning_rate=.001)
    def base_load(*args):
      self.assertIsNone(args[1]);runner.current_learning_iteration=2000;return infos
    with patch('torch.load',return_value={'infos':infos}),patch.object(MjlabOnPolicyRunner,'load',side_effect=base_load):
      runner.load('same_task.pt')
    self.assertEqual(env.common_step_counter,48024)
    self.assertEqual(runner.current_learning_iteration,2001)
    self.assertEqual(runner.alg.learning_rate,.0004)
    cmd.load_curriculum_state_dict.assert_called_once_with(initial_state(8))
    with patch.object(MjlabOnPolicyRunner,'save') as save:runner.save('saved.pt')
    saved=save.call_args.args[1]
    self.assertEqual(saved['contact_training_steps'],48024)
    self.assertEqual(saved['koryo_performance']['version'],3)
    self.assertEqual(saved['koryo_performance']['fixed_noise_scale'],1)
    self.assertEqual(saved['koryo_performance']['push_scale'],1)
    self.assertTrue(runner.resumes_performance_state(saved['koryo_performance']))
    with self.assertRaises(ValueError):object.__new__(KoryoContactMaxRunner).resumes_performance_state(state)
