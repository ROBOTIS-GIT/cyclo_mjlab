"""Restored-DR contract and checkpoint migration tests; no simulator or PPO run."""
import unittest
from copy import deepcopy
from types import SimpleNamespace as S
from unittest.mock import patch, Mock
import torch
from mjlab.rl import MjlabOnPolicyRunner
from source.tasks.mimic.config.k1_rev1.umr_stand_up_soft_entry_env_cfg import soft_entry_env_cfg
from source.tasks.mimic.config.k1_rev1.umr_stand_up_soft_entry_dr_env_cfg import soft_entry_dr_env_cfg, RESTORED_EVENTS
from source.tasks.mimic.config.k1_rev1.umr_stand_up_soft_landing_env_cfg import k1_rev1_umr_stand_up_soft_landing_env_cfg
from source.tasks.mimic.curriculum_runner import StandUpCurriculumRunner
from source.tasks.mimic.soft_entry_dr_runner import SoftEntryDRRunner
from source.tasks.mimic.mdp.friction_events import randomize_support_friction


class SoftEntryDRTests(unittest.TestCase):
  def test_restore_original_events_without_changing_soft_entry(self):
    original = k1_rev1_umr_stand_up_soft_landing_env_cfg()
    old = soft_entry_env_cfg(); new = soft_entry_dr_env_cfg()
    for name in RESTORED_EVENTS:
      self.assertEqual(new.events[name], original.events[name])
    self.assertTrue(new.commands['reference_trajectory'].resample_material_on_reset)
    self.assertFalse(old.commands['reference_trajectory'].resample_material_on_reset)
    self.assertNotIn('torso_com_offset_noise', old.events)
    self.assertEqual(new.actions, old.actions)
    self.assertEqual(new.rewards, old.rewards)
    self.assertEqual(new.observations, old.observations)
    self.assertEqual(new.scene.entities['robot'].articulation, old.scene.entities['robot'].articulation)
    self.assertEqual(new.events['push_robot'], old.events['push_robot'])
    self.assertEqual(set(new.events)-set(old.events), {'joint_home_offset_noise', 'torso_com_offset_noise'})
    expected = deepcopy(old.commands['reference_trajectory']); expected.resample_material_on_reset = True
    self.assertEqual(new.commands['reference_trajectory'], expected)
    self.assertEqual(soft_entry_dr_env_cfg(play=True).events, soft_entry_env_cfg(play=True).events)

  def test_friction_varies_both_contact_sides_at_each_stage(self):
    cfg = soft_entry_dr_env_cfg(); params = dict(cfg.events['physics_material'].params)
    params['asset_cfg'] = S(name='robot', geom_ids=slice(None))
    params['terrain_cfg'] = S(name='terrain', geom_ids=slice(None))
    c = S(difficulty_stage=0)
    model = S(geom_friction=torch.ones(256, 3, 3))
    env = S(num_envs=256, device='cpu', common_step_counter=0, sim=S(model=model),
      command_manager=S(get_term=lambda _: c),
      scene={'robot': S(indexing=S(geom_ids=torch.tensor([0,1]))),
             'terrain': S(indexing=S(geom_ids=torch.tensor([2])))})
    for stage, low in enumerate((.8,.4,.3,.2)):
      c.difficulty_stage=stage;torch.manual_seed(42)
      randomize_support_friction(env,None,**params)
      mu=model.geom_friction[:,:,0]
      self.assertGreaterEqual(float(mu.min()),low);self.assertLessEqual(float(mu.max()),1.2)
      self.assertGreater(float(mu[:,0].std()),.02)
      torch.testing.assert_close(mu[:,0],mu[:,1]);torch.testing.assert_close(mu[:,0],mu[:,2])
      if stage:self.assertTrue(bool((mu[:,0]<.8).any()))

  def runner(self):
    runner=object.__new__(SoftEntryDRRunner);runner.current_learning_iteration=9999
    command=S(load_curriculum_state_dict=Mock())
    env=S(common_step_counter=240000,command_manager=S(get_term=lambda _:command))
    runner.env=S(unwrapped=env,reset=Mock());runner._refresh=Mock()
    runner.alg=S(learning_rate=0.,optimizer=S(param_groups=[{'lr':5e-5}]))
    runner.entry_spec=Mock(return_value={'version':3,'motion_sha256':'same','restored_randomization':{'profile':'test'}})
    return runner,command

  def test_v2_warm_start_does_not_resume_finished_training(self):
    r,c=self.runner()
    checkpoint={'infos':{'ai_sapiens_entry_version':2,'ai_sapiens_entry_spec':{'motion_sha256':'same'}}}
    with patch('torch.load',return_value=checkpoint), patch.object(MjlabOnPolicyRunner,'load',return_value={}) as load:
      r.load('v2.pt')
    self.assertEqual(load.call_args.args[2],{'actor':True,'critic':True})
    self.assertEqual(r.current_learning_iteration,0);self.assertEqual(r.env.unwrapped.common_step_counter,0)
    self.assertEqual(c.load_curriculum_state_dict.call_args.args[0]['difficulty_stage'],0)
    r.env.reset.assert_called_once();r._refresh.assert_called_once()

  def test_dr_resume_and_profile_mismatch(self):
    r,c=self.runner();saved=r.entry_spec()
    checkpoint={'infos':{'ai_sapiens_entry_version':3,'ai_sapiens_entry_spec':saved}}
    def restore(*args):r.current_learning_iteration=249;return {}
    with patch('torch.load',return_value=checkpoint),patch.object(StandUpCurriculumRunner,'load',side_effect=restore):r.load('dr.pt')
    self.assertEqual(r.current_learning_iteration,250);self.assertEqual(r.alg.learning_rate,5e-5)
    checkpoint=deepcopy(checkpoint);checkpoint['infos']['ai_sapiens_entry_spec']['restored_randomization']={'profile':'changed'}
    with patch('torch.load',return_value=checkpoint),self.assertRaisesRegex(ValueError,'randomization'):r.load('other.pt')

  def test_reject_unrelated_motion_before_loading_weights(self):
    r,c=self.runner()
    checkpoint={'infos':{'ai_sapiens_entry_version':2,'ai_sapiens_entry_spec':{'motion_sha256':'different'}}}
    with patch('torch.load',return_value=checkpoint),patch.object(MjlabOnPolicyRunner,'load') as load,self.assertRaises(ValueError):r.load('other.pt')
    load.assert_not_called()

if __name__=='__main__':unittest.main()
