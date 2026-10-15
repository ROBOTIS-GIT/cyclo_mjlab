import unittest
from types import SimpleNamespace
import torch
import source.tasks
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from source.tasks.mimic.mdp.performance_contacts import contact_costs,foot_contact_cost,quiet_stance_mask,stance_foot_support


class PerformanceContactTests(unittest.TestCase):
  def test_steady_support_is_not_penalized(self):
    for force in (0.,175.,350.,600.):
      for cost in contact_costs(torch.full((3,2,5),force)):
        torch.testing.assert_close(cost,torch.zeros(3))

  def test_recontact_cost_distinguishes_one_step_from_chatter(self):
    touchdown=torch.tensor([[[0.,0.,0.,350.,350.]]])
    bouncing=torch.tensor([[[0.,350.,0.,350.,0.]]])
    self.assertEqual(contact_costs(touchdown)[2].item(),0.)
    self.assertGreater(contact_costs(bouncing)[2].item(),0.)
    self.assertGreater(contact_costs(bouncing)[1].item(),contact_costs(touchdown)[1].item())

  def test_short_force_spike_is_not_hidden_by_time_average(self):
    support=torch.full((1,2,5),350.)
    spike=support.clone();spike[0,1,2]=1800.
    self.assertEqual(contact_costs(support)[0].item(),0.)
    self.assertGreater(contact_costs(spike)[0].item(),8.)

  def test_reset_contact_is_excluded_from_reward(self):
    force=torch.zeros(2,2,5,3);force[:,:,:,2]=1800.
    cmd=SimpleNamespace(metrics={})
    env=SimpleNamespace(scene={'performance_feet':SimpleNamespace(data=SimpleNamespace(force_history=force))},
                        episode_length_buf=torch.tensor([1,2]),
                        command_manager=SimpleNamespace(get_term=lambda _:cmd))
    cost=foot_contact_cost(env,'impact')
    self.assertEqual(cost[0].item(),0.)
    self.assertGreater(cost[1].item(),0.)

  def test_kicks_pivots_and_swing_are_excluded_from_quiet_stance(self):
    velocity=torch.zeros(1,4,3);velocity[0,3,0]=0.5
    omega=torch.zeros(1,4,3);omega[0,2,2]=2.
    mask=quiet_stance_mask(velocity,omega,torch.tensor([[0.,.3,0.,0.]]),torch.zeros(4))
    self.assertEqual(mask.tolist(),[[True,False,False,False]])

  def test_hovering_planted_foot_costs_but_lifted_reference_does_not(self):
    force=torch.zeros(1,2,5,3);force[:,1,:,2]=350.
    position=torch.zeros(1,2,3)
    cmd=SimpleNamespace(_performance_feet=([0,1],torch.zeros(2)),
                        reference=SimpleNamespace(body_position_w=position),
                        frame_ids=torch.tensor([0]),body_lin_vel_w=torch.zeros(1,2,3),
                        body_ang_vel_w=torch.zeros(1,2,3),metrics={})
    env=SimpleNamespace(scene={'performance_feet':SimpleNamespace(data=SimpleNamespace(force_history=force))},
                        episode_length_buf=torch.tensor([2]),
                        command_manager=SimpleNamespace(get_term=lambda _:cmd))
    self.assertEqual(stance_foot_support(env).item(),1.)
    position[0,0,2]=.3
    self.assertEqual(stance_foot_support(env).item(),0.)

  def test_performance_keeps_policy_inputs_and_original_task_unchanged(self):
    base=load_env_cfg('Cyclo-Mimic-K1-Rev1-Koryo')
    perf=load_env_cfg('Cyclo-Mimic-K1-Rev1-Koryo-Performance')
    self.assertEqual(list(base.observations['actor'].terms),list(perf.observations['actor'].terms))
    self.assertEqual(base.commands['reference_trajectory'].trajectory_file,perf.commands['reference_trajectory'].trajectory_file)
    self.assertNotIn('performance_foot_impact',base.rewards)
    self.assertEqual(base.rewards['action_rate_l2'].weight,-0.1)
    self.assertEqual(perf.rewards['action_rate_l2'].weight,-0.2)
    self.assertEqual(perf.scene.sensors[-1].history_length,perf.decimation+1)
    rl=load_rl_cfg('Cyclo-Mimic-K1-Rev1-Koryo-Performance')
    self.assertEqual(rl.algorithm.learning_rate,5e-5)
    self.assertEqual(rl.algorithm.schedule,'fixed')


if __name__=='__main__':
  unittest.main()
