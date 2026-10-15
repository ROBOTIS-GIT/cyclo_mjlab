"""Prevent legitimate support/contact redistribution from being penalized."""
import unittest
from types import SimpleNamespace as S
import torch
from source.tasks.mimic.mdp.landing_contact_rewards import knee_impact, toe_load_change


def make_env(**forces):
  command=S(metrics={})
  return S(scene={k:S(data=S(force_history=v)) for k,v in forces.items()},
           command_manager=S(get_term=lambda _:command))


class LandingContacts(unittest.TestCase):
  def test_knee_support_allowed_and_substep_impact_detected(self):
    f=torch.zeros(2,2,4,3);f[0,:,:,2]=300;f[1,1,1,2]=1050
    torch.testing.assert_close(knee_impact(make_env(knees=f),'knees'),torch.tensor([0.,4.]))

  def test_transfer_between_toe_spheres_is_not_chatter(self):
    f=torch.zeros(1,5,5,3)
    for i in range(5):f[0,i,i,2]=-175
    env=make_env(left=f,right=f)
    torch.testing.assert_close(toe_load_change(env,('left','right')),torch.zeros(1))

  def test_toe_load_drop_across_policy_boundary_is_detected(self):
    f=torch.zeros(1,5,5,3);f[0,0,4,2]=-400
    env=make_env(toe=f)
    self.assertGreater(float(toe_load_change(env,('toe',))),0)

  def test_gradual_toe_unloading_is_allowed(self):
    f=torch.zeros(1,5,5,3);f[0,0,:,2]=torch.tensor([-200.,-160.,-120.,-80.,-40.])
    torch.testing.assert_close(toe_load_change(make_env(toe=f),('toe',)),torch.zeros(1))

  def test_isolated_recontact_gets_stronger_peak_penalty(self):
    # One 400 N touchdown across the policy boundary, then steady support.
    f=torch.zeros(1,5,5,3);f[0,0,:4,2]=-400
    old=toe_load_change(make_env(toe=f),('toe',))
    new=toe_load_change(make_env(toe=f),('toe',),peak_fraction=0.5)
    torch.testing.assert_close(old,torch.tensor([1.]))
    torch.testing.assert_close(new,torch.tensor([2.5]))
    # The new task also doubles the weight: 5x the isolated-spike cost.
    torch.testing.assert_close(0.2*new,5*0.1*old)

  def test_peak_penalty_allows_rolling_and_smooth_unloading(self):
    f=torch.zeros(1,5,5,3)
    for i in range(5):f[0,i,i,2]=-(200-40*i)
    torch.testing.assert_close(
      toe_load_change(make_env(toe=f),('toe',),peak_fraction=0.5),
      torch.zeros(1),
    )

  def test_both_feet_are_penalized_independently(self):
    f=torch.zeros(1,5,5,3);f[0,0,4,2]=-400
    env=make_env(left=f,right=torch.zeros_like(f))
    torch.testing.assert_close(
      toe_load_change(env,('left','right'),peak_fraction=0.5),
      torch.tensor([2.5]),
    )
    metrics=env.command_manager.get_term('reference_trajectory').metrics
    torch.testing.assert_close(metrics['left_load_change_n'],torch.tensor([400.]))
    torch.testing.assert_close(metrics['right_load_change_n'],torch.zeros(1))


if __name__=='__main__':unittest.main()
