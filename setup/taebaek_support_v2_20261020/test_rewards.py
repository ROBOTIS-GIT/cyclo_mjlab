import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from refinement import foot_costs, phase_drift, env_cfg, bounded_support_cost, support_ramp, guard_update, validate_inputs, agent_cfg


class Rewards(unittest.TestCase):
    def costs(self, drift=0., omega=0., acc=0., height=0., ref_height=0., vz=0., support=True):
        return foot_costs(torch.tensor([[[drift, 0.]]]), torch.tensor([[[omega, 0., 0.]]]),
                          torch.tensor([[[acc, 0., 0.]]]), torch.tensor([[height]]),
                          torch.tensor([[ref_height]]), torch.tensor([[vz]]),
                          torch.tensor([[support]]))

    def test_matching_reference_is_free(self):
        self.assertEqual(sum(v.item() for v in self.costs().values()), 0)

    def test_flight_not_pinned(self):
        self.assertEqual(sum(v.item() for v in self.costs(.2, 4., 100., 1., 0., 3., False).values()), 0)

    def test_drift_large_failure_keeps_gradient(self):
        self.assertLess(self.costs(.05)['drift'], self.costs(.15)['drift'])
        self.assertLess(self.costs(.15)['drift'], self.costs(.30)['drift'])
        self.assertEqual(self.costs(.019)['drift'], 0)

    def test_reference_tilt_clearance_allowed(self):
        self.assertEqual(self.costs(height=.039, ref_height=.03)['clearance'], 0)
        self.assertGreater(self.costs(height=.045, ref_height=.03)['clearance'], 0)

    def test_signed_velocity_error_and_reversal(self):
        self.assertGreater(self.costs(omega=2.)['rocking'], self.costs(omega=.5)['rocking'])
        self.assertEqual(self.costs(omega=-2.)['rocking'], self.costs(omega=2.)['rocking'])
        self.assertGreater(self.costs(acc=80.)['rocking_acc'], self.costs(acc=20.)['rocking_acc'])

    def test_phase_offset_not_reanchored_on_drift(self):
        error = torch.tensor([[[.1, .2]]])
        yes = torch.ones(1, 1, dtype=torch.bool)
        no = ~yes
        drift, anchor = phase_drift(error, torch.zeros_like(error), yes, no, torch.tensor([True]))
        self.assertEqual(drift.abs().sum(), 0)
        drift, same = phase_drift(error + .2, anchor, yes, yes, torch.tensor([True]))
        torch.testing.assert_close(same, anchor)
        torch.testing.assert_close(drift, torch.full_like(drift, .2))
        drift, _ = phase_drift(error + .2, anchor, yes, yes, torch.tensor([False]))
        self.assertEqual(drift.abs().sum(), 0)

    def test_whole_clip_and_original_rewards_preserved(self):
        from source.tasks.mimic.config.k1_rev1.base_env_cfg import k1_rev1_mimic_env_cfg
        base = k1_rev1_mimic_env_cfg()
        cfg = env_cfg()
        for name, reward in base.rewards.items():
            self.assertEqual(reward, cfg.rewards[name])
        self.assertGreater(cfg.episode_length_s, 47.18)
        self.assertEqual(cfg.scene.num_envs, 4096)
        self.assertEqual(cfg.commands['reference_trajectory'].first_frame_env_fraction, .2)

    def test_penalty_budget_monotone_and_bounded(self):
        raw=torch.tensor([0.,.1,.5,1.,10.,100.])
        cost=bounded_support_cost(raw)
        self.assertEqual(cost[0],0)
        self.assertTrue(torch.all(torch.diff(cost)>=0))
        self.assertTrue(torch.all(cost<=.600001))
        self.assertLess(cost[3],.6)

    def test_slow_ramp(self):
        self.assertAlmostEqual(support_ramp(0),.1)
        self.assertAlmostEqual(support_ramp(24000),.28)
        self.assertAlmostEqual(support_ramp(120000),1.)
        self.assertAlmostEqual(support_ramp(240000),1.)

    def test_guard_warmup_duplicate_and_consecutive_checks(self):
        state=guard_update({},0,100.)
        self.assertFalse(state['halt'])
        state=guard_update(state,250,1800.)
        state=guard_update(state,500,800.)
        self.assertEqual(state['bad_checks'],1)
        same=guard_update(state,500,100.)
        self.assertEqual(state,same)
        state=guard_update(state,750,1500.)
        self.assertEqual(state['bad_checks'],0)
        state=guard_update(state,1000,500.)
        state=guard_update(state,1250,500.)
        self.assertTrue(state['halt'])

    def test_guard_catches_failure_without_good_initial_window(self):
        state=guard_update({},500,500.)
        state=guard_update(state,750,400.)
        self.assertTrue(state['halt'])

    def test_original_source_and_learning_settings(self):
        validate_inputs()
        self.assertEqual(agent_cfg().algorithm.learning_rate,2e-5)
        self.assertEqual(agent_cfg().max_iterations,10000)

if __name__ == '__main__':
    unittest.main(verbosity=2)
