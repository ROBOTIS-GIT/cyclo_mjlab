"""Regression checks for time-limit censoring and checkpoint curriculum state."""
from types import SimpleNamespace as S
from unittest.mock import patch, Mock
import unittest
import torch
from source.tasks.mimic.mdp.stand_up_curriculum import StandUpCurriculumCommand as Command
from source.tasks.mimic.curriculum_runner import StandUpCurriculumRunner as Runner
from mjlab.rl import MjlabOnPolicyRunner


def command():
  c = object.__new__(Command)
  c.cfg = S(first_frame_env_fraction=1., curriculum_min_attempts=1024, curriculum_success_threshold=.8)
  c.trial_stage = torch.tensor([0]); c.difficulty_stage = 0
  c.frame_ids = torch.tensor([499])
  c.reference = S(num_frames=499, body_position_w=torch.tensor([[[0., 0., .8]]]))
  c.reference_anchor_body_id = 0
  c.robot_anchor_body_id = 0
  c.robot = S(data=S(body_link_pos_w=torch.tensor([[[0., 0., .8]]]), body_link_quat_w=torch.tensor([[[1., 0., 0., 0.]]])))
  c._env = S(num_envs=1, scene=S(env_origins=torch.zeros(1,3)), termination_manager=S(terminated=torch.tensor([False])))
  c.attempts = c.successes = c.completed_successes = c.failed_attempts = c.interrupted_attempts = 0
  c.last_success_rate = 0.
  return c


class Outcomes(unittest.TestCase):
  def test_timeout_does_not_reduce_success_rate(self):
    c=command();c.cfg.curriculum_min_attempts=3
    for frame in [499, 499, 3, 499]:
      c.frame_ids[:]=frame;c._record_outcomes(torch.tensor([0]))
    self.assertEqual(c.last_success_rate,1.)
    self.assertEqual(c.difficulty_stage,1)
    self.assertEqual(c.interrupted_attempts,1)
    self.assertEqual(c.failed_attempts,0)

  def test_real_failure_counts_even_at_timeout(self):
    c=command();c.frame_ids[:]=3;c._env.termination_manager.terminated[:]=True
    c._record_outcomes(torch.tensor([0]))
    self.assertEqual((c.attempts,c.failed_attempts,c.interrupted_attempts),(1,1,0))

  def test_invalid_final_pose_is_failure(self):
    c=command();c.robot.data.body_link_pos_w[:,:,2]=.1
    c._record_outcomes(torch.tensor([0]))
    self.assertEqual((c.failed_attempts,c.successes),(1,0))

  def test_checkpoint_roundtrip_and_actor_only(self):
    c=command();c.difficulty_stage=2;c.attempts=100;c.successes=91
    runner=object.__new__(Runner);runner.env=S(unwrapped=S(command_manager=S(get_term=lambda _:c)),reset=Mock())
    with patch.object(MjlabOnPolicyRunner,'save') as save:
      runner.save('unused.pt');infos=save.call_args.args[1]
    c.difficulty_stage=0
    with patch.object(MjlabOnPolicyRunner,'load',return_value=infos):
      runner.load('unused.pt',load_cfg={'actor':True})
      self.assertEqual(c.difficulty_stage,0)
      runner.load('unused.pt')
    self.assertEqual((c.difficulty_stage,c.attempts,c.successes),(2,100,91))
    self.assertEqual(int(c.trial_stage[0]),-1)
    runner.env.reset.assert_called_once()


if __name__=='__main__': unittest.main()
