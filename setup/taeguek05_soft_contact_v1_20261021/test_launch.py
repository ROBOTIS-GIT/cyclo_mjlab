"""CLI target accounting and checkpoint resume regression checks; no GPU training."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import launch
from refinement import MARKER, RUN_NAME, revision, validate_resume


class LaunchBehavior(unittest.TestCase):
    def run_launch(self, completed, *args):
        output = io.StringIO()
        with patch('sys.argv', ['launch.py', '0', '--num-envs', '4096', *args]), \
                patch.object(launch, 'validate_inputs'), \
                patch.object(launch, 'choose', return_value=(Path('/checkpoint.pt'), completed)), \
                contextlib.redirect_stdout(output):
            launch.main()
        return output.getvalue()

    def test_requested_20000_and_default(self):
        self.assertIn('target=20000; completed=0; remaining=20000',
                      self.run_launch(0, '--iter', '20000', '--dry-run'))
        self.assertIn('target=10000; completed=0; remaining=10000',
                      self.run_launch(0, '--dry-run'))

    def test_resume_only_remaining_updates(self):
        self.assertIn('target=20000; completed=15000; remaining=5000',
                      self.run_launch(15000, '--iter', '20000', '--dry-run'))

    def test_completed_or_lower_target_never_launches(self):
        with patch.object(launch.os, 'execv') as execute:
            for completed in (20000, 21000):
                self.assertIn('remaining=0', self.run_launch(completed, '--iter', '20000'))
            execute.assert_not_called()

    def test_nonpositive_target_rejected(self):
        for target in ('0', '-1'):
            with self.subTest(target=target), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    self.run_launch(0, '--iter', target, '--dry-run')
                self.assertEqual(caught.exception.code, 2)

    def test_remaining_updates_forwarded_to_trainer(self):
        with patch.object(launch.os, 'fdopen') as fdopen, \
                patch.object(launch.os, 'open'), \
                patch.object(launch.fcntl, 'flock'), \
                patch.object(launch.subprocess, 'check_output', side_effect=['GPU-test', '']), \
                patch.object(launch.os, 'chdir'), \
                patch.dict(launch.os.environ), \
                patch.object(launch.os, 'set_inheritable'), \
                patch.object(launch.os, 'execv') as execute:
            fdopen.return_value.fileno.return_value = 123
            self.run_launch(15000, '--iter', '20000')
            command = execute.call_args.args[1]
            self.assertEqual(command[command.index('--agent.max-iterations') + 1], '5000')
            self.assertEqual(command[command.index('--checkpoint') + 1], '/checkpoint.pt')

    def test_checkpoint_above_default_resumes_and_integrity_still_checked(self):
        completed = 15000
        data = {
            'iter': completed - 1,
            'infos': {MARKER: {'revision': revision(), 'num_envs': 4096},
                      'env_state': {'common_step_counter': completed * 24}},
            'optimizer_state_dict': {'state': {0: {'step': torch.tensor(1.)}}},
            'actor_state_dict': {'weight': torch.ones(1)},
            'critic_state_dict': {'weight': torch.ones(1)},
        }
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / f'2026-10-20_{RUN_NAME}_test'
            run.mkdir()
            checkpoint = run / f'model_{completed - 1}.pt'
            torch.save(data, checkpoint)
            self.assertEqual(launch.choose(4096, Path(directory)), (checkpoint, completed))
        data['infos']['env_state']['common_step_counter'] -= 24
        with self.assertRaisesRegex(ValueError, 'counter mismatch'):
            validate_resume(data, 4096)


if __name__ == '__main__':
    unittest.main()
