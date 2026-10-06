"""Landing retiming and substep force-penalty regression checks."""
import unittest
from types import SimpleNamespace as S
import numpy as np
import torch
from scripts.tools.motion.retime_umr_soft_landing import retime
from source.tasks.mimic.mdp.impact_rewards import pelvis_impact


class SoftLanding(unittest.TestCase):
  def test_retiming_preserves_path_and_endpoints(self):
    t = np.arange(501)/50
    root = np.column_stack([t, t*2, t*0])
    joints = np.tile(t[:, None], (1,23))
    quat = np.tile([1.,0,0,0], (len(t),1))
    rows, _ = retime(root, quat, joints, 50)
    np.testing.assert_allclose(np.diff(rows[:,0]), .02)
    np.testing.assert_allclose(rows[:,2], rows[:,1]*2)
    np.testing.assert_allclose(rows[:,8], rows[:,1])
    np.testing.assert_allclose(rows[[0,-1],1], [0,10])
    self.assertTrue(np.all(np.diff(rows[:,1]) >= 0))
    active=(rows[:-1,1]>4.9)&(rows[:-1,1]<5.2)
    self.assertLess(np.max(np.diff(rows[:,1])[active]/.02),.401)

  def test_support_allowed_and_short_impact_detected(self):
    force=torch.zeros(3,1,4,3)
    force[0,0,:,2]=250  # Sustained support is allowed.
    force[1,0,2,2]=900  # Peak occurred before the final physics substep.
    force[2,0,1,2]=9000 # Bounded penalty for extreme contacts.
    command=S(metrics={})
    env=S(scene={'sensor':S(data=S(force_history=force))},
          command_manager=S(get_term=lambda _:command))
    result=pelvis_impact(env,'sensor',300)
    torch.testing.assert_close(result,torch.tensor([0.,4.,25.]))


if __name__=='__main__': unittest.main()
