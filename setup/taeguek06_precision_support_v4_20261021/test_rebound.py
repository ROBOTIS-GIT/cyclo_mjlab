import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from rebound import rebound_cost,impact_tail

class ReboundTests(unittest.TestCase):
    def test_normal_support_is_free(self):
        c=rebound_cost(torch.zeros(1,2),torch.zeros(1,2,3),torch.ones(1,2,dtype=torch.bool))
        self.assertTrue(all(x.sum()==0 for x in c))
    def test_rise_is_penalized_before_height_changes(self):
        c=rebound_cost(torch.tensor([[.25,1.]]),torch.zeros(1,2,3),torch.tensor([[True,False]]))
        self.assertGreater(c[0][0,0],0);self.assertEqual(c[0][0,1],0)
        self.assertEqual(c[1].sum(),0)
    def test_landing_and_intentional_flight(self):
        c=rebound_cost(torch.tensor([[-.8,-.8]]),torch.ones(1,2,3),torch.tensor([[True,False]]))
        self.assertGreater(c[1][0,0],0)
        self.assertTrue(all(x[0,1]==0 for x in c))
    def test_grf_tail_preserves_initial_and_restores_final(self):
        raw=torch.tensor([0.,.1,1.,5.,10.])
        torch.testing.assert_close(impact_tail(raw,0),torch.zeros_like(raw))
        torch.testing.assert_close(torch.tanh(raw)+impact_tail(raw,48000),raw)
        self.assertGreater(impact_tail(raw,48000)[-1],impact_tail(raw,48000)[-2])

if __name__=='__main__':unittest.main()
