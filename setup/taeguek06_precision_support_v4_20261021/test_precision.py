import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from precision import filtered_gravity,path_cost,precision_cost

class PrecisionTests(unittest.TestCase):
    def test_airborne_reference_not_mixed_into_support(self):
        g=torch.tensor([[[0.,0.,1.]],[[0.,0.,1.]],[[1.,0.,0.]]]);mask=torch.tensor([[True],[True],[False]])
        torch.testing.assert_close(filtered_gravity(g,mask),g)
    def test_reference_filter_reduces_alternating_jitter(self):
        g=torch.zeros(21,2,3);g[...,2]=1;g[::2,:,0]=.04;g[1::2,:,0]=-.04
        g/=g.norm(dim=-1,keepdim=True)
        sm=filtered_gravity(g,torch.ones(21,2,dtype=torch.bool))
        self.assertLess(float(sm[2:-2,:,0].abs().max()),.001)
        torch.testing.assert_close(sm.norm(dim=-1),torch.ones(21,2))
    def test_path_cost_preserves_global_translation_invariance(self):
        p=torch.tensor([[.14,0.,0.]]);r=torch.zeros_like(p);shift=torch.tensor([[25.,-12.,1.]])
        torch.testing.assert_close(path_cost(p,r),path_cost(p+shift,r+shift))
        self.assertGreater(path_cost(p,r),path_cost(p/2,r))
        self.assertEqual(path_cost(p/10,r),0)
    def test_small_angle_precision_and_swing_exclusion(self):
        e=torch.tensor([[[.025,0.,0.],[.025,0.,0.]]]);v=torch.zeros_like(e)
        cost=precision_cost(e,v,torch.tensor([[True,False]]))
        self.assertGreater(cost[0,0],0);self.assertEqual(cost[0,1],0)

if __name__=='__main__':unittest.main()
