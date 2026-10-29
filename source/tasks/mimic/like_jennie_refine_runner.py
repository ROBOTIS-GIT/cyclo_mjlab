"""Warm-start refinement from a frozen policy with a fresh PPO optimizer."""
import hashlib
from pathlib import Path
import torch
from mjlab.rl import MjlabOnPolicyRunner


class LikeJennieRefineRunner(MjlabOnPolicyRunner):
    def load(self,path,load_cfg=None,strict=True,map_location=None):
        if load_cfg is not None:
            return super().load(path,load_cfg,strict,map_location)
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        for key in ('actor_state_dict','critic_state_dict'):
            if key not in checkpoint or not all(torch.isfinite(v).all() for v in checkpoint[key].values()):
                raise ValueError(f'Invalid checkpoint {key}')
        result=super().load(path,{'actor':True,'critic':True,'optimizer':False,'iteration':False,'rnd':False},strict,map_location or self.device)
        for model,key in ((self.alg.actor,'actor_state_dict'),(self.alg.critic,'critic_state_dict')):
            if not all(torch.equal(v.cpu(),checkpoint[key][k]) for k,v in model.state_dict().items()):
                raise RuntimeError(f'{key} warm start mismatch')
        if self.alg.optimizer.state:raise RuntimeError('Expected fresh optimizer')
        self.current_learning_iteration=0
        self.env.unwrapped.common_step_counter=0
        print('LIKE_JENNIE_WARMSTART',{'source':str(Path(path).resolve()),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'source_iteration':checkpoint['iter'],'optimizer':'fresh'},flush=True)
        return result
