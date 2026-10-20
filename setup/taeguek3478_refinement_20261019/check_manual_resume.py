"""CPU-only restoration test: no simulator, environment steps, or training."""
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace as NS
import json
import torch
from tensordict import TensorDict
from manual_sequence import choose
from train_manual import ManualRunner,agent_cfg,env_cfg
path,completed=choose(3)
assert completed>0
saved=torch.load(path,map_location='cpu',weights_only=False)
a=saved['actor_state_dict']['mlp.0.weight'].shape[1];c=saved['critic_state_dict']['mlp.0.weight'].shape[1]
obs=TensorDict({'actor':torch.zeros(2,a),'critic':torch.zeros(2,c)},batch_size=[2])
env=NS(num_envs=2,num_actions=23,device='cpu',get_observations=lambda:obs,
    unwrapped=NS(common_step_counter=0),cfg=env_cfg(3))
runner=ManualRunner(env,asdict(agent_cfg()),None,device='cpu')
runner.load(str(path),map_location='cpu')
assert runner.current_learning_iteration==completed
assert env.unwrapped.common_step_counter==completed*24
for model,key in [(runner.alg.actor,'actor_state_dict'),(runner.alg.critic,'critic_state_dict')]:
    assert all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in saved[key].items())
actual=runner.alg.optimizer.state_dict();expected=saved['optimizer_state_dict']
assert actual['param_groups']==expected['param_groups']
for k,values in expected['state'].items():
    for name,v in values.items():
        got=actual['state'][k][name]
        assert torch.equal(got.cpu(),v) if isinstance(v,torch.Tensor) else got==v
result=dict(checkpoint=str(path),completed_updates=completed,remaining=10000-completed,
    next_iteration=runner.current_learning_iteration,actor_critic_exact=True,optimizer_exact=True,
    no_training_executed=True)
(Path(__file__).resolve().parent/'manual_resume_check.json').write_text(json.dumps(result,indent=2))
print('MANUAL_RESUME_CHECK',json.dumps(result),flush=True)
