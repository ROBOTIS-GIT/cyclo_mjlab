"""Bounded Rev.2 integration test; no production training is launched."""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import json,sys,hashlib
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import numpy as np
import torch,yaml
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from source.utils import export_sim2real_cfg
from source.tasks.mimic.config.k1_rev2.taeguek01_env_cfg import k1_rev2_taeguek01_env_cfg,k1_rev2_taeguek01_ppo_cfg


def main():
    inputs=json.loads((HERE/'inputs.json').read_text())
    for rel,h in inputs['hashes'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
    configure_torch_backends();torch.set_num_threads(4)
    cfg=k1_rev2_taeguek01_env_cfg();cfg.scene.num_envs=4096
    log=ROOT/'logs/rsl_rl/k1_rev2_taeguek01_smoke'/datetime.now().strftime('%Y%m%d_%H%M%S')
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    model=env.sim.mj_model
    robot=env.scene['robot'];command=env.command_manager.get_term('reference_trajectory')
    assert len(robot.joint_names)==23 and abs(model.body_mass.sum()-37.3424451)<1e-5
    assert abs(env.step_dt-.02)<1e-9 and cfg.decimation==4 and float(command.reference.fps)==50.
    # Runtime loads this Rev.2 reference, not Rev.1 body-position targets.
    assert '/K1_rev2/' in command.cfg.trajectory_file
    obs,_=env.reset();assert obs['actor'].shape==(4096,124)
    assert torch.isfinite(obs['actor']).all()
    export_sim2real_cfg(env,log);interface=yaml.safe_load((log/'params/sim2real.yaml').read_text())
    assert interface['step_dt']==.02 and len(interface['policy_joints'])==23
    wrapped=RslRlVecEnvWrapper(env,clip_actions=10.)
    agent=k1_rev2_taeguek01_ppo_cfg();agent.logger='tensorboard';agent.upload_model=False
    runner=MjlabOnPolicyRunner(wrapped,asdict(agent),str(log),'cuda:0')
    runner.learn(num_learning_iterations=2,init_at_random_ep_len=True)
    ckpt=log/'model_1.pt';saved=torch.load(ckpt,map_location='cpu',weights_only=False)
    assert saved['iter']==1 and saved['optimizer_state_dict']['state']
    for key in ['actor_state_dict','critic_state_dict']:
        assert all(torch.isfinite(v).all() for v in saved[key].values())
    step=min(float(v['step']) for v in saved['optimizer_state_dict']['state'].values());assert step>=40
    runner.export_policy_to_jit(str(log/'exported'),'policy.pt')
    runner.export_policy_to_onnx(str(log/'exported'),'policy.onnx')
    policy=runner.get_inference_policy('cuda:0');obs=wrapped.get_observations()
    with torch.inference_mode():
        for _ in range(64):
            act=policy(obs);assert torch.isfinite(act).all()
            obs,rewards,dones,extras=wrapped.step(act);assert torch.isfinite(rewards).all() and torch.isfinite(obs['actor']).all()
    result={'status':'passed','num_envs':4096,'ppo_updates':2,'optimizer_step':step,'finite_rollout_steps':64,'mass_kg':float(model.body_mass.sum()),'actor_observation_size':124,'actions':23,'policy_hz':50,'physics_hz':200,'checkpoint':str(ckpt),'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),'exported_jit_onnx':True,'cuda_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,'production_started':False}
    (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print('VALIDATION_OK',json.dumps(result),flush=True)
    env.close()

if __name__=='__main__':main()
