"""Mechanics checks and bounded PPO smoke test; not a robustness-success claim."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
from dataclasses import asdict
import sys,json,hashlib,math,copy
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch,numpy as np,mujoco
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from source.tasks.mimic.config.k1_rev1.limbo_robust_env_cfg import k1_rev1_limbo_robust_env_cfg,k1_rev1_limbo_robust_runner_cfg
from source.tasks.mimic.limbo_robust_runner import RobustRunner
from source.tasks.mimic.mdp import limbo_robust as robust
from source.tasks.mimic.mdp import limbo_prepare_obs124 as prep


def verify():
    torch.set_num_threads(2)
    out=ROOT/'logs/limbo_verification/mechanics';out.mkdir(parents=True,exist_ok=True)
    cfg=k1_rev1_limbo_robust_env_cfg();cfg.scene.num_envs=32
    cfg.commands['reference_trajectory'].debug_vis=False
    env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
    agent=k1_rev1_limbo_robust_runner_cfg();wrap=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
    runner=RobustRunner(wrap,asdict(agent),str(out),device='cuda:0')
    checks={}
    try:
        runner.load(str(ROOT/'source/assets/checkpoints/limbo/robust_warm_start.pt'),load_cfg={'actor':True,'critic':True},strict=True,map_location='cuda:0')
        env.reset(seed=42);obs=wrap.get_observations();c=prep.command(env);a=env.action_manager.get_term('joint_pos')
        assert obs['actor'].shape==obs['critic'].shape==(32,124)
        assert tuple(cfg.observations['actor'].terms)==('motion_command','motion_anchor_ori_b','base_ang_vel','joint_pos_rel','joint_vel_rel','last_action')
        assert torch.isfinite(obs['actor']).all();checks['legacy_obs124_preserved']=True
        for name,p in runner.alg.actor.distribution.named_parameters():
            value=p.exp() if name in ('log_std','log_std_param') else p
            assert torch.allclose(value,torch.full_like(value,.08),atol=1e-6)
        checks['warm_start_exploration_std_008']=True
        assert c.pool_roots.shape==(3,256,7) and c.pool_joints.shape==(3,256,23)
        m=env.sim.mj_model;d=mujoco.MjData(m)
        jids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'robot/'+n) for n in c.robot.joint_names]
        feet=[i for i in range(m.ngeom) if 'ankle_roll_link_collision_' in (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '')]
        for level in range(3):
            for ix in range(0,256,17):
                d.qpos[:7]=c.pool_roots[level,ix].cpu().numpy();d.qpos[m.jnt_qposadr[jids]]=c.pool_joints[level,ix].cpu().numpy();mujoco.mj_forward(m,d)
                assert np.min(d.geom_xpos[feet,2]-m.geom_size[feet,0])>=.0009
        checks['reset_pool_foot_projection']=True
        assert c.prep_origin.any()
        assert c.robot.data.joint_vel[c.prep_origin].abs().max()>0
        assert c.robot.data.root_link_vel_w[c.prep_origin].abs().max()>0
        friction=env.sim.model.geom_friction[:,:,0]
        assert friction.std(dim=0).max()>0
        assert not torch.equal(obs['actor'],obs['critic'])
        checks['nonzero_initial_velocities_and_dr_noise']=True
        # Numerical equivalence to the deployed per-physics-step cosine blend.
        c.prep_origin[:]=True;c.entry_duration[:]=1.;c.entry_start_gain[:]=.6
        a.reset(torch.arange(32,device=env.device))
        raw=torch.full((32,23),.2,device=env.device);env.action_manager.process_action(raw)
        for t,alpha in [(0.,0.),(.5,.5),(1.,1.),(1.2,1.)]:
            a.elapsed[:]=t;a.apply_actions()
            expected=a.start+alpha*(a._processed_actions-a.start)
            torch.testing.assert_close(a.applied_target,expected)
            gain=.6+alpha*.4
            torch.testing.assert_close(env.sim.model.actuator_gainprm[:,a.actuator_ids,0],(a.kp*gain).expand(32,-1).to(torch.float32))
        current=env.observation_manager.compute(update_history=True)['actor']
        torch.testing.assert_close(current[:,-23:],raw)
        checks['cosine_target_gain_and_raw_last_action_match_runtime']=True
        # Exactly one kick, with the commanded delta, in the initial window.
        env.reset(seed=123);c=prep.command(env);c.prep_origin[:]=True;c.push_done[:]=False;c.push_at[:]=.2;c.push_delta[:]=0;c.push_delta[:,0]=.05
        env.episode_length_buf[:]=15
        before=c.robot.data.root_link_vel_w.clone();robust.entry_push(env);env.sim.forward()
        after=c.robot.data.root_link_vel_w.clone();torch.testing.assert_close(after[:,0]-before[:,0],torch.full((32,),.05,device=env.device),atol=1e-5,rtol=1e-4)
        robust.entry_push(env);env.sim.forward();torch.testing.assert_close(c.robot.data.root_link_vel_w,after)
        assert c.push_done.all();checks['entry_kick_once_and_exact_velocity_delta']=True
        # Prefix states are not forced to copy the aggressive teacher action.
        states=runner.alg.replay_obs[:32].clone();states[:,:46]=runner.alg.prep_command
        td=TensorDict({'actor':states,'critic':states},batch_size=[32])
        p=[p for p in runner.alg.actor.parameters() if p.requires_grad][-1]
        original=p.detach().clone()
        with torch.no_grad():p.add_(.01)
        live,replay=runner.alg.anchor_losses(td)
        assert live.item()==0 and replay.item()>0
        replay.backward();assert p.grad is not None and p.grad.abs().sum()>0
        assert all(p.grad is None for p in runner.alg.teacher.parameters())
        with torch.no_grad():p.copy_(original)
        runner.alg.optimizer.zero_grad();checks['prefix_excluded_motion_anchor_has_gradients']=True
        # Costs must detect excessive targets, not only visible applied motion.
        env.reset(seed=2026);c=prep.command(env);c.prep_origin[:]=True
        env.action_manager.process_action(torch.zeros((32,23),device=env.device))
        baseline=robust.entry_action_cost(env,'excess')
        env.action_manager.process_action(torch.full((32,23),8.,device=env.device))
        assert (robust.entry_action_cost(env,'excess')>baseline).all()
        c.prep_origin[:]=False;assert robust.entry_action_cost(env,'excess').max()==0
        checks['initial_output_cost_active_and_motion_ungated']=True
        env.reset(seed=42)
        weights={k:v.clone() for k,v in runner.alg.actor.state_dict().items()}
        runner.learn(3)
        assert any(not torch.equal(v,weights[k]) for k,v in runner.alg.actor.state_dict().items() if v.is_floating_point())
        assert all(torch.isfinite(v).all() for v in runner.alg.actor.state_dict().values() if v.is_floating_point())
        checks['ppo_three_updates_finite']=True
        # Save a normal, standalone MLP ONNX. No entry-clock or extra input.
        from rsl_rl.models import MLPModel
        cpu_actor=MLPModel(TensorDict({'actor':torch.zeros(1,124)},batch_size=[1]),{'actor':['actor']},'actor',23,hidden_dims=agent.actor.hidden_dims,activation=agent.actor.activation,obs_normalization=agent.actor.obs_normalization,distribution_cfg=dict(agent.actor.distribution_cfg))
        cpu_actor.load_state_dict({k:v.detach().cpu() for k,v in runner.alg.actor.state_dict().items()},strict=True)
        wrapper=cpu_actor.as_onnx(verbose=False).eval()
        torch.onnx.export(wrapper,wrapper.get_dummy_inputs(),str(out/'policy_smoke.onnx'),input_names=wrapper.input_names,output_names=wrapper.output_names,opset_version=17,dynamo=False,external_data=False)
        import onnx
        from onnx.reference import ReferenceEvaluator
        model=onnx.load(out/'policy_smoke.onnx');onnx.checker.check_model(model)
        assert [x.dim_value for x in model.graph.input[0].type.tensor_type.shape.dim]==[1,124]
        assert [x.dim_value for x in model.graph.output[0].type.tensor_type.shape.dim]==[1,23]
        ref=ReferenceEvaluator(model)
        x=wrap.get_observations()['actor'][:1].detach()
        with torch.no_grad():native=runner.alg.actor(TensorDict({'actor':x},batch_size=[1]),stochastic_output=False).cpu().numpy()
        exported=ref.run(None,{model.graph.input[0].name:x.cpu().numpy()})[0]
        np.testing.assert_allclose(native,exported,atol=2e-5,rtol=2e-4)
        checks['onnx124_export_and_parity']=True
        report={'all_pass':True,'checks':checks,'new_policy_robustness_validated':False,'long_training_started':False}
        (out/'verification.json').write_text(json.dumps(report,indent=2));print('ROBUST_MECHANICS_PASS',json.dumps(report),flush=True)
    finally:env.close()

if __name__=='__main__':verify()
