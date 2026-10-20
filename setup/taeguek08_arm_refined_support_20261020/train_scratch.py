"""Taegeuk 8 corrected reference + tested support rewards, fresh PPO training."""
from pathlib import Path
import sys,json,fcntl
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import source.tasks
from mjlab.tasks.registry import register_mjlab_task
from mjlab.managers.reward_manager import RewardTermCfg
from source.tasks.mimic.config.k1_rev1.taeguek_support_env_cfg import k1_rev1_taeguek_support_env_cfg
from source.tasks.mimic.config.k1_rev1.agents.rsl_rl_ppo_cfg import k1_rev1_mimic_ppo_runner_cfg
sys.path.insert(0,str(ROOT/'setup/taeguek3478_refinement_20261019'))
from refinement import SupportRefinement
TASK='Cyclo-Mimic-K1-Rev1-Taeguek08-ArmRefined-SupportScratch'
MOTION=HERE/'motion/k1_taeguek08_arm_refined_50hz_converted.npz'

def env_cfg(play=False):
    cfg=k1_rev1_taeguek_support_env_cfg(8,play=play)
    cmd=cfg.commands['reference_trajectory']
    cmd.trajectory_file=str(MOTION)
    cmd.first_frame_env_fraction=0.2 if not play else 0.
    cmd.start_sampling_uniform_mix=0.2
    cfg.episode_length_s=(2519/50+2) if not play else int(1e9)
    cfg.scene.num_envs=4096 if not play else 1
    cfg.rewards['taeguek_support_quality']=RewardTermCfg(func=SupportRefinement,weight=-1.,params={
        'height_weight':1.2,'rising_weight':.35,'flight_weight':.6,'ramp_steps':24000})
    return cfg

agent=k1_rev1_mimic_ppo_runner_cfg()
agent.max_iterations=30000
agent.run_name='taeguek08_arm_refined_support_scratch'
agent.resume=False
agent.upload_model=False
register_mjlab_task(task_id=TASK,env_cfg=env_cfg(),play_env_cfg=env_cfg(True),rl_cfg=agent)

if __name__=='__main__':
    if sys.argv[1:]==['--check']:
        import torch
        z=np.load(MOTION,allow_pickle=False);csv=np.loadtxt(MOTION.with_suffix('.csv'),delimiter=',')
        assert z['joint_pos'].shape==(2519,23) and float(z['fps'][0])==50
        for k in ['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']:
            assert np.isfinite(z[k]).all(),k
        assert np.max(abs(csv[:,7:]-z['joint_pos']))<1e-6
        cfg=env_cfg();assert abs(cfg.sim.mujoco.timestep*cfg.decimation-.02)<1e-9
        assert torch.cuda.is_available()
        print(json.dumps({'task':TASK,'motion':str(MOTION),'frames':len(csv),'duration_s':(len(csv)-1)/50,
            'num_envs':cfg.scene.num_envs,'max_iterations':agent.max_iterations,'resume':agent.resume,
            'checkpoint':None,'learning_rate':agent.algorithm.learning_rate,'schedule':agent.algorithm.schedule,
            'support_reward':cfg.rewards['taeguek_support_quality'].params,'episode_length_s':cfg.episode_length_s,
            'gpu':torch.cuda.get_device_name(0),'training_started':False},indent=2))
    else:
        if any(a=='--checkpoint' or a.startswith('--checkpoint=') for a in sys.argv):
            raise ValueError('This entry point is for scratch training; checkpoint loading is disabled')
        if '--agent.resume' in sys.argv or any(a.startswith('--agent.resume=') for a in sys.argv):
            raise ValueError('Do not override scratch training resume=False')
        lock=(HERE/'training.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from scripts.reinforcement_learning.train import main
        sys.argv.insert(1,TASK)
        main()
