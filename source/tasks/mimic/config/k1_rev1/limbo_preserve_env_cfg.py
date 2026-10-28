"""Preserve a validated 124-input actor while refining complete limbo episodes."""
from dataclasses import dataclass, fields
import torch
from mjlab.rl import RslRlPpoAlgorithmCfg
from source import SRC_PATH
from source.tasks.mimic.mdp.limbo_prepare_obs124 import Obs124PreparationCommand, Obs124PreparationCommandCfg
from .limbo_prepare_obs124_env_cfg import k1_rev1_limbo_prepare_obs124_env_cfg, k1_rev1_limbo_prepare_obs124_runner_cfg

class PreservePreparationCommand(Obs124PreparationCommand):
    def _sample_start_frames(self, env_ids):
        super()._sample_start_frames(env_ids)
        ids=torch.as_tensor(env_ids,device=self.device,dtype=torch.long)
        replay=ids[~self.prep_origin[ids]]
        self.frame_ids[replay]=torch.randint(self.reference.num_frames,(len(replay),),device=self.device)
        self.started_at_zero[ids]=self.frame_ids[ids]==0
        # Replay is intentionally uniform, rather than concentrated by failures.
        self.metrics['start_sampling_entropy'][:]=1.
        self.metrics['start_sampling_top1_prob'][:]=1./self.start_bin_count
        self.metrics['start_sampling_top1_bin'][:]=0.

@dataclass(kw_only=True)
class PreservePreparationCommandCfg(Obs124PreparationCommandCfg):
    def build(self,env):return PreservePreparationCommand(self,env)

@dataclass
class PreservePpoCfg(RslRlPpoAlgorithmCfg):
    teacher_checkpoint:str=''
    replay_file:str=''
    teacher_coef:float=1.
    replay_coef:float=4.
    replay_batch_size:int=1024

def k1_rev1_limbo_preserve_env_cfg(play=False):
    cfg=k1_rev1_limbo_prepare_obs124_env_cfg(play=play)
    cfg.scene.num_envs=1 if play else 4096
    old=cfg.commands['reference_trajectory']
    values={f.name:getattr(old,f.name) for f in fields(old)}
    values['preparation_fraction']=.5
    cfg.commands['reference_trajectory']=PreservePreparationCommandCfg(**values)
    cfg.rewards['prepare_reached'].weight=50.
    cfg.rewards['prepare_failed'].weight=-50.
    cfg.rewards['prepare_orientation'].weight=-.25
    cfg.rewards['prepare_stability'].weight=-.1
    cfg.rewards['limbo_full_success'].weight=500.
    return cfg

def k1_rev1_limbo_preserve_runner_cfg():
    cfg=k1_rev1_limbo_prepare_obs124_runner_cfg()
    cfg.experiment_name='k1_limbo_1p2_preserve_obs124'
    cfg.run_name='bar120_preserve_v5'
    cfg.max_iterations=20000
    cfg.save_interval=100
    cfg.algorithm=PreservePpoCfg(**{f.name:getattr(cfg.algorithm,f.name) for f in fields(cfg.algorithm)})
    cfg.algorithm.class_name='source.tasks.mimic.limbo_preserve_ppo:PreservePPO'
    root=SRC_PATH/'assets/checkpoints/limbo'
    cfg.algorithm.teacher_checkpoint=str(root/'obs124_teacher500.pt')
    cfg.algorithm.replay_file=str(root/'teacher500_replay.pt')
    cfg.algorithm.learning_rate=2e-6
    cfg.algorithm.num_learning_epochs=3
    return cfg
