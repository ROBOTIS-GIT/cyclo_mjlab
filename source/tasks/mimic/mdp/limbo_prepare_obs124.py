"""Timed preparation carried by the reference, with no custom actor inputs."""
from dataclasses import dataclass
import torch
from . import limbo_prepare_v1 as v1, limbo_prepare_v2 as v2, limbo_1p2_v2
from .limbo_prepare_v2 import command

class Obs124PreparationCommand(v2.LimboPreparationV2Command):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.prefix_ok=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.prefix_reached_pulse=torch.zeros(self.num_envs,device=self.device)
        self.prefix_failed_pulse=torch.zeros(self.num_envs,device=self.device)
        self.prefix_errors=torch.zeros((self.num_envs,5),device=self.device)

    def _resample_command(self,env_ids):
        super()._resample_command(env_ids)
        self.prefix_ok[env_ids]=False
        self.prefix_reached_pulse[env_ids]=0
        self.prefix_failed_pulse[env_ids]=0
        self.prefix_errors[env_ids]=0

    def _update_command(self):
        # Advance every policy tick, exactly like the legacy CSV runtime.
        old=self.frame_ids.clone()
        limbo_1p2_v2.Limbo1p2V2Command._update_command(self)
        entering=self.prep_origin & (old<self.cfg.prefix_frames) & (self.frame_ids>=self.cfg.prefix_frames)
        errors=preparation_errors(self._env)
        self.prefix_errors[entering]=errors[entering]
        ready=(errors[:,:4]<torch.tensor([.1,.5,.15,.4],device=self.device)).all(-1)&(errors[:,4]>.5)
        self.prefix_ok[entering]=ready[entering]
        self.prefix_reached_pulse[:]=(entering & ready).float()
        self.prefix_failed_pulse[:]=(entering & ~ready).float()
        self.preparing[:]=self.prep_origin & (self.frame_ids<self.cfg.prefix_frames)
        self.prepared[:]=self.prep_origin & (self.frame_ids>=self.cfg.prefix_frames)
        self.prep_elapsed[:]=self.frame_ids.float().clamp(max=self.cfg.prefix_frames)*self._env.step_dt
        self.prep_timed_out[:]=False

@dataclass(kw_only=True)
class Obs124PreparationCommandCfg(v2.LimboPreparationV2CommandCfg):
    prefix_frames:int=100
    def build(self,env):return Obs124PreparationCommand(self,env)

def preparation_errors(env):
    c=command(env)
    joint=(c.robot_joint_pos-c.reference.joint_position[0]).square().mean(-1).sqrt()
    speed=c.robot_joint_vel.square().mean(-1).sqrt()
    orientation=v1.quat_error_magnitude(c.robot_anchor_quat_w,c.anchor_quat_w)
    omega=torch.linalg.vector_norm(c.robot_body_ang_vel_w[:,0],dim=-1)
    contact=torch.stack([torch.linalg.vector_norm(env.scene[f'limbo_{s}_contact'].data.force,dim=-1).reshape(env.num_envs,-1).amax(-1)>5 for s in ('left','right')],-1).all(-1)
    # No unobservable absolute XY foot-placement gate, no time gate on rollout.
    return torch.stack((joint,speed,orientation,omega,contact.float()),-1)

def preparation_ready(env):
    errors=preparation_errors(env)
    return (errors[:,:4]<torch.tensor([.1,.5,.15,.4],device=errors.device)).all(-1)&(errors[:,4]>.5)

def prefix_reached(env):return command(env).prefix_reached_pulse

def prefix_failed(env):return command(env).prefix_failed_pulse

def full_success(env):
    c=command(env)
    return limbo_1p2_v2.full_success(env) & (~c.prep_origin | c.prefix_ok)

def completion_reward(env):return full_success(env).float()

def record_episode(env):
    c=command(env)
    c.episode_seen[:]=True
    c.episode_reached |= c.prefix_ok
    c.episode_success |= full_success(env)
    return torch.zeros(env.num_envs,device=env.device)
