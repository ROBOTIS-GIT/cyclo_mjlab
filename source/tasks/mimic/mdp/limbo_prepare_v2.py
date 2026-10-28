"""Observable preparation, bounded entry geometry and pre-reset episode counts."""
from dataclasses import dataclass
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse
from . import limbo_prepare_v1 as v1

class LimboPreparationV2Command(v1.LimboPreparationCommand):
  def __init__(self,cfg,env):
    super().__init__(cfg,env)
    self.progress_previous=torch.zeros(self.num_envs,device=self.device)
    self.progress_valid=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
    self.episode_seen=torch.zeros_like(self.progress_valid)
    self.episode_reached=torch.zeros_like(self.progress_valid)
    self.episode_success=torch.zeros_like(self.progress_valid)
    self.total_prepare_episodes=0;self.total_prepare_reached=0;self.total_prepare_successes=0
    self.metrics['prep_relative_foot_error']=torch.zeros(self.num_envs,device=self.device)

  def _resample_command(self,env_ids):
    super()._resample_command(env_ids)
    ids=torch.as_tensor(env_ids,device=self.device,dtype=torch.long)
    self.progress_previous[ids]=0;self.progress_valid[ids]=False
    self.episode_seen[ids]=False;self.episode_reached[ids]=False;self.episode_success[ids]=False
    # Half of preparation episodes explicitly start at full standing. The
    # remainder use all 65 bank poses; motion replay remains in the parent.
    prep_ids=ids[self.preparing[ids]]
    if self.cfg.start_from_zero or len(prep_ids)==0:return
    chosen=prep_ids[torch.rand(len(prep_ids),device=self.device)<self.cfg.full_standing_fraction]
    if len(chosen)==0:return
    roots=self.reset_roots[-1].expand(len(chosen),-1).clone();roots[:,:3]+=self._env.scene.env_origins[chosen]
    joints=self.reset_joints[-1].expand(len(chosen),-1)
    self.robot.write_joint_state_to_sim(joints,torch.zeros_like(joints),env_ids=chosen)
    self.robot.write_root_link_pose_to_sim(roots,env_ids=chosen)
    self.robot.write_root_link_velocity_to_sim(torch.zeros((len(chosen),6),device=self.device),env_ids=chosen)
    self.robot.clear_state(env_ids=chosen);self.reset_alpha[chosen]=1.

  def relative_foot_error(self):
    actual=self.robot_body_pos_w[:,self.foot_ids]
    target=self.body_pos_w[:,self.foot_ids]
    actual=actual-actual.mean(1,keepdim=True)
    target=target-target.mean(1,keepdim=True)
    return torch.linalg.vector_norm(actual-target,dim=-1).amax(-1)

  def readiness(self):
    good,joint,feet,speed=super().readiness()
    return good & (self.relative_foot_error()<self.cfg.ready_relative_foot_error),joint,feet,speed

  def reset(self,env_ids):
    # These flags were captured by record_episode() in the reward pass, before
    # scene/reset/command processing can replace the terminal state.
    ids=env_ids[self.episode_seen[env_ids]&self.prep_origin[env_ids]]
    n=len(ids)
    if n:
      self.total_prepare_episodes+=n
      self.total_prepare_reached+=int(self.episode_reached[ids].sum())
      self.total_prepare_successes+=int(self.episode_success[ids].sum())
    info=super().reset(env_ids)
    info.update(prep_completed_episodes_total=float(self.total_prepare_episodes),
                prep_reached_episodes_total=float(self.total_prepare_reached),
                prep_successful_episodes_total=float(self.total_prepare_successes),
                prep_reached_rate_cumulative=self.total_prepare_reached/max(self.total_prepare_episodes,1),
                prep_full_success_rate_cumulative=self.total_prepare_successes/max(self.total_prepare_episodes,1))
    return info

  def _update_metrics(self):
    super()._update_metrics()
    self.metrics['prep_relative_foot_error'][:]=self.relative_foot_error()

@dataclass(kw_only=True)
class LimboPreparationV2CommandCfg(v1.LimboPreparationCommandCfg):
  initial_alpha_max: float=1.
  curriculum_steps: int=1
  ready_foot_error: float=.07
  ready_relative_foot_error: float=.04
  full_standing_fraction: float=.5
  def build(self,env):return LimboPreparationV2Command(self,env)

def command(env):return env.command_manager.get_term('reference_trajectory')
def preparation_state(env):
  """12 channels: body-frame signed foot errors(6), root COM velocity(3),
  measured contact flags(2), elapsed/max preparation time(1). Zero in LIMBO.
  """
  c=command(env);q=c.robot_body_quat_w[:,0]
  delta=c.body_pos_w[:,c.foot_ids]-c.robot_body_pos_w[:,c.foot_ids]
  err=quat_apply_inverse(q[:,None,:].expand(-1,2,-1),delta).reshape(env.num_envs,6)
  vel=quat_apply_inverse(q,c.robot_body_lin_vel_w[:,0])
  contacts=torch.stack([torch.linalg.vector_norm(env.scene[f'limbo_{s}_contact'].data.force,dim=-1).reshape(env.num_envs,-1).amax(-1)>5 for s in ('left','right')],-1).float()
  elapsed=(c.prep_elapsed/c.cfg.maximum_prepare_s).clamp(0,1)[:,None]
  return torch.cat((err,vel,contacts,elapsed),-1)*c.preparing[:,None].float()

def progress_cost(c):
  _,joint,feet,speed=c.readiness()
  return joint/.1+feet/.07+c.relative_foot_error()/.04+.25*speed/.15

def preparation_progress(env):
  c=command(env);cost=progress_cost(c)
  delta=torch.where(c.progress_valid & c.preparing,c.progress_previous-cost,torch.zeros_like(cost))
  c.progress_previous[:]=cost.detach();c.progress_valid[:]=c.preparing
  return delta/env.step_dt

def motion_only(env,original_func,original_params):
  return original_func(env,**original_params)*(~command(env).preparing).float()
def preparation_failed(env):
  c=command(env)
  return (c.preparing & env.termination_manager.terminated).float()
def record_episode(env):
  c=command(env);c.episode_seen[:]=True
  c.episode_reached |= c.prepared
  c.episode_success |= v1.full_success(env)
  return torch.zeros(env.num_envs,device=env.device)
