"""State-gated preparation before 120cm limbo; no AMP and no handoff teleport.

Actor contract stays 127 channels. The existing phase channel is -1 in PREP,
then 0..1 in LIMBO; the final two channels are planned left/right contacts.
This contract requires a matching runtime gate; do not deploy as legacy limbo.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_error_magnitude
from . import limbo, limbo_1p2_v2

class LimboPreparationCommand(limbo_1p2_v2.Limbo1p2V2Command):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    with np.load(cfg.reset_bank_file, allow_pickle=False) as z:
      names=list(z['joint_names'])
      order=[names.index(n) for n in self.robot.joint_names]
      self.reset_roots=torch.tensor(z['root_pose'],device=self.device,dtype=torch.float32)
      self.reset_joints=torch.tensor(z['joint_pos'][:,order],device=self.device,dtype=torch.float32)
    # The hold target must actually be stationary and double-supported.
    assert self.reference.joint_velocity[0].abs().max()<1e-4
    assert self.reference.body_linear_velocity_w[0].abs().max()<1e-4
    assert self.reference.body_angular_velocity_w[0].abs().max()<1e-4
    assert self.contact_plan[0].all()
    self.preparing=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
    self.prep_origin=self.preparing.clone()
    self.prepared=self.preparing.clone()
    self.prep_timed_out=self.preparing.clone()
    self.prep_elapsed=torch.zeros(self.num_envs,device=self.device)
    self.prep_stable=self.prep_elapsed.clone()
    self.prep_steps=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
    self.stable_steps=torch.zeros_like(self.prep_steps)
    self.prep_release_pulse=self.prep_elapsed.clone()
    self.reset_alpha=self.prep_elapsed.clone()
    for key in ('prep_active','prep_ready','prep_timeout','prep_seconds','prep_reset_alpha',
                'prep_joint_rmse','prep_foot_error','prep_root_speed','prep_complete_success'):
      self.metrics[key]=torch.zeros(self.num_envs,device=self.device)

  def _sample_start_frames(self, env_ids):
    super()._sample_start_frames(env_ids)
    ids=torch.as_tensor(env_ids,device=self.device,dtype=torch.long)
    prep=torch.ones(len(ids),device=self.device,dtype=torch.bool) if self.cfg.start_from_zero else torch.rand(len(ids),device=self.device)<self.cfg.preparation_fraction
    self.preparing[ids]=prep;self.prep_origin[ids]=prep
    self.prepared[ids]=False;self.prep_timed_out[ids]=False
    self.prep_elapsed[ids]=0;self.prep_stable[ids]=0;self.prep_release_pulse[ids]=0;self.reset_alpha[ids]=0
    self.prep_steps[ids]=0;self.stable_steps[ids]=0
    self.frame_ids[ids[prep]]=0
    self.started_at_zero[ids]=self.frame_ids[ids]==0

  def _resample_command(self, env_ids):
    super()._resample_command(env_ids)
    ids=torch.as_tensor(env_ids,device=self.device,dtype=torch.long)
    ids=ids[self.preparing[ids]]
    if len(ids)==0:return
    steps=float(getattr(self._env,'common_step_counter',0))
    alpha_max=1. if self.cfg.start_from_zero else min(1.,self.cfg.initial_alpha_max+(1-self.cfg.initial_alpha_max)*steps/max(self.cfg.curriculum_steps,1))
    alpha=torch.full((len(ids),),self.cfg.play_reset_alpha,device=self.device) if self.cfg.start_from_zero else torch.rand(len(ids),device=self.device)*alpha_max
    ix=torch.round(alpha*(len(self.reset_roots)-1)).long().clamp(0,len(self.reset_roots)-1)
    self.reset_alpha[ids]=ix.float()/(len(self.reset_roots)-1)
    roots=self.reset_roots[ix].clone();roots[:,:3]+=self._env.scene.env_origins[ids]
    self.robot.write_joint_state_to_sim(self.reset_joints[ix],torch.zeros_like(self.reset_joints[ix]),env_ids=ids)
    self.robot.write_root_link_pose_to_sim(roots,env_ids=ids)
    self.robot.write_root_link_velocity_to_sim(torch.zeros((len(ids),6),device=self.device),env_ids=ids)
    self.robot.clear_state(env_ids=ids)

  def readiness(self):
    joint_rmse=(self.robot_joint_pos-self.reference.joint_position[0]).square().mean(-1).sqrt()
    joint_speed=self.robot_joint_vel.square().mean(-1).sqrt()
    feet=torch.linalg.vector_norm(self.robot_body_pos_w[:,self.foot_ids]-self.body_pos_w[:,self.foot_ids],dim=-1).amax(-1)
    root_height=(self.robot_body_pos_w[:,0,2]-self.body_pos_w[:,0,2]).abs()
    root_speed=torch.linalg.vector_norm(self.robot_body_lin_vel_w[:,0],dim=-1)
    root_omega=torch.linalg.vector_norm(self.robot_body_ang_vel_w[:,0],dim=-1)
    orientation=quat_error_magnitude(self.robot_anchor_quat_w,self.anchor_quat_w)
    contacts=torch.stack([torch.linalg.vector_norm(self._env.scene[f'limbo_{s}_contact'].data.force,dim=-1).reshape(self.num_envs,-1).amax(-1)>5 for s in ('left','right')],-1).all(-1)
    good=contacts & (joint_rmse<self.cfg.ready_joint_rmse) & (joint_speed<self.cfg.ready_joint_speed) & (feet<self.cfg.ready_foot_error) & (root_height<.05) & (root_speed<self.cfg.ready_root_speed) & (root_omega<.4) & (orientation<.15)
    return good,joint_rmse,feet,root_speed

  def advance_gate(self, ready):
    """Pure gate state update, separated for boundary/timeout regression tests."""
    active=self.preparing.clone()
    self.prep_release_pulse.zero_()
    dt=self._env.step_dt
    self.prep_steps[active]+=1
    self.stable_steps[active]=torch.where(ready[active],self.stable_steps[active]+1,torch.zeros_like(self.stable_steps[active]))
    self.prep_elapsed[active]=self.prep_steps[active].float()*dt
    self.prep_stable[active]=self.stable_steps[active].float()*dt
    min_steps=math.ceil(self.cfg.minimum_prepare_s/dt-1e-9)
    dwell_steps=math.ceil(self.cfg.stable_prepare_s/dt-1e-9)
    max_steps=math.ceil(self.cfg.maximum_prepare_s/dt-1e-9)
    release=active & (self.prep_steps>=min_steps) & (self.stable_steps>=dwell_steps) & (self.prep_steps<=max_steps)
    self.prepared[release]=True;self.preparing[release]=False;self.prep_release_pulse[release]=1
    self.prep_timed_out |= active & ~release & (self.prep_steps>=max_steps)
    return active

  def _update_command(self):
    active=self.advance_gate(self.readiness()[0])
    # Parent increments once. Cancel that increment during PREP, including the
    # release tick, so the first actual LIMBO action observes exactly frame 0.
    self.frame_ids[active]=-1
    super()._update_command()

  def _update_metrics(self):
    super()._update_metrics()
    _,joint,feet,speed=self.readiness()
    for key,value in [('prep_active',self.preparing),('prep_ready',self.prepared),('prep_timeout',self.prep_timed_out),('prep_seconds',self.prep_elapsed),('prep_reset_alpha',self.reset_alpha),('prep_joint_rmse',joint),('prep_foot_error',feet),('prep_root_speed',speed),('prep_complete_success',full_success(self._env)&self.prep_origin)]:
      self.metrics[key][:]=value.float()

@dataclass(kw_only=True)
class LimboPreparationCommandCfg(limbo_1p2_v2.Limbo1p2V2CommandCfg):
  reset_bank_file: str=''
  preparation_fraction: float=.8
  initial_alpha_max: float=.25
  curriculum_steps: int=48000
  play_reset_alpha: float=1.
  minimum_prepare_s: float=.4
  stable_prepare_s: float=.16
  maximum_prepare_s: float=3.
  ready_joint_rmse: float=.10
  ready_joint_speed: float=.5
  ready_foot_error: float=.05
  ready_root_speed: float=.15
  def build(self,env):return LimboPreparationCommand(self,env)

def command(env):return env.command_manager.get_term('reference_trajectory')
def phase_and_contacts(env):
  c=command(env);phase=c.frame_ids.float()/(c.reference.num_frames-1)
  return torch.cat((torch.where(c.preparing,torch.full_like(phase,-1),phase)[:,None],c.contact_plan[c.frame_ids].float()),-1)
def preparation_timeout(env):return command(env).prep_timed_out
def full_success(env):
  c=command(env)
  return limbo_1p2_v2.full_success(env) & (~c.prep_origin | c.prepared) & ~c.prep_timed_out

def completion_reward(env):return full_success(env).float()
def preparation_posture(env):
  c=command(env)
  err=(c.robot_joint_pos-c.reference.joint_position[0]).square().mean(-1)
  return c.preparing.float()*torch.exp(-err/.15**2)
def preparation_stillness(env):
  c=command(env)
  return c.preparing.float()*torch.exp(-c.robot_body_lin_vel_w[:,0].square().sum(-1)/.15**2-c.robot_joint_vel.square().mean(-1)/.7**2)
def preparation_time_cost(env):return command(env).preparing.float()
def preparation_reached(env):return command(env).prep_release_pulse
def stance_motion(env,angular=False):
  c=command(env)
  # Permit stance adjustment during preparation, retain original motion costs.
  return limbo.stance_motion(env,angular)*torch.where(c.preparing,.15,1.)
