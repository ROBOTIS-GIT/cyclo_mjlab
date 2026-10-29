"""Taegeuk 4 selected-run reward terms, reused for Like Jennie.

The sidekick term intentionally retains its original 14.4–14.85 s mask;
its relevance to this different motion requires evaluation.
"""
import torch
from mjlab.utils.lab_api.math import quat_apply
from source.tasks.mimic.mdp.performance_contacts import foot_load_history
from source.tasks.mimic.mdp.taeguek_support import TaegeukFootSupportQuality
from source.tasks.mimic.mdp.taeguek_support_v2 import sole_bottom_height, kick_height_cost
from source.tasks.mimic.mdp.kpop_support import support_ramp, huber

RAMP_STEPS = 120000

def arm_pose_tracking(env,std=.35):
    c=env.command_manager.get_term('reference_trajectory')
    ids=[i for i,n in enumerate(c.robot.joint_names) if any(x in n for x in ['shoulder','elbow','wrist'])]
    return torch.exp(-((c.robot_joint_pos[:,ids]-c.joint_pos[:,ids])/std).square()).mean(-1)


class KickPreservation(TaegeukFootSupportQuality):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.pelvis=self.cmd.cfg.body_names.index('pelvis')
        self.kick=((self.reference_bottom-self.reference_bottom.amin(0))>.25)&~self.reference_support

    def __call__(self,env,ramp_steps=RAMP_STEPS):
        c=self.cmd;f=c.frame_ids
        b=sole_bottom_height(c.robot_body_pos_w[:,self.feet],c.robot_body_quat_w[:,self.feet],self.offsets,self.radii)-env.scene.env_origins[:,2,None]
        p=c.robot_body_pos_w[:,self.pelvis,2]-env.scene.env_origins[:,2]
        rp=c.reference.body_position_w[f,self.pelvis,2]
        cost=kick_height_cost(b,self.reference_bottom[f],p,rp,self.kick[f]).sum(-1)
        valid=(env.episode_length_buf>8).float()
        c.metrics['t6_kick_height_abs_error_m']=((b-self.reference_bottom[f]).abs()*self.kick[f]).sum(-1)*valid
        return support_ramp(env.common_step_counter,ramp_steps)*cost*valid


def impact_terms(loads,weight_n,dt):
    x=loads[:,:,:-1]
    peak=huber((x.amax(-1)/weight_n-2.2).clamp_min(0)).sum(-1)
    peak+=huber((x.sum(1).amax(-1)/weight_n-3.).clamp_min(0)/1.5)
    rise=((loads[:,:,:-1]-loads[:,:,1:])/(weight_n*dt)).clamp_min(0)
    rate=huber((rise-80.).clamp_min(0)/120.).amax(-1).sum(-1)
    return peak,rate


def soft_contact(env,ramp_steps=RAMP_STEPS):
    c=env.command_manager.get_term('reference_trajectory');loads=foot_load_history(env)
    peak,rate=impact_terms(loads,float(env.sim.mj_model.body_mass.sum())*9.81,env.cfg.sim.mujoco.timestep)
    valid=(env.episode_length_buf>8).float()
    c.metrics['t6_contact_peak_n']=loads[:,:,:-1].sum(1).amax(-1)*valid
    # At most .15 total cost after the RewardTerm weight; auxiliary to tracking.
    return support_ramp(env.common_step_counter,ramp_steps)*torch.tanh(peak+.08*rate)*valid


def extra_grf_cost(loads, weight_n, dt):
    """Excess total force peak and rising load at physical substeps."""
    x=loads[:,:,:-1].sum(1) / weight_n
    peak=huber((x.amax(-1)-2.5).clamp_min(0) / 1.5)
    excess=huber((x-2.5).clamp_min(0) / 1.5).mean(-1)
    total=loads.sum(1) / weight_n
    rate=((total[:,:-1]-total[:,1:])/dt).clamp_min(0)
    rise=huber((rate-80.).clamp_min(0)/120.).amax(-1)
    return peak,excess,rise


def extra_grf(env, ramp_steps=48000):
    loads=foot_load_history(env)
    peak,excess,rise=extra_grf_cost(loads,float(env.sim.mj_model.body_mass.sum())*9.81,env.cfg.sim.mujoco.timestep)
    valid=(env.episode_length_buf>8).float()
    ramp=1. if ramp_steps == 1 else min(1.,env.common_step_counter/max(1,ramp_steps))
    c=env.command_manager.get_term('reference_trajectory')
    c.metrics['t4_grf_peak_cost']=peak*valid
    c.metrics['t4_grf_excess_cost']=excess*valid
    c.metrics['t4_grf_rise_cost']=rise*valid
    c.metrics['t4_grf_ramp']=torch.full_like(peak,ramp)
    return ramp*(peak + .5*excess + .08*rise)*valid


def sidekick_stability_cost(bottom, ref_bottom, actual_up, reference_up, mask):
    # Yaw-free sole normal error: a pure pivot around the world vertical is free.
    tilt_error=torch.linalg.vector_norm(actual_up-reference_up,dim=-1)
    tilt=huber((tilt_error-.035).clamp_min(0)/.087)
    lift=huber((bottom-ref_bottom-.003).clamp_min(0)/.01)
    # Bound added costs so recovery and kick tracking remain possible.
    return torch.tanh(tilt)*mask, torch.tanh(lift)*mask


class SidekickSupport(TaegeukFootSupportQuality):
    def __call__(self,env,ramp_steps=72000):
        c=self.cmd;f=c.frame_ids
        actual_pos=c.robot_body_pos_w[:,self.feet]
        actual_quat=c.robot_body_quat_w[:,self.feet]
        bottom=sole_bottom_height(actual_pos,actual_quat,self.offsets,self.radii)-env.scene.env_origins[:,2,None]
        up=torch.zeros((env.num_envs,3),device=env.device);up[:,2]=1.
        actual_up=quat_apply(actual_quat[:,0],up)
        reference_up=quat_apply(c.body_quat_w[:,self.feet[0]],up)
        t=f/float(c.reference.fps)
        mask=((t>=14.4)&(t<14.85)&self.reference_support[f,0]&(env.episode_length_buf>8)).float()
        tilt,lift=sidekick_stability_cost(bottom[:,0],self.reference_bottom[f,0],actual_up,reference_up,mask)
        ramp=min(1.,env.common_step_counter/max(1,ramp_steps))
        c.metrics['t4_sidekick_tilt_cost']=tilt
        c.metrics['t4_sidekick_lift_cost']=lift
        c.metrics['t4_sidekick_ramp']=torch.full_like(tilt,ramp)
        return ramp*(.08*tilt+.06*lift)

