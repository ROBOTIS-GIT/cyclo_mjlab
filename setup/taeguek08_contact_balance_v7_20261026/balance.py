import torch
from mjlab.utils.lab_api.math import quat_apply_inverse
from source.tasks.mimic.mdp.performance_contacts import foot_load_history
from rewards import ContactGeometry
from balance_costs import support_gate, support_terms, recovery_terms, WEIGHTS


class ContactBalance(ContactGeometry):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        ref=self.cmd.reference;fps=float(ref.fps)
        self.time=torch.arange(ref.num_frames, device=env.device)/fps
        self.ref_vz=torch.gradient(self.reference_bottom, spacing=(1/fps,), dim=0)[0]
        self.pelvis=self.cmd.cfg.body_names.index('pelvis')
        self.up=torch.tensor([0.,0.,1.],device=env.device)
        self.ref_gravity=quat_apply_inverse(ref.body_orientation_w[:,self.feet],
                                           self.up.expand(ref.num_frames,2,3))
        eligible=(self.reference_support & ((self.phase==1)|(self.phase==2)) &
                  (self.reference_bottom<.015) & (self.reference_lowpoint_speed<.20)).cpu()
        ages=torch.zeros_like(eligible,dtype=torch.float32)
        for i in range(1,ref.num_frames):
            ages[i]=torch.where(eligible[i]&eligible[i-1],ages[i-1]+1/fps,0.)
        self.gates=support_gate(self.reference_bottom,self.reference_lowpoint_speed,
                               self.phase,self.reference_support,ages.to(env.device))

    def __call__(self,env,ramp_steps=48000):
        if self.last_step==env.common_step_counter:return self.cached
        frame,_,valid=self.advance(env)
        c=self.cmd;_,bottom,_,speed,down=self.geometry(env)
        loads=foot_load_history(env)[:,:,:env.cfg.decimation+1]
        gravity=quat_apply_inverse(c.robot_body_quat_w[:,self.feet],
                                   self.up.expand(env.num_envs,2,3))
        terms=support_terms(self.gates[frame],speed,self.reference_lowpoint_speed[frame],
                            loads,self.weight_n,bottom,self.reference_bottom[frame],
                            gravity,self.ref_gravity[frame])
        terms.update(recovery_terms(self.time[frame],
            c.robot_body_pos_w[:,self.pelvis,:2],c.body_pos_w[:,self.pelvis,:2],
            bottom,down,self.reference_bottom[frame],self.ref_vz[frame],
            c.robot_body_lin_vel_w[:,self.pelvis,2],c.body_lin_vel_w[:,self.pelvis,2]))
        ramp=min(1.,env.common_step_counter/max(1,ramp_steps))
        self.cached=ramp*valid*sum(WEIGHTS[k]*terms[k] for k in WEIGHTS)
        for k,v in terms.items():c.metrics['t8_v7_'+k]=v*valid
        c.metrics['t8_v7_cost']=self.cached.clone()
        c.metrics['t8_v7_ramp']=torch.full_like(valid,ramp)
        c.metrics['t8_v7_support_feet']=self.gates[frame].sum(-1)*valid
        self.last_frame.copy_(frame);self.last_step=env.common_step_counter
        return self.cached
