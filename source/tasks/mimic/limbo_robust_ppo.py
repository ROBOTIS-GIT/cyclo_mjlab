"""Motion-only expert anchoring; preparation is free to learn recovery actions."""
import numpy as np
import torch
from tensordict import TensorDict
from .limbo_preserve_ppo import PreservePPO


class RobustPPO(PreservePPO):
    def __init__(self,*args,prep_reference_file,anchor_action_std=.2,**kwargs):
        super().__init__(*args,**kwargs)
        with np.load(prep_reference_file,allow_pickle=False) as z:
            prefix = np.concatenate([z['joint_pos'][0],z['joint_vel'][0]])
        self.prep_command = torch.tensor(prefix,device=self.device,dtype=torch.float32)
        self.anchor_action_std = anchor_action_std
        mask = ~self.is_prefix(self.replay_obs)
        self.motion_replay_obs = self.replay_obs[mask]
        self.motion_replay_actions = self.replay_actions[mask]
        if not len(self.motion_replay_obs):
            raise ValueError('No motion expert states outside stationary preparation')

    def is_prefix(self,observations):
        return (observations[:,:46]-self.prep_command).abs().amax(-1)<1e-4

    def anchor_losses(self,observations):
        states = observations['actor']
        with torch.no_grad():
            target = self.teacher(observations,stochastic_output=False)
        current = self.actor(observations,stochastic_output=False)
        per_state = .5*((current-target)/self.anchor_action_std).square().mean(-1)
        motion = (~self.is_prefix(states)).float()
        live = (per_state*motion).sum()/motion.sum().clamp_min(1.)
        ids = torch.randint(len(self.motion_replay_obs),(self.replay_batch_size,),device=self.device)
        batch = TensorDict({'actor':self.motion_replay_obs[ids],'critic':self.motion_replay_obs[ids]},batch_size=[len(ids)])
        prediction = self.actor(batch,stochastic_output=False)
        replay = .5*((prediction-self.motion_replay_actions[ids])/self.anchor_action_std).square().mean()
        return live,replay
