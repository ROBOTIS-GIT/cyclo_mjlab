# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Record one complete, unteleported Koryo attempt at physics-rate foot contacts."""
import argparse
import json
import os
from pathlib import Path
import sys
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import torch
import source.tasks
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from source.tasks.mimic.mdp.performance_contacts import add_foot_sensors, FOOT_BODIES, SENSOR
from source.tasks.mimic.motion_curriculum import motion_identity, validate_checkpoint_motion


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', required=True, type=Path)
  parser.add_argument('--output-dir', required=True, type=Path)
  parser.add_argument('--seed', default=42, type=int)
  parser.add_argument('--condition', choices=('clean','play'), default='clean')
  parser.add_argument('--physics-dt', type=float, choices=(0.005,0.0025), default=0.005)
  args = parser.parse_args()
  identity = motion_identity('koryo')
  cfg = load_env_cfg('Cyclo-Mimic-K1-Rev1-Koryo', play=True)
  cfg.sim.mujoco.timestep = args.physics_dt
  cfg.decimation = round(0.02/args.physics_dt)
  add_foot_sensors(cfg)
  cfg.seed = args.seed
  cfg.scene.num_envs = 1
  cfg.episode_length_s = identity['frames']/identity['fps'] + 0.1
  cmdcfg = cfg.commands['reference_trajectory']
  cmdcfg.debug_vis = False
  if args.condition == 'clean':
    cfg.events = {}
    cmdcfg.reset_pose_noise = {}
    cmdcfg.reset_velocity_noise = {}
    cmdcfg.reset_joint_position_noise = (0.,0.)
  env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
  try:
    agent = load_rl_cfg('Cyclo-Mimic-K1-Rev1-Koryo')
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device='cuda:0')
    infos = runner.load(str(args.checkpoint), load_cfg={'actor':True}, map_location='cuda:0')
    validate_checkpoint_motion(infos, 'koryo', identity)
    policy = runner.get_inference_policy(device='cuda:0')
    cmd = env.command_manager.get_term('reference_trajectory')
    feet = [cmd.cfg.body_names.index(n) for n in FOOT_BODIES]
    wrapped.reset()
    cmd.frame_ids -= 1
    cmd._update_command()
    obs = wrapped.get_observations()
    data = {k:[] for k in ('time','force_history','foot_pos','foot_vel','ref_foot_pos','ref_foot_vel','action','body_error','anchor_error')}
    finished = False
    outcome = {}
    terms = env.termination_manager
    original_compute = terms.compute

    def record():
      nonlocal finished, outcome
      done = original_compute()
      if finished:
        return done
      tensors = {
        'time': cmd.frame_ids.float()/cmd.reference.fps,
        'force_history':env.scene[SENSOR].data.force_history,
        'foot_pos':cmd.robot_body_pos_w[:,feet], 'foot_vel':cmd.robot_body_lin_vel_w[:,feet],
        'ref_foot_pos':cmd.body_pos_w[:,feet], 'ref_foot_vel':cmd.body_lin_vel_w[:,feet],
        'action':env.action_manager.action,
        'body_error':(cmd.aligned_body_position_w-cmd.robot_body_pos_w).norm(dim=-1).mean(dim=-1),
        'anchor_error':(cmd.anchor_pos_w-cmd.robot_anchor_pos_w).norm(dim=-1),
      }
      for k,v in tensors.items():
        data[k].append(v[0].detach().cpu().numpy().copy())
      end = bool(cmd.frame_ids[0] >= cmd.reference.num_frames-1)
      if bool(done[0]) or end:
        finished = True
        outcome = {'completed':end and not bool(terms.terminated[0]),
                   'failed':bool(terms.terminated[0]),
                   'reference_s':float(cmd.frame_ids[0])/cmd.reference.fps,
                   'terminations':[n for n in terms.active_terms if bool(terms.get_term(n)[0])]}
      return done

    terms.compute = record
    with torch.inference_mode():
      for _ in range(env.max_episode_length + 1):
        obs,_,_,_ = wrapped.step(policy(obs, stochastic_output=False))
        if finished:
          break
    if not finished:
      raise RuntimeError('Attempt did not finish within the evaluation window.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir/'contacts.npz', **{k:np.asarray(v) for k,v in data.items()})
    outcome.update(checkpoint=str(args.checkpoint),condition=args.condition,seed=args.seed,
                   physics_dt=cfg.sim.mujoco.timestep,decimation=cfg.decimation,
                   motion=identity,mean_body_error_m=float(np.mean(data['body_error'])),
                   mean_anchor_error_m=float(np.mean(data['anchor_error'])))
    (args.output_dir/'outcome.json').write_text(json.dumps(outcome,indent=2)+'\n')
    print(json.dumps(outcome,indent=2))
  finally:
    env.close()


if __name__ == '__main__':
  main()
