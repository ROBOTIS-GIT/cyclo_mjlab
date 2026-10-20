"""Generate reproducible settled ReadyPose reset states from the deployment snapshot."""
from pathlib import Path
import argparse, hashlib, json
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / 'source/assets/robots/ai_sapiens_entry'
MOTION = ROOT / 'source/assets/motions/K1_rev1/umr_stand_up/k1_0008_stand_up_soft_landing_converted.csv'

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--count', type=int, default=64)
  parser.add_argument('--seed', type=int, default=20261019)
  args = parser.parse_args()
  if args.count < 1:
    raise ValueError('count must be positive')
  contract = json.loads((ASSET / 'contract.json').read_text())
  m = mj.MjModel.from_xml_path(str(ASSET / 'mujoco/k1/scene.xml'))
  names = contract['joints']
  jids = np.array([mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, j) for j in names])
  qa, va = m.jnt_qposadr[jids], m.jnt_dofadr[jids]
  acts = np.array([np.where(m.actuator_trnid[:, 0] == j)[0][0] for j in jids])
  kp, kd, target = [np.asarray(contract[k]) for k in ('ready_kp', 'ready_kd', 'ready_target')]
  m.actuator_biastype[acts] = mj.mjtBias.mjBIAS_AFFINE
  m.actuator_ctrllimited[acts] = False
  m.actuator_biasprm[acts, 1] = -kp
  m.actuator_biasprm[acts, 2] = -kd
  first = np.loadtxt(MOTION, delimiter=',')[0]
  np.testing.assert_allclose(first[7:], target, atol=1e-6)
  rng = np.random.default_rng(args.seed)
  states = {k: [] for k in ('root_pose', 'root_velocity', 'joint_pos', 'joint_vel')}
  for i in range(args.count):
    d = mj.MjData(m)
    d.qpos[:3] = first[:3]
    perturb = np.zeros(3) if i == 0 else rng.uniform(-0.08, 0.08, 3)
    rot = Rotation.from_quat(first[3:7]) * Rotation.from_rotvec(perturb)
    d.qpos[3:7] = np.roll(rot.as_quat(), 1)
    d.qpos[qa] = target + (0 if i == 0 else rng.uniform(-.03, .03, len(names)))
    seconds = 3. if i == 0 else rng.uniform(2.5, 4.)
    for _ in range(round(seconds / m.opt.timestep)):
      d.ctrl[acts] = target * kp
      mj.mj_step(m, d)
    mj.mj_forward(m, d)
    # Freejoint translation velocity is world-frame; rotation velocity is local.
    world_ang = d.xmat[1].reshape(3, 3) @ d.qvel[3:6]
    pose = d.qpos[:7].copy()
    pose[:2] = 0  # translation is irrelevant; preserve settled height/orientation.
    states['root_pose'].append(pose)
    states['root_velocity'].append(np.r_[d.qvel[:3], world_ang])
    states['joint_pos'].append(d.qpos[qa].copy())
    states['joint_vel'].append(d.qvel[va].copy())
  data = {k: np.asarray(v, dtype=np.float32) for k, v in states.items()}
  assert all(np.isfinite(v).all() for v in data.values())
  np.savez_compressed(ASSET / 'ready_bank.npz', **data, joint_names=np.array(names),
    contract_sha256=hashlib.sha256((ASSET / 'contract.json').read_bytes()).hexdigest(),
    model_sha256=hashlib.sha256((ASSET / 'mujoco/k1/k1.xml').read_bytes()).hexdigest(),
    seed=args.seed)
  print(json.dumps({'count':args.count, 'max_joint_error':float(abs(data['joint_pos']-target).max()),
    'max_joint_speed':float(abs(data['joint_vel']).max()), 'root_height_range':data['root_pose'][:,2].tolist()[0:3]}))

if __name__ == '__main__':
  main()
