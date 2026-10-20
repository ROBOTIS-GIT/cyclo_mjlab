"""Build a settled-pose prefix without changing the deployment reference clock."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.tools.motion.csv_to_npz import (
  Motion, load_model, get_model_layout, save_motion_files, _slerp,
  K1_REV1_MOTION_CSV_JOINT_NAMES,
)
ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / 'source/assets/robots/ai_sapiens_entry'
DIR = ROOT / 'source/assets/motions/K1_rev1/umr_stand_up'
SOURCE = DIR / 'k1_0008_stand_up_soft_landing_converted.csv'
DEST = DIR / 'k1_0008_stand_up_ai_sapiens_soft_entry'

def make_prefix(original, pose, joints, fps=50., hold=.3, end=1.5):
  t = np.arange(len(original)) / fps
  s = np.clip((t-hold)/(end-hold), 0., 1.)
  w = s**3 * (10. - 15.*s + 6.*s*s)
  out = original.copy()
  out[:, :3] = pose[:3] + w[:, None] * (original[:, :3] - pose[:3])
  out[:, 7:] = joints + w[:, None] * (original[:, 7:] - joints)
  quat = _slerp(np.broadcast_to(pose[3:7], (len(out), 4)), original[:, [6,3,4,5]], w)
  out[:, 3:7] = quat[:, [1,2,3,0]]
  out[t >= end] = original[t >= end]
  return out

def main():
  original = np.loadtxt(SOURCE, delimiter=',')
  bank = np.load(ASSET / 'ready_bank.npz', allow_pickle=False)
  order = [bank['joint_names'].tolist().index(n) for n in K1_REV1_MOTION_CSV_JOINT_NAMES]
  pose, joints = bank['root_pose'][0].astype(float), bank['joint_pos'][0, order].astype(float)
  # Keep the original horizontal origin; reset alignment is translation invariant.
  pose[:2] = original[0, :2]
  out = make_prefix(original, pose, joints)
  model = load_model(ASSET / 'mujoco/k1/k1.xml')
  layout = get_model_layout(model)
  motion = Motion(50., out[:, :3], out[:, [6,3,4,5]], out[:, 7:])
  errors = save_motion_files(model, layout, motion, DEST.with_suffix('.npz'), DEST.with_suffix('.csv'))
  assert max(errors) < 1e-5
  np.testing.assert_allclose(out[75:], original[75:], atol=0, rtol=0)
  np.testing.assert_allclose(out[:16, 7:], np.broadcast_to(joints, (16,23)), atol=1e-12)
  report = dict(version=2, fps=50, frames=len(out), settled_prefix_seconds=.3,
    original_trajectory_restored_at_seconds=1.5, interpolation='quintic + quaternion slerp',
    original_height_m=float(original[0,2]), settled_height_m=float(out[0,2]),
    initial_joint_command_vs_measured_max_rad=float(abs(original[0,7:]-joints).max()),
    max_initial_root_speed_m_s=float(np.linalg.norm(np.diff(out[:76,:3],axis=0)*50,axis=1).max()),
    max_initial_joint_speed_rad_s=float(abs(np.diff(out[:76,7:],axis=0)*50).max()),
    fk_max_position_error_m=errors[0], fk_max_orientation_error_rad=errors[1],
    bank_sha256=hashlib.sha256((ASSET/'ready_bank.npz').read_bytes()).hexdigest(),
    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    csv_sha256=hashlib.sha256(DEST.with_suffix('.csv').read_bytes()).hexdigest(),
    npz_sha256=hashlib.sha256(DEST.with_suffix('.npz').read_bytes()).hexdigest())
  DEST.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
