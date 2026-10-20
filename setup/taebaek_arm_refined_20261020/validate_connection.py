from pathlib import Path
import sys,runpy,json,hashlib
import numpy as np,mujoco as mj,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));HERE=Path(__file__).resolve().parent
mod=runpy.run_path(str(HERE/'train_taebaek.py'),run_name='taebaek_connection_validation')
from source.tasks.mimic.mdp.reference_trajectory import ReferenceTrajectory
m=mj.MjModel.from_xml_path(str(ROOT/'third_party/ai_sapiens/ai_sapiens_description/mujoco/k1/k1.xml'))
names=tuple(m.joint(j).name for j in range(m.njnt)if m.jnt_type[j]==mj.mjtJoint.mjJNT_HINGE);bodies=tuple(m.body(b).name for b in range(1,m.nbody))
path=mod['MOTION'];expected='6145604652ebb3aaa8fabf7be7682ff04ddea9ae1d33c11f17936127b04c47f9';assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
cfg=mod['env_cfg']();play=mod['env_cfg'](play=True);assert cfg.commands['reference_trajectory'].trajectory_file==play.commands['reference_trajectory'].trajectory_file==str(path)
ref=ReferenceTrajectory(str(path),names,bodies,tuple(cfg.commands['reference_trajectory'].body_names),device='cpu')
assert ref.joint_position.shape==(2360,23) and ref.fps==50
with np.load(path)as z:
 c=np.loadtxt(path.with_suffix('.csv'),delimiter=',');assert c.shape==(2360,30);assert np.max(abs(c[:,7:]-z['joint_pos']))<2e-7
 assert all(np.isfinite(z[k]).all()for k in ('joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w'))
assert abs(cfg.sim.mujoco.timestep-.005)<1e-10 and abs(cfg.sim.mujoco.timestep*cfg.decimation-.02)<1e-10
report={'motion':str(path),'sha256':expected,'reference_loader_passed':True,'train_and_play_connected':True,'robot_joint_names_match':True,'robot_body_names_match':True,'csv_npz_match':True,'frames':2360,'fps':50,'duration_s':47.18,'physics_hz':200,'control_hz':50,'training_started':False}
(HERE/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
