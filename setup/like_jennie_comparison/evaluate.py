"""Deterministic CPU MuJoCo rollout for comparing Like Jennie ONNX policies.

Uses each policy's sim2real.yaml and the supplied matching CSV/NPZ reference.
Output is a reproducible diagnostic, not an official Sim2Sim or hardware test.
"""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import mujoco as mj
import onnxruntime as ort
import yaml
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XML = ROOT/'third_party/ai_sapiens/ai_sapiens_description/mujoco/k1/scene.xml'
OBS = ['motion_command','motion_anchor_ori_b','base_ang_vel','joint_pos_rel','joint_vel_rel','last_action']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def rotation(q):return R.from_quat(np.roll(q,-1))
def yaw(rot):
    m=rot.as_matrix();return np.arctan2(m[1,0],m[0,0])

def run(args):
    paths=[args.policy,args.config,args.csv,args.npz,args.xml]
    for p in paths:
        if not p.is_file():raise FileNotFoundError(p)
    cfg=yaml.safe_load(args.config.read_text());x=np.loadtxt(args.csv,delimiter=',');z=np.load(args.npz,allow_pickle=False)
    names=cfg['policy_joints']; count=len(names)
    if list(z['joint_names'])!=names:raise ValueError('Reference joint order does not match policy config')
    if list(cfg['observations'])!=OBS:raise ValueError('Unexpected observation ordering; evaluator needs explicit adaptation')
    for name,term in cfg['observations'].items():
        if term.get('history_length',1)!=1 or term.get('clip') is not None or not np.allclose(term.get('scale',1),1):raise ValueError(f'Unsupported observation processing: {name}')
    if cfg['actions']['joint_pos'].get('clip') is not None:raise ValueError('Action clip requires explicit evaluator implementation')
    if x.shape!=(len(z['joint_pos']),7+count) or not np.allclose(x[:,7:],z['joint_pos'],atol=1e-6):raise ValueError('CSV/NPZ motion mismatch')
    if float(np.asarray(z['fps']).reshape(-1)[0])!=50 or cfg['step_dt']!=.02:raise ValueError('Expected 50 Hz policy and reference')
    if not np.isfinite(x).all() or not np.isfinite(z['joint_vel']).all():raise ValueError('Reference contains nonfinite values')
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(args.policy),options,providers=['CPUExecutionProvider'])
    if session.get_inputs()[0].shape != [1,124] or session.get_outputs()[0].shape != [1,count]:raise ValueError('Unsupported ONNX dimensions')
    meta=session.get_modelmeta().custom_metadata_map
    if meta.get('joint_names') and meta['joint_names'].split(',')!=names:raise ValueError('ONNX joint names disagree with config')
    if meta.get('observation_names') and meta['observation_names'].split(',')!=OBS:raise ValueError('ONNX observation order disagrees with config')
    m=mj.MjModel.from_xml_path(str(args.xml));d=mj.MjData(m);ref=mj.MjData(m)
    jids=np.array([m.joint(n).id for n in names]);qa=m.jnt_qposadr[jids];va=m.jnt_dofadr[jids]
    acts=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in jids])
    m.actuator_biastype[acts]=mj.mjtBias.mjBIAS_AFFINE;m.actuator_ctrllimited[acts]=False;m.actuator_biasprm[acts]=0
    kp=np.array([cfg['joint_properties'][j]['stiffness'] for j in names]);kd=np.array([cfg['joint_properties'][j]['damping'] for j in names]);default=np.array([cfg['joint_properties'][j]['default_position'] for j in names]);scale=np.array(cfg['actions']['joint_pos']['scale']);offset=np.array(cfg['actions']['joint_pos']['offset'])
    d.qpos[:3]=x[0,:3];d.qpos[3:7]=x[0,[6,3,4,5]];d.qpos[qa]=x[0,7:];mj.mj_forward(m,d)
    initial_torso=rotation(d.qpos[3:7])*R.from_euler('z',d.qpos[qa[12]])
    ref_torso=R.from_quat(x[0,3:7])*R.from_euler('z',x[0,19]);align=R.from_euler('z',yaw(initial_torso)-yaw(ref_torso))
    dt=m.opt.timestep;substeps=round(.02/dt)
    if abs(substeps*dt-.02)>1e-9:raise ValueError('Physics timestep does not divide policy period')
    floor=m.geom('floor').id;feet=[m.body('left_ankle_roll_link').id,m.body('right_ankle_roll_link').id]
    last=np.zeros(count,dtype=np.float32);poses=[d.qpos.copy()];vels=[d.qvel.copy()];samples=[];failure=None
    for i in range(len(x)-1):
        mj.mj_forward(m,d)
        actual=rotation(d.qpos[3:7])*R.from_euler('z',d.qpos[qa[12]])
        target=R.from_quat(x[i,3:7])*R.from_euler('z',x[i,19]);ori=(actual.inv()*align*target).as_matrix()[:,:2].reshape(-1)
        obs=np.r_[x[i,7:],z['joint_vel'][i],ori,d.qvel[3:6],d.qpos[qa]-default,d.qvel[va],last].astype(np.float32)
        last=session.run(None,{session.get_inputs()[0].name:obs[None]})[0][0]
        command=offset+scale*last
        if not np.isfinite(command).all():failure=f'nonfinite output at frame {i}';break
        alpha=min(i*.02/.3,1);alpha=alpha*alpha*(3-2*alpha);command=(1-alpha)*x[0,7:]+alpha*command
        m.actuator_biasprm[acts,1]=-kp;m.actuator_biasprm[acts,2]=-kd;d.ctrl[acts]=command*kp
        for _ in range(substeps):
            mj.mj_step(m,d);mj.mj_forward(m,d)
            forces=np.zeros(2);speeds=np.zeros(2);other=0.;cf=np.zeros(6);jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv))
            for c in range(d.ncon):
                contact=d.contact[c]
                if floor not in (contact.geom1,contact.geom2):continue
                geom=contact.geom2 if contact.geom1==floor else contact.geom1;body=m.geom_bodyid[geom]
                mj.mj_contactForce(m,d,c,cf);normal=max(float(cf[0]),0.)
                if body not in feet:other+=normal;continue
                k=feet.index(body);forces[k]+=normal
                if normal>1:
                    mj.mj_jac(m,d,jp,jr,contact.pos,body);speeds[k]=max(speeds[k],float(np.linalg.norm((jp@d.qvel)[:2])))
            samples.append(np.r_[d.time,forces,speeds,other])
        if not np.isfinite(d.qpos).all() or np.any(d.warning.number):failure=f'physics warning at frame {i}: {d.warning.number.tolist()}';break
        poses.append(d.qpos.copy());vels.append(d.qvel.copy())
    poses=np.array(poses);vels=np.array(vels);samples=np.array(samples);num=len(poses)
    joint_error=poses[:,qa]-x[:num,7:];arm_error=joint_error[:,13:];foot_error=[]
    for i in range(num):
        d.qpos[:]=poses[i];ref.qpos[:3]=x[i,:3];ref.qpos[3:7]=x[i,[6,3,4,5]];ref.qpos[qa]=x[i,7:]
        mj.mj_forward(m,d);mj.mj_forward(m,ref);foot_error.append(np.linalg.norm(d.xpos[feet]-ref.xpos[feet],axis=1))
    foot_error=np.array(foot_error);tilt=np.degrees(np.arccos(np.clip(1-2*(poses[:,4]**2+poses[:,5]**2),-1,1)))
    loaded=samples[:,1:3]>30.;sliding=loaded & (samples[:,3:5]>.15)
    model_xml=args.xml.with_name('k1.xml')
    sidecar=args.policy.with_name(args.policy.name+'.data')
    if model_xml.is_file():paths.append(model_xml)
    if sidecar.is_file():paths.append(sidecar)
    report={'policy':str(args.policy.resolve()),'config':str(args.config.resolve()),'motion_csv':str(args.csv.resolve()),'motion_npz':str(args.npz.resolve()),'model':str(args.xml.resolve()),'sha256':{str(p.resolve()):sha(p) for p in paths},'mode':'first reference pose; zero velocity; 0.3 s target blend; nominal deterministic MuJoCo; no disturbances','fps':50,'physics_dt':dt,'frames':num,'requested_frames':len(x),'failure':failure,'joint_rmse_rad':float(np.sqrt(np.mean(joint_error**2))),'arm_joint_rmse_rad':float(np.sqrt(np.mean(arm_error**2))),'pelvis_xy_rmse_m':float(np.sqrt(np.mean(np.sum((poses[:,:2]-x[:num,:2])**2,axis=1)))),'foot_position_rmse_m':float(np.sqrt(np.mean(foot_error**2))),'minimum_pelvis_height_m':float(poses[:,2].min()),'peak_pelvis_tilt_deg':float(tilt.max()),'maximum_nonfoot_ground_normal_N':float(samples[:,5].max()),'peak_foot_normal_N':np.max(samples[:,1:3],axis=0).tolist(),'loaded_contact_point_speed_over_0_15_fraction':float(sliding.any(axis=1).mean()),'caveat':'Contact speed counts instantaneous loaded collision points; not direct measured physical slip. No fall or stability claim from one trial.'}
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'rollout.npz',qpos=poses,qvel=vels,fps=50,robot_xml=str(args.xml.resolve()),joint_error=joint_error,foot_error=foot_error,contact_samples=samples,contact_columns=np.array(['time','left_normal','right_normal','left_point_speed','right_point_speed','nonfoot_normal']))
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if failure:raise RuntimeError(failure)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--csv',type=Path,required=True);p.add_argument('--npz',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--xml',type=Path,default=DEFAULT_XML);run(p.parse_args())
