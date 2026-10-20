"""Reproducible initial-state augmentation from measured sim2sim transitions."""
from pathlib import Path
import hashlib,json
import numpy as np
import mujoco as mj
HERE=Path(__file__).resolve().parent
AUDIT=HERE.parent/'taeguek_selected_stability_20261019'
model_path=Path('/home/hc/robotis_mujoco/assets/ai_sapiens_description/mujoco/k1_rev1/k1_mjlab.xml')
m=mj.MjModel.from_xml_path(str(model_path));d=mj.MjData(m)
jids=[j for j in range(m.njnt) if m.jnt_type[j]==mj.mjtJoint.mjJNT_HINGE]
assert len(jids)==23
qadr=m.jnt_qposadr[jids];vadr=m.jnt_dofadr[jids];names=[m.joint(j).name for j in jids]
rows=np.genfromtxt(AUDIT/'continuous.csv',names=True,delimiter=',')
manifest={'source':str(AUDIT/'continuous.csv'),'source_sha256':hashlib.sha256((AUDIT/'continuous.csv').read_bytes()).hexdigest(),
          'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),'banks':{}}
for chapter,previous in [(4,2),(7,6),(8,7)]:
    seq=rows[rows['chapter']==previous]
    q=np.column_stack([seq[f'q{i}'] for i in range(m.nq)])
    end=float(seq['policy_time'][-1]);indices=np.flatnonzero((seq['policy_time']>=end-.32)&(seq['policy_time']<=end-.01))[::2]
    poses=[];vels=[];jpos=[];jvel=[]
    for i in indices:
        velocity=np.zeros(m.nv)
        mj.mj_differentiatePos(m,velocity,float(seq['time'][i+2]-seq['time'][i-2]),q[i-2],q[i+2])
        d.qpos[:]=q[i];d.qvel[:]=velocity;mj.mj_forward(m,d)
        jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv))
        mj.mj_jacBodyCom(m,d,jp,jr,m.body('pelvis').id)
        poses.append(q[i,:7]);vels.append(np.r_[jp@velocity,jr@velocity]);jpos.append(q[i,qadr]);jvel.append(velocity[vadr])
    poses=np.array(poses);vels=np.array(vels);jpos=np.array(jpos);jvel=np.array(jvel)
    assert np.isfinite(np.concatenate([poses,vels,jpos,jvel],axis=1)).all()
    assert (poses[:,2]>.5).all() and np.max(abs(jvel))<25
    np.testing.assert_allclose(np.linalg.norm(poses[:,3:],axis=1),1,atol=1e-4)
    out=HERE/f'entry_t{chapter}.npz'
    np.savez_compressed(out,root_pose=poses,root_com_velocity=vels,joint_pos=jpos,joint_vel=jvel,joint_names=np.array(names))
    manifest['banks'][str(chapter)]=dict(previous_chapter=previous,samples=len(poses),policy_time_range=seq['policy_time'][indices[[0,-1]]].tolist(),
        root_height_range=[float(poses[:,2].min()),float(poses[:,2].max())],max_joint_velocity=float(abs(jvel).max()),sha256=hashlib.sha256(out.read_bytes()).hexdigest())
(HERE/'entry_banks.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
