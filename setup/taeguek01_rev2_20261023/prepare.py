"""Validate supplied Rev.2 URDF/MJCF and regenerate Taegeuk 1 FK/velocities."""
from pathlib import Path
import sys,json,hashlib,xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from scripts.tools.motion.csv_to_npz import (Motion,load_model,get_model_layout,
    save_motion_files,motion_to_qpos)
from source.assets.robots.robotis_k1.k1_rev2 import K1_REV2_XML_PATH


def main():
    description=ROOT/'third_party/ai_sapiens/ai_sapiens_description'
    urdf=description/'urdf/k1_rev2/k1.urdf'
    m=load_model(K1_REV2_XML_PATH);layout=get_model_layout(m)
    u=ET.parse(urdf).getroot();mass=0.;missing=[]
    for link in u.findall('link'):
        name=link.get('name');bi=m.body(name).id;it=link.find('inertial');origin=it.find('origin')
        mi=float(it.find('mass').get('value'));mass+=mi
        np.testing.assert_allclose(m.body_mass[bi],mi,atol=1e-9)
        np.testing.assert_allclose(m.body_ipos[bi],np.fromstring(origin.get('xyz'),sep=' '),atol=1e-9)
        a=it.find('inertia').attrib
        I=np.array([[float(a['ixx']),float(a['ixy']),float(a['ixz'])],
                    [float(a['ixy']),float(a['iyy']),float(a['iyz'])],
                    [float(a['ixz']),float(a['iyz']),float(a['izz'])]])
        R=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
        U=Rotation.from_quat(m.body_iquat[bi][[1,2,3,0]]).as_matrix()
        np.testing.assert_allclose(U@np.diag(m.body_inertia[bi])@U.T,R@I@R.T,atol=1e-8)
        ev=np.linalg.eigvalsh(I);assert ev.min()>0 and ev.max()<=ev.sum()-ev.max()+1e-9
        if not link.findall('collision'):missing.append(name)
    for j in u.findall('joint'):
        child=m.body(j.find('child').get('link')).id
        assert m.body(m.body_parentid[child]).name==j.find('parent').get('link')
        origin=j.find('origin');np.testing.assert_allclose(m.body_pos[child],np.fromstring(origin.get('xyz','0 0 0'),sep=' '),atol=1e-8)
        R=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
        np.testing.assert_allclose(R,Rotation.from_quat(m.body_quat[child][[1,2,3,0]]).as_matrix(),atol=1e-8)
        if j.get('type')!='fixed':
            ji=m.joint(j.get('name')).id
            np.testing.assert_allclose(m.jnt_axis[ji],np.fromstring(j.find('axis').get('xyz'),sep=' '),atol=1e-8)
            lim=j.find('limit');np.testing.assert_allclose(m.jnt_range[ji],[float(lim.get('lower')),float(lim.get('upper'))],atol=1e-8)
    assert np.isclose(mass,37.3424451) and m.nu==23 and m.nq==30
    source=ROOT/'source/assets/motions/K1_rev1/taeguek_01/k1_taeguek01_collision_only_50hz_converted.csv'
    rows=np.loadtxt(source,delimiter=',');assert rows.shape==(1582,30) and np.isfinite(rows).all()
    motion=Motion(fps=50.,root_pos=rows[:,:3],root_quat_wxyz=rows[:,[6,3,4,5]],joint_pos_csv_order=rows[:,7:])
    qpos=motion_to_qpos(m,layout,motion)
    ranges=m.jnt_range[layout.joint_ids];assert np.all(qpos[:,layout.joint_qpos_addresses]>=ranges[:,0]-1e-6) and np.all(qpos[:,layout.joint_qpos_addresses]<=ranges[:,1]+1e-6)
    md=mujoco.MjData(m);feet=[m.body(s+'_ankle_roll_link').id for s in ['left','right']]
    gids=[np.where((m.geom_bodyid==bi)&(m.geom_contype!=0))[0] for bi in feet]
    assert all(len(g)==9 for g in gids)
    bottoms=[];penetrations=[]
    for q in qpos:
        md.qpos[:]=q;mujoco.mj_forward(m,md)
        bottoms.append([float(np.min(md.geom_xpos[g,2]-m.geom_size[g,0])) for g in gids])
        penetrations.extend(float(c.dist) for c in md.contact if c.dist<-.001)
    bottoms=np.array(bottoms);assert bottoms.min()>-1e-5 and bottoms.min(axis=1).max()<.02
    out=ROOT/'source/assets/motions/K1_rev2/taeguek_01/k1_rev2_taeguek01_50hz'
    pe,qe=save_motion_files(m,layout,motion,out.with_suffix('.npz'),out.with_suffix('.csv'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assets={str(p.relative_to(ROOT)):sha(p) for folder in ['urdf','meshes','mujoco'] for p in (description/folder/'k1_rev2').rglob('*') if p.is_file()}
    assets.update({str(p.relative_to(ROOT)):sha(p) for p in [source,out.with_suffix('.npz'),out.with_suffix('.csv')]})
    report={'robot':'K1 Rev.2','urdf_mjcf_mass_kg':mass,'links':m.nbody-1,'actuated_joints':m.nu,'collision_shapes':int(((m.geom_contype!=0)|(m.geom_conaffinity!=0)).sum()),'links_without_collision':missing,'frames':len(rows),'duration_s':(len(rows)-1)/50,'fps':50,'source_motion':str(source.relative_to(ROOT)),'motion_adaptation':'Identical joint/root trajectory; Rev.2 FK and all body velocities regenerated. Rev.2 knee lateral offsets cancel at feet. Not a new human-motion retarget.','minimum_foot_bottom_m':bottoms.min(0).tolist(),'max_lowest_foot_bottom_m':float(bottoms.min(1).max()),'self_contact_penetration_over_1mm_samples':len(penetrations),'minimum_self_contact_distance_m':min(penetrations,default=0.),'fk_roundtrip_position_error_m':pe,'fk_roundtrip_rotation_error_rad':qe,'hashes':assets}
    (HERE/'inputs.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='hashes'},indent=2))

if __name__=='__main__':main()
