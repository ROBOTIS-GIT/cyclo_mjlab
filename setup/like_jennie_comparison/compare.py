"""Compare two evaluate.py outputs that used the same reference and model."""
from pathlib import Path
import argparse,json
import numpy as np

def digest(report,key):return report['sha256'][report[key]]
def summarize(data,lo,hi,fps):
    p=data['qpos'];j=data['joint_error'];foot=data['foot_error'];a=int(lo*fps);b=min(int(hi*fps),len(p))
    c=data['contact_samples'];c=c[(c[:,0]>=lo)&(c[:,0]<hi)]
    if a>=b or not len(c):raise ValueError('Empty comparison window')
    loaded=c[:,1:3]>30.;slide=loaded&(c[:,3:5]>.15)
    return {'joint_rmse_rad':float(np.sqrt(np.mean(j[a:b]**2))),'arm_rmse_rad':float(np.sqrt(np.mean(j[a:b,13:]**2))),'foot_rmse_m':float(np.sqrt(np.mean(foot[a:b]**2))),'min_pelvis_height_m':float(p[a:b,2].min()),'nonfoot_contact_peak_n':float(c[:,5].max()),'foot_peak_n':np.max(c[:,1:3],axis=0).tolist(),'loaded_contact_point_speed_over_0_15_fraction':float(slide.any(axis=1).mean())}

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r1=json.loads((a.baseline/'report.json').read_text());r2=json.loads((a.candidate/'report.json').read_text())
    for key in ('motion_csv','motion_npz','model'):
        if digest(r1,key)!=digest(r2,key):raise ValueError(f'{key} mismatch; evaluate both policies with the same reference and model')
    for key in ('fps','physics_dt','mode','frames','failure'):
        if r1[key]!=r2[key]:raise ValueError(f'{key} differs between runs')
    if r1['failure'] is not None or r1['frames']!=r1['requested_frames'] or r2['frames']!=r2['requested_frames']:raise ValueError('Incomplete or failed rollout')
    # XML include (k1.xml) must match too, regardless of checkout path.
    for key in ('k1.xml',):
        hashes=[[v for k,v in r['sha256'].items() if Path(k).name==key] for r in (r1,r2)]
        if len(hashes[0])!=1 or hashes[0]!=hashes[1]:raise ValueError(f'{key} hash missing or differs')
    z1=np.load(a.baseline/'rollout.npz');z2=np.load(a.candidate/'rollout.npz');end=(r1['frames']-1)/r1['fps']
    windows={'full':(0,end),'early':(0,3),'middle':(3,40),'late':(40,end)}
    result={'baseline':str(a.baseline.resolve()),'candidate':str(a.candidate.resolve()),'windows':{name:{'seconds':[lo,hi],'baseline':summarize(z1,lo,hi,r1['fps']),'candidate':summarize(z2,lo,hi,r1['fps'])} for name,(lo,hi) in windows.items()},'caveat':'One nominal deterministic rollout; contact speed is instantaneous, not measured slip; lower pelvis height can be an intended pose.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
