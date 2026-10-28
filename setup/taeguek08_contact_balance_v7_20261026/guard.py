"""Warn on sustained training-metric regression without stopping; not a sim2sim quality certificate."""
import math
KEYS=('arm','hand','tilt','joint','length')

def update_guard(state,iteration,values):
    s=dict(state)
    if iteration<=s.get('last_iteration',-1):return s
    if set(values)!=set(KEYS) or not all(math.isfinite(v) for v in values.values()):
        return {**s,'last_iteration':iteration,'halt':True,'reasons':['invalid_metrics']}
    if iteration>=250 and 'baseline' not in s:s['baseline']=dict(values)
    reasons=[];base=s.get('baseline',values)
    limits={
        'arm':min(.14,1.15*base['arm']+.003),
        'hand':min(.045,1.2*base['hand']+.002),
        'tilt':min(.12,1.2*base['tilt']+.005),
        'joint':1.2*base['joint'],
        'length':max(1500,.8*base['length']),
    }
    if iteration>=1000:
        for k in ('arm','hand','tilt','joint'):
            if values[k]>limits[k]:reasons.append(k)
        if values['length']<limits['length']:reasons.append('length')
    bad=s.get('bad_checks',0)+1 if reasons else 0
    s.update(last_iteration=iteration,values=dict(values),limits=limits,reasons=reasons,bad_checks=bad,warning=bad>=2,halt=False)
    return s
