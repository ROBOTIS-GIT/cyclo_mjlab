"""Periodic robustness evaluation, nominal regression guard, candidate retention."""
from pathlib import Path
import math,json,shutil,subprocess,sys,hashlib
import torch
from .limbo_prepare_obs124_runner import Obs124Runner
ROOT=Path(__file__).resolve().parents[3]


class RobustRunner(Obs124Runner):
    def load(self,*args,**kwargs):
        result=super().load(*args,**kwargs)
        cfg=kwargs.get('load_cfg')
        if cfg is not None and not cfg.get('optimizer',False):
            # A warm start must not silently restore the V5 fixed std of .02.
            for name,p in self.alg.actor.distribution.named_parameters():
                with torch.no_grad():
                    if name in ('log_std','log_std_param'):p.fill_(math.log(.08))
                    elif name in ('std','std_param'):p.fill_(.08)
                    else:raise ValueError(f'Unexpected Gaussian parameter: {name}')
        return result

    def save(self,path,infos=None):
        super().save(path,infos)
        step=self.current_learning_iteration
        if step==0 or (step%1000 and step!=self.cfg['max_iterations']-1):return
        checkpoint=Path(path);out=checkpoint.parent/'robust_validation'/f'iteration_{step}';out.mkdir(parents=True,exist_ok=True)
        args=[sys.executable,str(ROOT/'scripts/reinforcement_learning/evaluate_limbo_robust.py'),'--checkpoint',str(checkpoint),'--output',str(out)]
        with (out/'evaluation.log').open('w') as log:
            r=subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=360)
        if r.returncode:raise RuntimeError(f'Robust evaluator failed; artifacts retained at {out}')
        reports=json.loads((out/'evaluation.json').read_text())
        if len(reports)!=4 or {r['profile'] for r in reports}!={'nominal','runtime_entry','heldout_medium','heldout_full'}:
            raise RuntimeError('Incomplete robustness profiles; no candidate promoted')
        for report in reports:
            expected=32 if report['profile'].startswith('heldout') else 65
            if report['num_cases']!=expected or len(report['cases'])!=expected or not 0<=report['successes']<=expected:
                raise RuntimeError('Invalid evaluation counts; no candidate promoted')
            if report['success_rate']!=report['successes']/expected or not math.isfinite(report['first2s_peak_raw_action']):
                raise RuntimeError('Invalid evaluation metrics; no candidate promoted')
        nominal=next(r for r in reports if r['profile']=='nominal')
        stress=[r for r in reports if r['profile']!='nominal']
        rates=[r['success_rate'] for r in stress]
        score=[min(rates),sum(rates)/len(rates),-max(r['first2s_peak_raw_action'] for r in reports)]
        best=checkpoint.parent/'best_robust.json'
        previous=json.loads(best.read_text()) if best.exists() else None
        selected=nominal['success_rate']>=.95 and (previous is None or tuple(score)>tuple(previous['score']))
        record={'iteration':step,'checkpoint':str(checkpoint),'score':score,'nominal_rate':nominal['success_rate'],
            'profiles':[{k:v for k,v in r.items() if k!='cases'} for r in reports],
            'selected_candidate':selected,'deployment_gate_passed':nominal['success_rate']==1. and all(v>=.95 for v in rates)}
        if selected:
            target=checkpoint.parent/'best_robust.pt';temp=target.with_suffix('.tmp');shutil.copy2(checkpoint,temp);temp.replace(target)
            record['sha256']=hashlib.sha256(target.read_bytes()).hexdigest();best.write_text(json.dumps(record,indent=2))
        failure=checkpoint.parent/'nominal_regression.json'
        count=json.loads(failure.read_text())['consecutive'] if failure.exists() else 0
        count=count+1 if step>=2000 and nominal['success_rate']<.80 else 0
        failure.write_text(json.dumps({'iteration':step,'consecutive':count}))
        record['nominal_regression_count']=count
        (out/'guard.json').write_text(json.dumps(record,indent=2));print('ROBUST_VALIDATION',json.dumps(record),flush=True)
        if count>=2:raise RuntimeError('Two consecutive severe nominal regressions; stopped, previous candidates preserved')
