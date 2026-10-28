"""CPU-only checks: reject incomplete evaluation and preserve candidates on regression."""
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import sys,json,tempfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from source.tasks.mimic.limbo_robust_runner import RobustRunner
from source.tasks.mimic.limbo_prepare_obs124_runner import Obs124Runner


def verify():
    out=ROOT/'logs/limbo_verification/guard';out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        folder=Path(tmp);best=folder/'best_robust.pt';best.write_bytes(b'protected candidate')
        (folder/'best_robust.json').write_text(json.dumps({'score':[.9,.95,-3.]}))
        runner=object.__new__(RobustRunner);runner.cfg={'max_iterations':20000};runner.current_learning_iteration=2000
        state={'incomplete':False}
        def fake_evaluation(args,**kwargs):
            dest=Path(args[args.index('--output')+1]);rows=[]
            for profile in ['nominal','runtime_entry','heldout_medium','heldout_full']:
                n=32 if profile.startswith('heldout') else 65
                rows.append({'profile':profile,'num_cases':n,'cases':[{}]*n,'successes':0,'success_rate':0.,'first2s_peak_raw_action':2.})
            (dest/'evaluation.json').write_text(json.dumps(rows[:1] if state['incomplete'] else rows))
            return SimpleNamespace(returncode=0)
        with patch.object(Obs124Runner,'save',return_value=None),patch('source.tasks.mimic.limbo_robust_runner.subprocess.run',side_effect=fake_evaluation):
            runner.save(str(folder/'model_2000.pt'))
            assert json.loads((folder/'nominal_regression.json').read_text())['consecutive']==1
            runner.current_learning_iteration=3000
            try:runner.save(str(folder/'model_3000.pt'))
            except RuntimeError as e:assert 'Two consecutive' in str(e)
            else:raise AssertionError('Severe nominal regression did not stop training')
            assert best.read_bytes()==b'protected candidate'
            state['incomplete']=True;runner.current_learning_iteration=4000
            try:runner.save(str(folder/'model_4000.pt'))
            except RuntimeError as e:assert 'Incomplete' in str(e)
            else:raise AssertionError('Incomplete validation was accepted')
            assert best.read_bytes()==b'protected candidate'
    report={'all_pass':True,'two_severe_regressions_stop':True,'incomplete_evaluation_rejected':True,'previous_candidate_preserved':True}
    (out/'verification.json').write_text(json.dumps(report,indent=2));print('ROBUST_GUARD_PASS',json.dumps(report))

if __name__=='__main__':verify()
