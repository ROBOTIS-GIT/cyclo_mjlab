"""Export each completed model with matching CSV and validated ONNX/JIT."""
from pathlib import Path
import sys,runpy,json
chapter=int(sys.argv[1]);root=Path('/workspace/cyclo_mjlab')
audit=root/'setup/taeguek45_support_v2_20261018'
runs=list((root/'logs/rsl_rl/k1_mimic').glob(f'*taeguek{chapter:02d}_support_v2_finetune_20261018_support_v2'))
assert len(runs)==1,runs
run=runs[0];assert (run/'model_9999.pt').is_file()
out=audit/'exports'/f't{chapter}_support_v2_9999'
out.parent.mkdir(exist_ok=True)
sys.argv=['export_snapshot.py',str(run),str(out),'model_9999.pt']
runpy.run_path(str(audit/'export_snapshot.py'),run_name='__main__')
(audit/f'chapter_{chapter:02d}.complete.json').write_text(json.dumps(dict(run=str(run),export=str(out)),indent=2))
print(f'EXPORT_COMPLETE chapter={chapter} path={out}',flush=True)
