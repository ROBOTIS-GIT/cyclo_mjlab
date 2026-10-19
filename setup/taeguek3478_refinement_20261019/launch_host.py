from pathlib import Path
import subprocess,json
HERE=Path(__file__).resolve().parent
assert (HERE/'launch_validation.json').is_file()
assert not (HERE/'driver.started').exists()
uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
assert not any(line.split(',')[0].strip()==uuid for line in apps.splitlines()),apps
subprocess.run(['docker','exec','-d','-w','/workspace/cyclo_mjlab','cyclo_mjlab','bash','-c',
    'exec python -u setup/taeguek3478_refinement_20261019/train_sequence.py > setup/taeguek3478_refinement_20261019/driver.log 2>&1'],check=True)
print('DISPATCHED GPU0: 3 -> 4 -> 7 -> 8; 10000 additional updates each')
