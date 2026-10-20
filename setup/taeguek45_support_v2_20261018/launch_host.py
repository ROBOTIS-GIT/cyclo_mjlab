"""Launch once only when GPU 0 is unused; no unrelated process is stopped."""
from pathlib import Path
import subprocess,json
root=Path('/home/ubuntu/cyclo_mjlab')
setup=root/'setup/taeguek45_support_v2_20261018'
assert (setup/'launch_validation.json').is_file()
assert not (setup/'driver.started').exists()
uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
assert not any(line.split(',')[0].strip()==uuid for line in apps.splitlines()),apps
subprocess.run(['docker','exec','-d','-w','/workspace/cyclo_mjlab','cyclo_mjlab',
  'bash','-c','exec bash setup/taeguek45_support_v2_20261018/train_sequence.sh > setup/taeguek45_support_v2_20261018/driver.log 2>&1'],check=True)
print('DISPATCHED GPU0 chapter04->05 additional_updates=10000 each')
