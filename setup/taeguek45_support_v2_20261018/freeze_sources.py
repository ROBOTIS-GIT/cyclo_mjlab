from pathlib import Path
import hashlib,json,shutil
root=Path('/home/ubuntu/cyclo_mjlab');audit=root/'setup/taeguek45_support_v2_20261018'
manifest=json.loads((audit/'source_manifest.json').read_text())
for relative,old_hash in list(manifest.items()):
    target=root/relative
    if relative!='tests/test_taeguek_support_v2.py':
        assert hashlib.sha256(target.read_bytes()).hexdigest()==old_hash,relative
    destination=audit/'source_snapshot'/relative
    destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,destination)
    manifest[relative]=hashlib.sha256(target.read_bytes()).hexdigest()
for name in ['train_sequence.sh','run_smoke.sh','validate_launch.py','export_final.py','export_snapshot.py']:
    p=audit/name
    manifest[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
(audit/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
print('FROZEN',len(manifest),'files')
