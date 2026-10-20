"""Host-only, explicit --apply: stop only the two recorded old queues and their children.

Default is read-only. PID start-time/command fingerprints prevent PID-reuse errors.
"""
from pathlib import Path
import argparse,hashlib,json,os,signal,time
HERE=Path(__file__).resolve().parent

def proc(pid):
    try:
        p=Path(f'/proc/{pid}');raw=(p/'stat').read_text().rsplit(')',1)[1].split()
        command=(p/'cmdline').read_bytes()
        return dict(pid=pid,ppid=int(raw[1]),state=raw[0],start=raw[19],
            command=command.replace(b'\0',b' ').decode(),sha256=hashlib.sha256(command).hexdigest())
    except (FileNotFoundError,ProcessLookupError,PermissionError):return None

def same(saved):
    now=proc(saved['pid'])
    return now if now and now['start']==saved['start'] and now['sha256']==saved['sha256'] and now['state']!='Z' else None

def descendants(parents):
    rows=[proc(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()]
    ids={p['pid'] for p in parents};children=[]
    changed=True
    while changed:
        changed=False
        for p in rows:
            if p and p['ppid'] in ids and p['pid'] not in ids:
                children.append(p);ids.add(p['pid']);changed=True
    return children

def send(p,sig):
    if same(p):
        try:os.kill(p['pid'],sig)
        except ProcessLookupError:pass

def wait(rows,seconds):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        live=[p for p in rows if same(p)]
        if not live:return []
        time.sleep(.2)
    return [p for p in rows if same(p)]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    if Path('/.dockerenv').exists():
        raise SystemExit('Run this script on the Ubuntu host, not inside Docker.')
    if args.apply and os.geteuid()!=0:
        raise SystemExit('Docker training processes require root: sudo python3 '+str(HERE/'manual_stop.py')+' --apply')
    saved=json.loads((HERE/'manual_stop_targets.json').read_text())
    parents=[p for p in saved['parents'] if same(p)]
    children={p['pid']:p for p in descendants(parents)}
    children.update({p['pid']:p for p in saved['children'] if same(p)})
    print(json.dumps(dict(apply=args.apply,parents=parents,children=list(children.values())),indent=2),flush=True)
    if not args.apply:return
    try:
        # Freeze dispatchers first so neither can start the next chapter.
        for p in parents:send(p,signal.SIGSTOP)
        children.update({p['pid']:p for p in descendants(parents)})
        rows=list(children.values())
        # Keep tee alive while Python writes its interruption traceback.
        loggers=[p for p in rows if p['command'].split()[0]=='tee']
        workers=[p for p in rows if p not in loggers]
        for p in workers:send(p,signal.SIGINT)
        live=wait(workers,15)
        for p in live:send(p,signal.SIGTERM)
        live=wait(live,10)
        for p in loggers:send(p,signal.SIGTERM)
        live+=wait(loggers,2)
    finally:
        for p in parents:
            send(p,signal.SIGTERM);send(p,signal.SIGCONT)
        parent_live=wait(parents,5)
    record=dict(stopped_at=time.time(),remaining=[p['pid'] for p in live+parent_live])
    (HERE/'manual_stop_result.json').write_text(json.dumps(record,indent=2))
    if record['remaining']:raise RuntimeError(f'Some recorded processes are still running: {record}')
    print('OLD_QUEUES_STOPPED. Checkpoints preserved. Ready for manual GPU0/GPU1 runs.',flush=True)
if __name__=='__main__':main()
