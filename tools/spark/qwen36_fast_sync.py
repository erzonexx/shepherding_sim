"""Own the retained NVFP4/MTP3 container for one NVIDIA Sync session."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time
from urllib.request import urlopen

NAME='codex-qwen36-35b-nvfp4-mtp3-20261015'
MODEL='nvidia/Qwen3.6-35B-A3B-NVFP4'
CMD_HASH='290a843f7992a91646e1c271ccef4df064d56e24485990d4aedf30edf79a9d61'
PORT=8002

def run(args,timeout=30):
    p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if p.returncode:raise RuntimeError('Failed: '+' '.join(args[:2]))
    return p.stdout

def info():
    v=json.loads(run(['docker','inspect',NAME]))[0]
    assert v['Config']['Labels'].get('codex.deployment')=='qwen36-nvfp4-20261015','Deployment mismatch'
    assert hashlib.sha256(json.dumps(v['Config']['Cmd']).encode()).hexdigest()==CMD_HASH,'Frozen model settings changed'
    assert not v['HostConfig'].get('AutoRemove'),'Auto-remove container rejected'
    assert v['HostConfig']['PortBindings']['8000/tcp']==[{'HostIp':'127.0.0.1','HostPort':str(PORT)}],'Unexpected API binding'
    return v

def healthy():
    try:
        with urlopen(f'http://127.0.0.1:{PORT}/health',timeout=2) as r:
            if r.status!=200:return False
        with urlopen(f'http://127.0.0.1:{PORT}/v1/models',timeout=2) as r:
            return MODEL in [v['id'] for v in json.load(r)['data']]
    except Exception:return False

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    v=info()
    if args.check:
        print(json.dumps({'model':MODEL,'mtp_tokens':3,'configuration_valid':True,'running':v['State']['Running'],'healthy':healthy()}),flush=True)
        return
    cache=Path.home()/'.cache/shepherding-model-launcher';cache.mkdir(exist_ok=True,parents=True)
    stopping=False
    def stop(*unused):
        nonlocal stopping
        stopping=True
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
    with (cache/'session.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another model session is active. Stop it before starting this entry.')
        if v['State']['Running']:raise RuntimeError('NVFP4 is already owned by another session; no changes made.')
        deadline=time.monotonic()+30
        while run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip():
            if time.monotonic()>deadline:raise RuntimeError('Another GPU task is active; no model was started.')
            if stopping:return
            time.sleep(1)
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))/1024**2
        assert available>=64,'Insufficient available memory; no model was started'
        owned=False
        try:
            if stopping:return
            owned=True
            run(['docker','start',v['Id']])
            print('Starting Qwen3.6 NVFP4 + MTP3. Loading may take several minutes.',flush=True)
            deadline=time.monotonic()+900;ready=False;next_update=0
            while not stopping:
                assert info()['State']['Running'],'Model container exited'
                if not ready:
                    ready=healthy()
                    if ready:print(f'READY {MODEL} | http://127.0.0.1:{PORT}/v1 | Stop with NVIDIA Sync.',flush=True)
                    elif time.monotonic()>deadline:raise RuntimeError('Model loading timed out')
                    elif time.monotonic()>next_update:
                        print('Loading model...',flush=True);next_update=time.monotonic()+30
                time.sleep(1)
        finally:
            if owned:
                print('Stopping NVFP4 + MTP3...',flush=True)
                run(['docker','stop','--time','20',v['Id']],timeout=45)
                print('STOPPED. Model weights and settings retained; FP8 is not restarted.',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('ERROR:',str(exc),flush=True)
        raise SystemExit(1)
