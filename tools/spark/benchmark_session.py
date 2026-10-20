"""A bounded benchmark lease of the retained fast model; never restores FP8."""
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid
from urllib.request import urlopen

MODEL = 'nvidia/Qwen3.6-35B-A3B-NVFP4'
FAST = 'codex-qwen36-35b-nvfp4-mtp3-20261015'
OLD = 'codex-qwen36-35b-fp8-20261015'
RECIPE = '290a843f7992a91646e1c271ccef4df064d56e24485990d4aedf30edf79a9d61'
SSH = ['ssh', '-4', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
       '-o', 'Hostname=192.168.11.24', '-o', 'HostKeyAlias=spark-9572.local', 'spark_csslab']


def write(path, record):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(record, indent=2), encoding='utf-8')
    temporary.replace(path)


def inspect(name):
    return json.loads(subprocess.check_output(['docker', 'inspect', name], timeout=20))[0]


def command_hash(container):
    return hashlib.sha256(json.dumps(container['Config']['Cmd']).encode()).hexdigest()


def healthy():
    try:
        with urlopen('http://127.0.0.1:8002/health', timeout=2) as response:
            if response.status != 200:
                return False
        with urlopen('http://127.0.0.1:8002/v1/models', timeout=2) as response:
            return [v['id'] for v in json.load(response)['data']] == [MODEL]
    except Exception:
        return False


def remote_lease(folder):
    import fcntl
    folder = Path(folder).resolve()
    assert folder.parent == Path.home()/'shepherding/experiments'
    fast, old = inspect(FAST), inspect(OLD)
    assert not old['State']['Running'], 'FP8 is running; preserve state and refuse overlap'
    assert command_hash(fast) == RECIPE and not fast['HostConfig'].get('AutoRemove')
    assert fast['HostConfig']['PortBindings']['8000/tcp'] == [{'HostIp':'127.0.0.1','HostPort':'8002'}]
    before = {name:dict(running=v['State']['Running'], command_sha256=command_hash(v))
              for name, v in ((FAST, fast), (OLD, old))}
    write(folder/'before.json', before)
    borrowed, owned, lock = fast['State']['Running'], False, None
    try:
        if not borrowed:
            lock_path = Path.home()/'.cache/shepherding-model-launcher/session.lock'
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock = lock_path.open('a')
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            assert not subprocess.check_output(
                ['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True, timeout=20).strip()
            available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                                 if x.startswith('MemAvailable:')))/1024**2
            assert available > 64, 'Insufficient available memory'
            owned = True
            subprocess.run(['docker','start',FAST], check=True, capture_output=True, timeout=30)
        deadline = time.monotonic()+900
        while not healthy():
            assert inspect(FAST)['State']['Running'] and time.monotonic()<deadline, 'Model loading failed'
            if (folder/'finish').exists():
                return
            time.sleep(2)
        cmd = fast['Config']['Cmd']
        flags = ('--max-num-seqs','--max-num-batched-tokens','--max-model-len','--speculative-config')
        write(folder/'ready.json', dict(model=MODEL, command_sha256=RECIPE, owned=owned, borrowed=borrowed,
            healthy=True, serving_flags={key:cmd[cmd.index(key)+1] for key in flags},
            meminfo=Path('/proc/meminfo').read_text().splitlines()[:3]))
        deadline = time.monotonic()+3600
        while not (folder/'finish').exists():
            assert inspect(FAST)['State']['Running'] and not inspect(OLD)['State']['Running']
            assert time.monotonic()<deadline, 'Benchmark lease expired'
            time.sleep(2)
    except BaseException as exc:
        write(folder/'error.json', dict(type=type(exc).__name__, reason=str(exc)[-1000:]))
        raise
    finally:
        if owned and inspect(FAST)['State']['Running']:
            subprocess.run(['docker','stop','--time','20',FAST], check=True, capture_output=True, timeout=45)
        if lock:
            lock.close()
        after = {name:dict(running=inspect(name)['State']['Running'], command_sha256=command_hash(inspect(name)))
                 for name in (FAST, OLD)}
        assert all(after[name] == before[name] for name in before), 'Prior model state not restored'
        if borrowed:
            assert healthy()
        write(folder/'cleanup.json', dict(state_restored=True, fp8_started=False, settings_unchanged=True,
                                         before=before, after=after, owned_model_stopped=owned))


class ModelSession:
    def __init__(self, output, name, port):
        self.output, self.port = Path(output), int(port)
        self.remote = '/home/csslab/shepherding/experiments/'+name
        self.token, self.tunnel = uuid.uuid4().hex, None
        self.started = False

    @property
    def base_url(self):
        return f'http://127.0.0.1:{self.port}/v1'

    def ssh(self, source):
        process = subprocess.run(SSH+['python3','-'], input=source, text=True, encoding='utf-8',
                                 capture_output=True, timeout=45)
        if process.returncode:
            raise RuntimeError('Spark action failed: '+process.stderr[-1200:])
        return process.stdout

    def status(self):
        return json.loads(self.ssh(
            'import json\nfrom pathlib import Path\nr=Path('+repr(self.remote)+')\n'
            'assert (r/"owner_token").read_text()=='+repr(self.token)+'\n'
            'print(json.dumps({n:json.loads((r/n).read_text()) for n in '
            '("ready.json","error.json","cleanup.json") if (r/n).exists()}))\n'))

    def __enter__(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', self.port))
        payload = base64.b64encode(Path(__file__).read_bytes()).decode()
        try:
            self.ssh('import base64,subprocess\nfrom pathlib import Path\nr=Path('+repr(self.remote)+')\n'
                'assert not r.exists(),"Existing lease must not be replaced"\nr.mkdir(parents=True)\n'
                '(r/"owner_token").write_text('+repr(self.token)+')\n'
                '(r/"lease.py").write_bytes(base64.b64decode('+repr(payload)+'))\n'
                'with (r/"lease.log").open("w") as log:\n'
                ' p=subprocess.Popen(["python3","-u",str(r/"lease.py"),"--lease",str(r)],'
                'stdout=log,stderr=log,start_new_session=True)\n'
                '(r/"lease.pid").write_text(str(p.pid))\n')
            self.started = True
            self.tunnel = subprocess.Popen(
                SSH[:-1]+['-N','-o','ExitOnForwardFailure=yes','-L',
                          f'127.0.0.1:{self.port}:127.0.0.1:8002',SSH[-1]],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            deadline, last = time.monotonic()+960, 0
            while True:
                assert self.tunnel.poll() is None, 'API tunnel exited'
                state = self.status()
                assert 'error.json' not in state, str(state.get('error.json'))
                if 'ready.json' in state:
                    with urlopen(self.base_url+'/models', timeout=5) as response:
                        assert [v['id'] for v in json.load(response)['data']] == [MODEL]
                    write(self.output/'model_identity.json', dict(
                        exact_retained_recipe_verified=True, ready=state['ready.json']))
                    print('Exact NVFP4 MTP3 model ready; one model instance.', flush=True)
                    return self
                assert time.monotonic()<deadline, 'Model readiness timed out'
                if time.monotonic()-last>30:
                    print(json.dumps(dict(event='model_loading')), flush=True)
                    last = time.monotonic()
                time.sleep(3)
        except BaseException:
            if self.started:
                self.close()
            raise

    def close(self):
        try:
            if self.started:
                self.ssh('from pathlib import Path\nr=Path('+repr(self.remote)+')\n'
                    'assert (r/"owner_token").read_text()=='+repr(self.token)+'\n(r/"finish").touch()\n')
                deadline = time.monotonic()+120
                while True:
                    state = self.status()
                    if 'cleanup.json' in state:
                        write(self.output/'cleanup.json', state['cleanup.json'])
                        assert state['cleanup.json']['state_restored'] and not state['cleanup.json']['fp8_started']
                        print('Original model state restored; FP8 was not started.', flush=True)
                        break
                    assert time.monotonic()<deadline, 'Model cleanup requires inspection'
                    time.sleep(2)
                self.started = False
        except BaseException as exc:
            write(self.output/'cleanup_failure.json', dict(type=type(exc).__name__, reason=str(exc)[-1200:]))
            raise
        finally:
            if self.tunnel is not None:
                self.tunnel.terminate()
                try:
                    self.tunnel.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.tunnel.kill()
                    self.tunnel.wait(timeout=5)
                self.tunnel = None

    def __exit__(self, *unused):
        self.close()


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--lease':
        raise SystemExit('This module is used by benchmark managers; it is not a model launcher shortcut.')
    remote_lease(sys.argv[2])
