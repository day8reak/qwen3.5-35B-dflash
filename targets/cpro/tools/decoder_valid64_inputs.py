"""Keep the nine files needed to replay a CPro valid64 input on the golden host."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

FILES = ('inputs.json', 'replay/result.json', 'replay/io.json', 'replay/cpro-valid64.json',
         'input_ids.bin', 'attention_mask.bin', 'effective_length.bin', 'allQLen.bin', 'output_pos.bin')


def endpoint():
    config = Path(os.getenv('CPRO_TRANSFER_CONFIG') or '~/.config/qwen35-cpro/transfer.json').expanduser()
    if not config.exists():
        return config, {'host': 'omc'}
    if config.stat().st_mode & 0o077:
        raise ValueError(f'Use mode 600 for the SSH settings in {config}')
    return config, json.loads(config.read_bytes())['omc']


def fetch(source, dest):
    config, remote = endpoint()
    if config.is_file() and importlib.util.find_spec('paramiko') is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
        from pull_artifacts import connection, download
        with connection(config, 'omc') as client, client.open_sftp() as sftp:
            for name in FILES:
                download(sftp, str(source/name), dest/name)
        return

    host = remote['host']
    options = ['-P', str(int(remote['port']))] if 'port' in remote else []
    if remote.get('user'):
        options += ['-o', 'User=' + remote['user']]
    # One OpenSSH connection, with normal agent/key or terminal authentication.
    # Remote paths in sftp commands may be quoted. Globs are literal here.
    def quoted(path):
        name = str(path)
        if any(c in name for c in ('\n', '\r', '\0')):
            raise ValueError('Expected a single-line replay path')
        return '"' + ''.join(('\\' if c in '\\"*?[]' else '') + c for c in name) + '"'
    batch = dest/'sftp.batch'
    batch.write_text(''.join(f'get {quoted(source/name)} {quoted(dest/name)}\n' for name in FILES))
    print('[CPU golden] Receiving the saved inputs with OpenSSH sftp', flush=True)
    subprocess.run(['sftp', '-o', 'BatchMode=no', *options, '-b', str(batch), '--', host], check=True)
    batch.unlink()


def complete(folder):
    try:
        sizes = {'input_ids.bin': 256, 'attention_mask.bin': 64*2048*4, 'effective_length.bin': 2,
                 'allQLen.bin': 8, 'output_pos.bin': 4}
        if any((folder/name).stat().st_size != size for name, size in sizes.items()):
            return False
        plan, result, io, summary = [json.loads((folder/name).read_bytes()) for name in FILES if name.endswith('.json')]
        return (result['status'] == 'EXECUTED' and result['exitcode'] == 0 and result['board_case'] == '40-prefill-t64'
                and result['attempt'] == summary['attempt'] and plan['source_attempt'] == summary['source_attempt']
                and summary['model']['sha256'] == result['omc_sha256']
                and (summary['valid_rows'], summary['output_pos']) == (64, 63)
                and result['input_hashes'] == [r['sha256'] for r in io['inputs']]
                and plan['valid_rows'] == 64 and plan['output_pos'] == 63 and len(plan['token_ids']) == 64)
    except (ValueError, OSError, KeyError, TypeError):
        return False


def local_plan(cache, plan):
    plan = plan.expanduser().absolute()
    if plan.name != 'inputs.json':
        raise ValueError('Select inputs.json from the successful 40-layer valid64 replay')
    if plan.is_file():
        return plan
    # Valid64 attempt paths are immutable on the board host.
    _, remote = endpoint()
    origin = [remote['host'], remote.get('port', 22), remote.get('user'), str(plan)]
    key = hashlib.sha256(json.dumps(origin).encode()).hexdigest()
    parent = cache/'replays'
    parent.mkdir(parents=True, exist_ok=True)
    saved = parent/key
    with (parent/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if complete(saved):
            return saved/'inputs.json'
        with tempfile.TemporaryDirectory(prefix='.receive-', dir=parent) as temporary:
            incoming = Path(temporary)/'inputs'
            (incoming/'replay').mkdir(parents=True)
            cmd = [sys.executable, str(Path(__file__).resolve()), str(plan.parent), str(incoming)]
            env = os.getenv('DECODER_TRANSFER_ENV')
            if env and env not in ('current', os.getenv('CONDA_DEFAULT_ENV'), os.getenv('CONDA_PREFIX')):
                cmd = [os.getenv('CONDA_EXE') or 'conda', 'run', '--no-capture-output',
                       '-p' if os.path.isabs(env) else '-n', env, 'python', *cmd[1:]]
            print(f'[CPU golden] {plan.parent} → {saved}', flush=True)
            subprocess.run(cmd, check=True)
            if not complete(incoming):
                raise ValueError(f'{plan}: incomplete or inconsistent 40-layer valid64 input bundle')
            if saved.exists():
                saved.rename(Path(tempfile.mkdtemp(prefix='incomplete-', dir=parent))/key)
            incoming.rename(saved)
    return saved/'inputs.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('dest', type=Path)
    args = parser.parse_args()
    fetch(args.source, args.dest)
