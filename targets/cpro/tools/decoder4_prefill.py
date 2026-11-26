#!/usr/bin/env python3
"""Review, export, compile and run the complete L0–L3 T64 decoder prefix."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'model'))


def ssh_file(client, remote, local, record=None):
    """Stream one file on an SSH exec channel and verify the exporter's receipt."""
    with client.get_transport().open_session(timeout=20) as channel:
        channel.settimeout(120)
        channel.exec_command('exec cat -- ' + shlex.quote(str(remote)))
        channel.shutdown_write()
        digest = hashlib.sha256()
        with local.open('xb') as dst:
            for chunk in iter(lambda: channel.recv(1 << 20), b''):
                dst.write(chunk)
                digest.update(chunk)
        if channel.recv_exit_status() != 0:
            error = channel.recv_stderr(4096).decode(errors='replace').strip()
            raise OSError(f'Could not read {remote} via SSH: {error}')
    if record is not None and (local.stat().st_size != record['bytes'] or digest.hexdigest() != record['sha256']):
        raise ValueError(f'Size or SHA-256 differs from the AIR manifest: {remote}')


def ssh_air(client, remote, staged):
    remote = Path(remote)
    staged.mkdir()
    manifest = staged / 'manifest.json'
    ssh_file(client, remote / manifest.name, manifest)
    meta = json.loads(manifest.read_bytes())
    if meta['kind'] != 'single-cpro-l0-l3' or meta['layers'] != [0, 1, 2, 3] or not meta['weights']:
        raise ValueError(f'Expected an exported L0–L3 AIR bundle at {remote}')
    records = [meta['air'], *meta['weights'], *meta['inputs'], meta['golden']]
    print(f"SSH transfer: {len(records)} files, {sum(r['bytes'] for r in records):,} bytes", flush=True)
    for record in records:
        rel = Path(record['path'])
        if rel.is_absolute() or '..' in rel.parts or not rel.parts:
            raise ValueError(f'Invalid path in AIR manifest: {rel}')
        local = staged / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        ssh_file(client, remote / rel, local, record)


def pull_air(run, remote, *, ssh_download=ssh_air, destination_name='air', before_download=None, sftp_download=None):
    import paramiko
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from pull_artifacts import DEFAULT_CONFIG, connection, pull

    if not remote.is_absolute() or '..' in remote.parts:
        raise ValueError(f'Expected an absolute AIR directory on the export host: {remote}')
    with connection(DEFAULT_CONFIG, 'air') as client:
        try:
            sftp = client.open_sftp()
        except (EOFError, paramiko.SSHException) as error:
            if not isinstance(error, EOFError) and str(error) != 'EOF during negotiation':
                raise
            print('SFTP closed during negotiation. Retrying over a regular SSH exec channel.', flush=True)
        else:
            with sftp:
                sftp.get_channel().settimeout(120)
                if before_download is not None:
                    with sftp.open(str(remote / 'manifest.json'), 'rb') as manifest:
                        before_download(json.load(manifest), run)
                if sftp_download is None:
                    pull(sftp, run / destination_name, remote)
                else:
                    pull(sftp, run / destination_name, remote, download_to=sftp_download)
            return
    with connection(DEFAULT_CONFIG, 'air') as client:
        pull(client, run / destination_name, remote, download_to=ssh_download)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='action', required=True)
    review = stages.add_parser('review')
    review.add_argument('--output', type=Path, required=True)
    air = stages.add_parser('air')
    for stage in (air, review):
        stage.add_argument('--source-root', type=Path, required=True)
    air.add_argument('--config', type=Path, required=True)
    air.add_argument('--reference', type=Path, required=True)
    air.add_argument('--prepared-source', type=Path, help='export these reviewed source files directly')
    transfer = stages.add_parser('pull', help='fetch the exported bundle from the AIR host')
    transfer.add_argument('--remote-air', type=Path, required=True)
    build = stages.add_parser('compile')
    build.add_argument('--ddk-root', type=Path, required=True)
    build.add_argument('--gdr-source', type=Path, required=True)
    build.add_argument('--tbe-diagnosis', type=Path)
    board = stages.add_parser('board')
    board.add_argument('--runtime-config', type=Path, help='explicitly run the separate HIAI slot probe')
    for stage in (air, transfer, build, board):
        stage.add_argument('--run', type=Path, required=True)
    a = parser.parse_args()
    from qwen35_moe_dflash.common.artifacts import output_path
    if a.action == 'review':
        from qwen35_moe_dflash.target.layer_probe.common import new_output
        from qwen35_moe_dflash.target.prefill_t64.decoder_four_source import stage
        out = new_output(a.output)
        stage(a.source_root.resolve(), out)
        print(out)
        return 0
    run = output_path(a.run)
    if a.action == 'pull':
        pull_air(run, a.remote_air)
        return 0
    if a.action == 'air':
        from qwen35_moe_dflash.target.prefill_t64.whole_export import export
        export(a.source_root.resolve(), a.config, a.reference, run / 'air',
               four_decoders=True, prepared_source=a.prepared_source)
        return 0
    from qwen35_moe_dflash.target.prefill_t64 import whole_run
    from qwen35_moe_dflash.target.prefill_t64.decoder_pair import load
    meta = load(run / 'air/manifest.json')
    if meta['kind'] != whole_run._FOUR_DECODERS or meta['layers'] != [0, 1, 2, 3]:
        raise ValueError('Expected one complete L0–L3 decoder AIR')
    if a.action == 'compile':
        whole_run.compile_model(run, a.ddk_root.resolve(), a.gdr_source.resolve(), a.tbe_diagnosis)
        return 0
    from qwen35_moe_dflash.target.board_tool_inspect import DEFAULT_TOOL
    from qwen35_moe_dflash.target.prefill_t64.chain_hilog import diagnose_decoder_pair
    case = Path(tempfile.mkdtemp(prefix='replay-', dir=run))
    out = case / 'L0-L3/board'
    print(f'Board attempt: {out}', flush=True)
    try:
        config = a.runtime_config.resolve(strict=True) if a.runtime_config else None
        result = whole_run.board(run, DEFAULT_TOOL, slot_config=config, board_dir=out)
    except whole_run.HiaiRunFailure:
        diagnose_decoder_pair(case, names=('L0-L3',))
        raise
    for scope in ('valid19', 'token18'):
        m = result[scope]
        print(f"L0–L3 {scope}: relative_l2={m['relative_l2']:.4%} cosine={m['cosine']:.9f}")
    print(out / 'accuracy.json')
    return 0 if result['status'] == 'PASSED' else 1


if __name__ == '__main__':
    sys.exit(main())
