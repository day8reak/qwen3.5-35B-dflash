"""Stage one measured L0 operator, including from the earlier three-operator batch."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from qwen35_moe_dflash.common.command_trace import note
from qwen35_moe_dflash.target.layer_probe.common import checked, identity, save_json, sha
from qwen35_moe_dflash.target.prefill_t64.gdn_tail_probe import case_id


def stage_one(source, dest, stage, basis, copy_file, host):
    """Publish only after the input, references, AIR and optional weights all match their hashes."""
    name = case_id(stage)
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix='.step-') as tmp:
        batch = Path(tmp)/'air'
        batch.mkdir()

        def fetch(relative):
            rel = Path(relative)
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError(f'Invalid relative fixture path: {rel}')
            target = batch/rel
            target.parent.mkdir(parents=True, exist_ok=True)
            note(f'[fixture from {host}] {source/rel} -> {target}')
            try:
                copy_file(str(source/rel), str(target))
            except FileNotFoundError as exc:
                raise FileNotFoundError(source/rel) from exc
            return target

        manifest_path = fetch('manifest.json')
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('golden_source') != basis:
            raise ValueError(f'{source}: expected a native 310P operator fixture for {stage}')
        cases = [c for c in manifest['cases'] if c['id'] == name and c['status'] == 'EXPORTED']
        if len(cases) != 1:
            raise ValueError(f'{source}: expected one exported {name}, found {len(cases)}')
        case = cases[0]
        model = Path('models')/name
        if Path(case['export']['path']) != model/'export.json':
            raise ValueError(f'{source}: unexpected export location for {stage}')
        fetch(case['export']['path'])
        spec = json.loads(checked(batch, case['export']).read_text())
        for entry in [spec['air'], *spec['source_inputs'], spec['golden'], spec['cpu'], *spec['weight_evidence'].values()]:
            fetch(model/entry['path'])
            checked(batch/model, entry)
        if 'boundaries' in manifest:
            fetch(manifest['boundaries']['path'])
            boundary = checked(batch, manifest['boundaries'])
        if len(manifest['cases']) > 1:
            manifest['parent_batch'] = dict(path=str(source), manifest_sha256=sha(manifest_path))
            manifest['cases'] = [case]
            if 'boundaries' in manifest:
                save_json(boundary, {stage: json.loads(boundary.read_text())[stage]})
                manifest['boundaries'] = identity(boundary, batch)
            save_json(manifest_path, manifest)
        batch.rename(dest)


def ensure_fixture(run, step, stage, basis, *, local=None, remote=None):
    dest = run/'air'
    if local and remote:
        raise ValueError('Select either AIR_ON_35 or AIR_ON_78')
    if (dest/'manifest.json').is_file():
        note(f'[reuse AIR and inputs] {dest}')
        return
    tests = Path(os.environ['AI_RUN_DIR'])/'target/tests'
    defaults = [tests/f"l0-{step.replace('_', '-')}-t64-valid19-01"/'air']
    if stage in ('rms_output', 'silu_output', 'out_proj_only'):
        defaults.append(tests/'gdn-op-matrix-l0-valid19-01'/'air')
    missing = []
    run.mkdir(parents=True, exist_ok=True)

    def select(host, candidates, copy_file):
        for path in candidates:
            try:
                stage_one(path, dest, stage, basis, copy_file, host)
            except FileNotFoundError as exc:
                missing.append(f'{host} {path} ({exc})')
            else:
                print(f'Using {stage} from {host}: {path} -> {dest}', flush=True)
                return True
        return False

    if not remote and select('35', [local] if local else defaults, shutil.copyfile):
        return
    if not local:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
        from pull_artifacts import DEFAULT_CONFIG, connection
        with connection(DEFAULT_CONFIG, 'air') as client, client.open_sftp() as sftp:
            sftp.get_channel().settimeout(120)
            if select('78', [remote] if remote else defaults, sftp.get):
                return
    attempts = '\n'.join(missing)
    raise FileNotFoundError(f'No complete {step} fixture. Checked:\n{attempts}\n'
                            f'On 78 run: bash targets/cpro/scripts/conv_silu.sh prepare --step {step}\n'
                            'For a custom air/ directory set AIR_ON_35 or AIR_ON_78 on 35.')
