"""Persist the 40-layer T64 lookup golden independently of CPro experiment directories."""
from copy import copy
import fcntl
import hashlib
from importlib import metadata
import inspect
import json
from pathlib import Path
import tempfile

import yaml

from decoder_handoff import complete_file, publish, receipt, receive
from qwen35_moe_dflash.common.hashing import sha256_file
from qwen35_moe_dflash.target.prefill_t64.native_40 import capture_and_cpu


def reference_inputs(source, config, *, valid64_cpu=False):
    cfg = yaml.safe_load(config.read_text())
    weights = {}
    for name in ('model_file', 'quanted_pth'):
        directory = (source / cfg[name]).resolve(strict=True)
        files = {}
        for path in sorted(directory.rglob('*')):
            rel = path.relative_to(directory)
            if any(part.startswith('.') or part == '__pycache__' for part in rel.parts) or not path.is_file():
                continue
            stat = path.stat()
            files[str(rel)] = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino]
        weights[name] = {'directory': str(directory), 'files': files}

    target = Path(__file__).resolve().parents[3]/'model/qwen35_moe_dflash/target'
    math_files = ('prefill_t64/chain_reference.py', 'prefill_t64/split_cpu_reference.py', 'prefill_fixture.py',
                  'prefill_t64/cpu_reference.py', 'prefill_t64/gdr_math.py', 'original_quant.py',
                  'layer_probe/qlinear_math.py', 'layer_probe/target_weights.py', 'layer_probe/reference_config.py',
                  'layer_probe/capture.py', 'layer_probe/steps.py')
    nz_files = ('inference_segment.py', 'inference.py', 'utils.py', 'models/export_model_wrapper_qwen3_5_moe.py',
                'models/modeling_qwen3_5_moe_hiai_nz.py')
    packages = {}
    for name in ('torch', 'torch-npu', 'transformers', 'numpy'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    binding = dict(case='40-prefill-t64', embedding='checkpoint_lookup', packages=packages,
                   nz_config=sha256_file(config), nz_sources={n: sha256_file(source/n) for n in nz_files if (source/n).is_file()},
                   reference_math={n: sha256_file(target/n) for n in math_files}, checkpoints=weights)
    if not valid64_cpu:
        binding.update(prompt=cfg['prompts'][0], capture=hashlib.sha256(inspect.getsource(capture_and_cpu).encode()).hexdigest())
    return binding


def complete(case, binding):
    try:
        if json.loads((case/'binding.json').read_bytes()) != binding:
            return False
        manifest = json.loads(receipt(case, 'golden').read_bytes())
        files = manifest['files']
        required = {'reference/manifest.json', *(f'L{i:02d}/cpu/manifest.json' for i in range(40))}
        return (manifest['stage'] == 'golden' and manifest['case'] == '40-prefill-t64'
                and required.issubset(row['path'] for row in files)
                and all(complete_file(case/'export'/row['path'], row) for row in files))
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        return False


def prepare(root, cache_root, args, source, config):
    binding = reference_inputs(source, config)
    local = copy(args)
    if args.cache_existing_golden:
        # Both a generated and a received golden carry the NZ capture. Older
        # cases recorded checkpoint paths but not shard revisions.
        export = root/'export'
        previous_ref = export/'reference/manifest.json'
        old = json.loads(previous_ref.read_bytes())
        if (old['config_sha256'] != binding['nz_config'] or old['source_sha256'] != binding['nz_sources']['inference_segment.py']
                or old['prompt'] != binding['prompt']):
            raise ValueError(f'{export}: the 310P configuration, prompt or driver differs from the current golden')
        if any(old[name] != stamp['directory'] for name, stamp in binding['checkpoints'].items()):
            raise ValueError(f'{export}: the recorded checkpoint paths differ from the current configuration')
    encoded = json.dumps(binding, sort_keys=True, ensure_ascii=False).encode()
    key = hashlib.sha256(encoded).hexdigest()
    saved = cache_root/key
    cache_root.mkdir(parents=True, exist_ok=True)
    with (cache_root/f'.{key}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if complete(saved, binding):
            if args.cache_existing_golden:
                manifests = ['reference/manifest.json', *(f'L{i:02d}/cpu/manifest.json' for i in range(40))]
                if any(sha256_file(export/name) != sha256_file(saved/'export'/name) for name in manifests):
                    raise ValueError(f'{saved}: this configuration already has a different golden')
            print(f'[golden] 复用 40 层 CPU 和 310P 参考：{saved}', flush=True)
        else:
            action = '归档已有' if args.cache_existing_golden else '生成'
            print(f'[golden] {action} 40 层 CPU 和 310P 参考 → {saved}', flush=True)
            with tempfile.TemporaryDirectory(dir=cache_root, prefix='.build-') as tmp:
                staged = Path(tmp)/root.name
                if args.cache_existing_golden:
                    staged.mkdir()
                    local.local_source = root
                    receive(staged, root, 'golden', local)
                    for i in range(40):
                        if not (staged/f'export/L{i:02d}/cpu/manifest.json').is_file():
                            raise ValueError(f'{root}: missing CPU golden for layer {i}')
                    if not (staged/'export/reference/manifest.json').is_file():
                        raise ValueError(f'{root}: missing 310P golden')
                # With a complete import, the native reference path verifies and
                # reuses every CPU layer and the original capture without inference.
                site = staged/'export'
                site.mkdir(parents=True, exist_ok=True)
                capture_and_cpu(site, site/'reference', source, config)
                if binding != reference_inputs(source, config):
                    raise ValueError('Golden inputs changed during capture. Rerun with the current 310P configuration.')
                (staged/'binding.json').write_bytes(encoded)
                publish(staged, 'golden', args)
                if saved.exists():
                    previous = Path(tempfile.mkdtemp(dir=cache_root, prefix='incomplete-'))/key
                    saved.rename(previous)
                    print(f'[golden] 已保留不完整的旧参考：{previous}', flush=True)
                staged.rename(saved)
        if not args.cache_existing_golden:
            local.local_source = saved
            receive(root, saved, 'golden', local)
