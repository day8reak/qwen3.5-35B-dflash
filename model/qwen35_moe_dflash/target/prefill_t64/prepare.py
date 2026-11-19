"""Bind one fresh T64 fixture and stage local CPro sources for the exporter."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import yaml

from ..layer_probe.common import checked

MODELING = 'models/modeling_qwen3_5_moe_hiai_cpro.py'
NZ = 'models/modeling_qwen3_5_moe_hiai_nz.py'
WRAPPER = 'models/export_model_wrapper_qwen3_5_moe.py'
EXPORTER = 'export_glm_model_qwen3_5_moe.py'
STATES = tuple(name for i in range(3) for name in (f'conv_state_{i}', f'recurrent_state_{i}')) + ('key_cache_3', 'value_cache_3')


def prepare_inputs(fixture, job):
    manifest = json.loads((fixture / 'manifest.json').read_text())
    if (manifest['status'] != 'CAPTURED' or manifest['layers'] != [0, 1, 2, 3]
            or manifest['sequence_length'] != 64 or manifest['input_contract']['cache_capacity'] != 2048):
        raise ValueError('Expected the fresh four-layer T64 Prefill fixture with cache capacity 2048')
    if [r['semantic_name'] for r in manifest['outputs']] != ['logits', *STATES]:
        raise ValueError('Unexpected Prefill output contract')

    positions = np.arange(64, dtype='<i4')
    controls = {'new_kv_cache_pos': positions, 'position_ids': positions[None],
                'output_pos': np.array([63], dtype='<i4'), 'effective_length': np.array([64], dtype='<i2'),
                'allQLen': np.array([64], dtype='<i8')}
    shapes = {'inputs_embeds': ('int8', [1, 64, 2048]), 'embed_scale': ('float32', [1, 64, 1]),
              'attention_mask': ('float32', [1, 1, 64, 2048]), 'input_ids': ('int32', [1, 64])}
    for name in STATES:
        shape = [64, 32, 32, 16]
        if name.startswith('conv_state'):
            shape = [1, 8192, 4]
        elif name.startswith('recurrent_state'):
            shape = [1, 32, 128, 128]
        shapes[name] = ('float16', shape)
    for name, value in controls.items():
        shapes[name] = (value.dtype.name, list(value.shape))
    specs = {r['name']: r for r in manifest['inputs']}
    if len(specs) != len(manifest['inputs']) or set(specs) != set(shapes) - {'input_ids'}:
        raise ValueError('Unexpected or repeated Prefill input')
    specs['input_ids'] = manifest['tokens']
    prepared = {}
    (job / 'inputs').mkdir()
    for name, spec in specs.items():
        dtype, shape = shapes[name]
        if spec['dtype'] != dtype or spec['shape'] != shape:
            raise ValueError(f'Unexpected {name} descriptor (expected {dtype} {shape})')
        value = np.fromfile(checked(fixture, spec), dtype=np.dtype(dtype).newbyteorder('<')).reshape(shape)
        if name in controls and not np.array_equal(value, controls[name]):
            raise ValueError(f'Unexpected fresh Prefill control {name}')
        if name in STATES and np.any(value):
            raise ValueError(f'Expected a fresh zero cache for {name}')
        if name == 'attention_mask':
            causal = np.where(np.arange(2048)[None, :] <= positions[:, None], 0., -np.inf)
            if not np.array_equal(value[0, 0], causal):
                raise ValueError('Expected the captured causal Prefill mask')
        if name == 'new_kv_cache_pos':
            value = value.astype('<f4')
        dest = Path('inputs') / (name + '.bin')
        raw = value.tobytes()
        (job / dest).write_bytes(raw)
        prepared[name] = dict(path=str(dest), shape=shape, dtype=value.dtype.name, bytes=len(raw),
                              sha256=hashlib.sha256(raw).hexdigest())
    (job / 'inputs.json').write_text(json.dumps(prepared, indent=2) + '\n')


def prepare(source, config, fixture, output, repo):
    from .source_model import stage_model
    from .source_wrapper import stage_wrapper
    from .source_exporter import stage_utils, stage_exporter

    source = Path(source).resolve(strict=True)
    fixture = Path(fixture).resolve(strict=True)
    output = Path(output).resolve()
    if any(output == p or p in output.parents or output in p.parents for p in (source, fixture, repo)):
        raise ValueError('Choose a job directory outside the source, repository and capture')
    config = Path(config)
    config = config.resolve(strict=True) if config.is_file() else (source / config).resolve(strict=True)
    cfg = yaml.safe_load(config.read_text())
    for key in ('model_file', 'quanted_pth'):
        cfg[key] = str((source / cfg[key]).resolve(strict=True))
    manifest = json.loads((fixture / 'manifest.json').read_text())
    segmented = 'valid_rows' in manifest
    transforms = {MODELING: lambda text: stage_model(text, single_token=not segmented),
                  WRAPPER: lambda text: stage_wrapper(text, segmented=segmented),
                  EXPORTER: stage_exporter, 'utils.py': stage_utils, NZ: None}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.prepare-t64-') as staging:
        job = Path(staging) / 'job'
        (job / 'models').mkdir(parents=True)
        (job / 'models/__init__.py').touch()
        for name, transform in transforms.items():
            original = (source / name).read_text()
            staged = transform(original) if transform else original
            ast.parse(staged, filename=name)
            (job / name).write_text(staged)
        if segmented:
            from .segment_export import prepare_segment
            prepare_segment(fixture, manifest, job)
        else:
            prepare_inputs(fixture, job)
        (job / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True))
        # The job owns its local private modules; public adapter imports resolve from this repository.
        (job / 'export_t64.py').write_text(
            'import os, sys\nfrom pathlib import Path\n'
            f"repo = Path(os.environ.get('REPO_DIR') or {str(repo)!r})\n"
            "sys.path.insert(0, str(repo / 'model'))\n"
            'from qwen35_moe_dflash.target.prefill_t64.export_job import export_air\n'
            'export_air(Path(__file__).resolve().parent)\n')
        if not segmented:
            (job / 'reference_t64.py').write_bytes(Path(__file__).with_name('reference.py').read_bytes())
        if output.exists():
            for path in job.rglob('*'):
                if not path.is_file():
                    continue
                existing = output / path.relative_to(job)
                if not existing.is_file() or existing.read_bytes() != path.read_bytes():
                    raise ValueError(f'Existing job differs at {existing}. Choose a fresh --output directory.')
        else:
            job.rename(output)
    return output
