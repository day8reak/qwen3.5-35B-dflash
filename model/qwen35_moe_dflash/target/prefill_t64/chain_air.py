"""Export original CPro embedding, individual complete W4 decoder blocks and LM head."""
from collections import Counter
from contextlib import contextmanager, nullcontext
import faulthandler
import fcntl
import json
import shutil
from pathlib import Path
import sys
import tempfile
import threading
import time

import numpy as np
import torch
import yaml

from ...common.artifacts import output_path, share_for_transfer
from ..layer_probe.common import KIND, checked, identity, read_array, save_json, sha
from ..layer_probe.replay import export_input_identity, source_bindings
from ..layer_probe.steps import working_directory
from .chain_reference import BASIS, LAYERS
from .prepare import EXPORTER, MODELING, NZ, WRAPPER
from .source_exporter import stage_exporter, stage_utils
from .source_model import stage_model
from .source_wrapper import stage_wrapper


class Block(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden, cache0, cache1, effective_length=None, mask=None,
                positions=None, cache_position=None, all_length=None):
        return self.layer(hidden, past_key_values=(cache0.clone(), cache1.clone()), use_cache=True,
                          effective_length=effective_length, attention_mask=mask, position_ids=positions,
                          new_kv_cache_pos=cache_position, allQLen=all_length, export_flag=True)[0]


class Head(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.norm = model.model.norm
        self.lm_head = model.lm_head

    def forward(self, hidden):
        return self.lm_head(self.norm(hidden.reshape(1, 2048))).reshape(1, 1, -1).float()


class Embedding(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embed = model.model.embed_tokens

    def forward(self, ids):
        return self.embed(ids).to(torch.float16)


@contextmanager
def export_progress(case_id, weight_dir):
    stop = threading.Event()
    started = time.monotonic()

    def observe():
        while not stop.wait(30):
            try:
                sizes = [p.stat().st_size for p in weight_dir.glob('*') if p.is_file()]
            except OSError:
                continue
            print(f'{case_id}: TorchAir active for {time.monotonic() - started:.0f}s; '
                  f'weight directory has {len(sizes)} files, {sum(sizes):,} bytes', flush=True)

    watcher = threading.Thread(target=observe, daemon=True)
    watcher.start()
    faulthandler.dump_traceback_later(180)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
        stop.set()
        watcher.join(timeout=1)


def _export(root, exporter, schema, index, role, module, args, golden, linear=False):
    from ...draft.incremental.torch_weight_export import external_weight_export, validate_external_weights
    from ..export_inventory import _file_constants
    from ..original_air_roundtrip import _attributes, _decode, _read_air
    from ..operator_air_review import _data_edges, port_names

    case_id = f'l{index:02d}_c00_{role}_o0'
    dest = root / 'models' / case_id
    output = 'logits' if role == 'head' else 'hidden'
    if not dest.exists():
        print('export', case_id, flush=True)
        with tempfile.TemporaryDirectory(dir=root.parent, prefix=f'.{root.name}-case-') as tmp:
            folder = Path(tmp) / 'case'
            folder.mkdir()
            def tensor(key, array):
                path = folder / (key + '.bin')
                array.tofile(path)
                return dict(identity(path, folder), dtype=array.dtype.name, shape=list(array.shape))
            specs = [dict(tensor(key, x.numpy()), semantic_name=key) for key, x in args.items()]
            ref = dict(tensor('golden', golden), name=output)
            torch._dynamo.reset()
            # Select storage before wrapping the converter for argument identity.
            # Both hooks must see the original Data slots before they become FileConstants.
            with external_weight_export() as weights, working_directory(folder), export_input_identity(list(args.values())) as evidence:
                adapter = (nullcontext() if role == 'embedding' else
                           exporter.cpro_export_contracts(True, chunk_size=64, gdn_layers=int(linear)))
                print(f'{case_id}: TorchAir export with external weights', flush=True)
                with adapter, export_progress(case_id, folder / 'weight'):
                    exporter.torchair.dynamo_export(model=module.eval(), export_path='weight', dynamic=False, **args)
            air = folder / 'export.air'
            (folder / 'weight/export.air').replace(air)
            print(f"{case_id}: checking AIR ({air.stat().st_size:,} bytes), "
                  f"{weights['weight_files']} weight files ({weights['weight_bytes']:,} bytes)", flush=True)
            raw, _ = _read_air(air)
            _, _, graph = _decode(schema, raw)
            # Check exact GE references against the original tensors and their on-disk extents.
            expected = {r['node']: r['path'] for r in weights['files']}
            observed = {n['name']: str(Path(_attributes(n)['file_path']['s'])) for n in _file_constants(graph)}
            if observed != expected:
                raise ValueError(f'{case_id}: FileConstant names or paths differ from the selected weights')
            sizes = {path: (folder / path).stat().st_size for path in observed.values()}
            validate_external_weights({'weight_files': sizes}, weights)
            bindings = source_bindings(graph, specs, evidence['nodes'], 1)
            edges = {s['semantic_name']: s['name'] + ':0' for s in bindings}
            if set(edges) != set(args):
                raise ValueError(f'{case_id}: runtime inputs changed: {list(edges)}')
            nodes = graph['op']
            counts = Counter(n['type'] for n in nodes)
            if role == 'decoder':
                opname, port, control = (('ChunkGatedDeltaRule', 'effective_length', 'effective_length') if linear
                    else ('AdnFusedInferAttention', 'all_seq_lengths_q', 'all_length'))
                op, = [n for n in nodes if n['type'] == opname]
                attrs = _attributes(op)
                slot, = [i for i, v in port_names(attrs, 'input').items() if v == port]
                if (_data_edges(op)[slot] != edges[control] or (linear and attrs['chunk_size']['i'] != 64)
                        or not any(n.lower().startswith('groupedmatmul') for n in counts)):
                    raise ValueError(f'{case_id}: unexpected attention, lengths or expert graph')
            # OMG can elide singleton dimensions at NetOutput. Only allow views
            # that preserve the original token/feature ordering.
            views = [[1, golden.shape[-1]], [golden.shape[-1]]] if role == 'head' else [[64, 2048]]
            save_json(folder / 'export.json', dict(air=identity(air, folder), source_inputs=bindings, golden=ref,
                weight_files=sizes,
                output_contract=dict(required_dtypes={'0': ref['dtype']}, stage_views={output: views}), graph_ops=dict(counts)))
            folder.rename(dest)
            print(f'{case_id}: AIR ready', flush=True)
    return dict(id=case_id, layer=index, call=0, point=role, ordinal=index, phase='prefill',
                status='EXPORTED', kind='prefill_chain', output_name=output, export=identity(dest / 'export.json', root))


def export(source, config, reference, output, *, decoder_pair=False):
    root = output_path(output)
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / f'.{root.name}-export.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'an AIR export is already active for {root}') from exc
        _export_batch(source, config, reference, root, decoder_pair=decoder_pair)


def stage_cpro_source(source, stage, adapt_wrapper, *, keep_weight_scale=False):
    (stage / 'models').mkdir()
    (stage / 'models/__init__.py').touch()
    for name, transform in ((MODELING, lambda t: stage_model(t, single_token=False, keep_weight_scale=keep_weight_scale)),
                            (WRAPPER, adapt_wrapper),
                            (EXPORTER, stage_exporter), ('utils.py', stage_utils), (NZ, lambda t: t)):
        (stage / name).write_text(transform((source / name).read_text()))


def checkpoint_paths(source, config, ref):
    cfg = yaml.safe_load(config.read_text())
    if sha(config) == ref['config_sha256']:
        # The two private source trees can have different roots. The reference recorded
        # how the exact YAML resolved relative paths in the original 310P working directory.
        paths = {key: Path(ref[key]).resolve(strict=True) for key in ('model_file', 'quanted_pth')}
    else:
        paths = {key: (source / cfg[key]).resolve(strict=True) for key in ('model_file', 'quanted_pth')}
    if any(path != Path(ref[key]) for key, path in paths.items()):
        raise ValueError('310P and CPro must use the same W4 model and checkpoint')
    # The private loader logs and skips absent expert keys. Check shard headers before
    # allowing a partially populated W4 pack to produce a 40-layer numerical report.
    from safetensors import safe_open
    keys = set()
    for shard in paths['quanted_pth'].glob('data*.safetensors'):
        with safe_open(str(shard), framework='pt', device='cpu') as f:
            keys.update(f.keys())
    required = {f'model.layers.{i}.mlp.experts.{proj}_quant_{part}'
                for i in range(LAYERS) for proj in ('gate_up_proj', 'down_proj') for part in ('weight', 'scale')}
    if required - keys:
        raise ValueError(f'40-layer W4 pack is incomplete; first missing keys: {sorted(required - keys)[:4]}')
    return paths


@torch.no_grad()
def _export_batch(source, config, reference, root, *, decoder_pair=False):
    from ...draft.incremental.torch_export import _producer_schema, _schema

    reference = reference.resolve(strict=True)
    config = (source / config).resolve(strict=True)
    ref = json.loads((reference / 'manifest.json').read_text())
    if (ref['reference'] != BASIS or ref['status'] != 'CAPTURED'
            or [r['layer'] for r in ref['blocks']] != list(range(LAYERS))):
        raise ValueError('expected the completed 40-layer 310P reference')
    paths = checkpoint_paths(source, config, ref)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'models').mkdir(exist_ok=True)
    binding = dict(reference=sha(reference / 'manifest.json'), sources={n: sha(source / n)
        for n in (MODELING, WRAPPER, EXPORTER, NZ, 'utils.py')}, exporter=sha(__file__))
    binding['weight_storage'] = 'native-torchair-fileconstant-v1'
    if decoder_pair:
        binding['decoder_layers'] = [0, 3]
    binding_path = root / 'source.json'
    if binding_path.exists():
        previous = json.loads(binding_path.read_text())
        if previous != binding:
            # The original export can be interrupted before the first case is published.
            # Move the whole attempt outside the transmitted tree, including any orphaned scratch.
            if (set(previous) != {'reference', 'sources', 'exporter'}
                    or any(previous[k] != binding[k] for k in ('reference', 'sources'))
                    or (root / 'manifest.json').exists() or any((root / 'models').glob('l*'))):
                raise ValueError('export sources changed; choose a new output directory')
            backup = Path(tempfile.mkdtemp(dir=root.parent, prefix=f'.{root.name}-interrupted-'))
            root.rename(backup / root.name)
            (root / 'models').mkdir(parents=True)
            print(f'Preserved interrupted AIR attempt in {backup}', flush=True)
    save_json(binding_path, binding)
    def array(spec):
        return read_array(reference, spec)
    def tensor(key):
        return torch.from_numpy(array(ref['inputs'][key]))
    cases = []
    with tempfile.TemporaryDirectory(dir=root.parent, prefix=f'.{root.name}-private-source-') as tmp:
        stage = Path(tmp)
        stage_cpro_source(source, stage, stage_wrapper)
        sys.path.insert(0, tmp)
        try:
            import export_glm_model_qwen3_5_moe as exporter
            from models.export_model_wrapper_qwen3_5_moe import load_npu_model
            from models.modeling_qwen3_5_moe_hiai_cpro import Qwen3_5MoeRMSNorm
            schema = _schema(*_producer_schema())
            for base in range(0, 4 if decoder_pair else LAYERS, 4):
                print(f'Preparing CPro weights for L{base}–L{base + 3}', flush=True)
                model = load_npu_model(str(paths['model_file']), 'cpu', torch.float16, variant='cpro',
                                       layer_num=4, start_layer_idx=base, quant_enable=True)
                exporter.quant_model_inplace(model, str(paths['quanted_pth']), model.config.num_experts,
                    start_layer_idx=base, device='cpu', quant_bits=4, qlinear_cls=exporter.CproQLinear)
                model.eval().to(dtype=torch.float16)
                if (model.model.start_layer_idx != base or len(model.model.layers) != 4
                        or model.config.num_hidden_layers != LAYERS or not isinstance(model.lm_head, exporter.CproQLinear)):
                    raise ValueError(f'failed to load W4 decoder group at {base}')
                for m in model.modules():
                    if isinstance(m, Qwen3_5MoeRMSNorm):
                        m.register_buffer('export_gamma', (m.weight + 1.0).detach().contiguous(), persistent=False)
                if base == 0 and not decoder_pair:
                    ids = tensor('input_ids').int()
                    emb = Embedding(model)
                    if not np.array_equal(emb(ids).numpy(), array(ref['inputs']['embedding'])):
                        raise ValueError('token lookup with these weights differs from 310P')
                    cases.append(_export(root, exporter, schema, 0, 'embedding', emb, dict(ids=ids),
                                         array(ref['inputs']['embedding'])))
                    del emb
                for offset, layer in enumerate(model.model.layers):
                    i = base + offset
                    if decoder_pair and i not in (0, 3):
                        continue
                    row = ref['blocks'][i]
                    if layer.block_type != row['block_type']:
                        raise ValueError(f'L{i} attention type differs from 310P')
                    linear = layer.block_type == 'linear_attention'
                    hidden = tensor('embedding') if i == 0 else torch.from_numpy(array(ref['blocks'][i-1]['hidden']))
                    if i == ref['split_layer']:
                        hidden[:, 19:] = 0
                    attn = layer.linear_attn if linear else layer.self_attn
                    names = (('in_proj_qkv', 'in_proj_a', 'in_proj_b', 'in_proj_z', 'out_proj') if linear
                             else ('q_proj', 'k_proj', 'v_proj', 'o_proj'))
                    if not all(isinstance(getattr(attn, n), exporter.CproQLinear) for n in names):
                        raise ValueError(f'L{i}: missing W4 attention weights')
                    if linear:
                        args = dict(hidden=hidden, cache0=torch.zeros((1, 8192, 4), dtype=torch.float16),
                                    cache1=torch.zeros((1, 32, 128, 128), dtype=torch.float16),
                                    effective_length=tensor('effective_length').short())
                    else:
                        if not torch.equal(layer.self_attn.block_table.cpu(), torch.arange(64)[None]):
                            raise ValueError(f'L{i}: unexpected CPro block table')
                        kv = torch.zeros((64, 32, 32, 16), dtype=torch.float16)
                        args = dict(hidden=hidden, cache0=kv, cache1=kv.clone(), mask=tensor('attention_mask').float(),
                                    positions=tensor('position_ids').int(),
                                    cache_position=tensor('new_kv_cache_pos').float().reshape(64, 1),
                                    all_length=torch.tensor([19], dtype=torch.int64))
                    cases.append(_export(root, exporter, schema, i, 'decoder', Block(layer), args, array(row['hidden']), linear))
                if base + 4 == LAYERS:
                    last = torch.from_numpy(array(ref['blocks'][-1]['hidden'])[:, 18, :].copy())
                    cases.append(_export(root, exporter, schema, 39, 'head', Head(model), dict(hidden=last),
                                         array(ref['outputs']['logits'])))
                del layer, model
        finally:
            sys.path.remove(tmp)
    shutil.copyfile(checked(reference, ref['token_text_map']), root / 'token-text.json')
    save_json(root / 'manifest.json', dict(artifact_kind=KIND, status='EXPORTED', cases=cases,
        scope=BASIS, golden_source=BASIS, reference_model_layers=LAYERS, capture_sha256=binding['reference'],
        token_text_map=identity(root / 'token-text.json', root), split_layer=ref['split_layer'], reference_top1=ref['top1'], reference_text=ref['token_text']))
    share_for_transfer(root)
