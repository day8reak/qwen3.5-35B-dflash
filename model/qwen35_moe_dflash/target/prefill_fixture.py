"""Capture one original four-layer Prefill invocation and its actual outputs."""
from __future__ import annotations

import inspect
import os
from pathlib import Path
from contextlib import contextmanager
from functools import wraps
import sys

import numpy as np
import torch

from .layer_probe.capture import bound, cpu
from .layer_probe.common import identity, new_output, save_json, sha
from .layer_probe.reference_config import hf_config_prefix
from .layer_probe.steps import build_four_layers, import_source, initial_state, working_directory


STATE_NAMES = tuple(name for layer in range(4) for name in (
    (f'conv_state_{layer}', f'recurrent_state_{layer}') if layer < 3
    else ('key_cache_3', 'value_cache_3')))
INPUT_NAMES = ('embed_scale', 'inputs_embeds', 'new_kv_cache_pos', 'effective_length',
               'position_ids', 'attention_mask', 'allQLen', 'output_pos')
DTYPES = {'float16': '<f2', 'float32': '<f4', 'int8': 'i1',
          'int16': '<i2', 'int32': '<i4', 'int64': '<i8', 'bool': '?'}


def load_tokens(path, sequence_length, max_len, vocab):
    if not 2 <= sequence_length <= min(max_len, 32767):
        raise ValueError('Prefill requires 2 <= sequence-length <= min(cache capacity, 32767)')
    path = Path(path).resolve(strict=True)
    if path.stat().st_size != sequence_length * 4:
        raise ValueError('input-ids must contain exactly sequence-length little-endian INT32 tokens; no padding/truncation')
    tokens = np.fromfile(path, dtype='<i4')
    if np.any(tokens < 0) or np.any(tokens >= vocab):
        raise ValueError('input token is outside the model vocabulary')
    return torch.from_numpy(tokens.astype(np.int64)).unsqueeze(0)


def prompt_tokens(config, tokenizer_factory, max_len, *, prompt_index=0, prompt_repeat=1,
                  sequence_length=None):
    """Encode one YAML prompt with the original tokenizer call, without padding."""
    fields = [name for name in ('prompts', 'prompt') if name in config]
    if len(fields) != 1:
        raise ValueError('YAML must contain exactly one of prompts (list) or prompt (string)')
    field = fields[0]
    prompts = config[field]
    if isinstance(prompts, str):
        prompts = [prompts]
    if (not isinstance(prompts, list) or not prompts
            or any(not isinstance(p, str) or not p.strip() for p in prompts)):
        raise ValueError('YAML prompts must contain nonempty strings')
    if type(prompt_index) is not int or not 0 <= prompt_index < len(prompts):
        raise ValueError(f'prompt-index must be between 0 and {len(prompts)-1}')
    if type(prompt_repeat) is not int or prompt_repeat < 1:
        raise ValueError('prompt-repeat must be a positive integer')
    tokenizer = tokenizer_factory.from_pretrained(config['model_file'])
    text = prompts[prompt_index] * prompt_repeat
    # Original inference.py overwrites its chat-template text before this call.
    # Keep tokenizer defaults, but make the old debug repetition explicit.
    encoded = tokenizer([text], return_tensors='pt')
    tokens, mask = encoded['input_ids'], encoded['attention_mask']
    if (not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.int64
            or tokens.ndim != 2 or tokens.shape[0] != 1
            or not isinstance(mask, torch.Tensor) or mask.shape != tokens.shape
            or not torch.all(mask == 1)):
        raise ValueError('tokenizer must return unpadded INT64 input_ids and all-one attention_mask for batch=1')
    length = tokens.shape[1]
    if not 2 <= length <= min(max_len, 32767):
        raise ValueError(f'prompt produced {length} tokens; Prefill requires 2..{min(max_len, 32767)}; no padding/truncation')
    if sequence_length is not None and sequence_length != length:
        raise ValueError(f'prompt produced {length} tokens, differs from --sequence-length {sequence_length}; no padding/truncation')
    return cpu(tokens), dict(source='YAML_PROMPT', yaml_field=field, prompt_index=prompt_index,
        prompt=prompts[prompt_index], prompt_repeat=prompt_repeat,
        tokenizer_path=config['model_file'], tokenizer_class=type(tokenizer).__name__,
        text_policy='RAW_PROMPT; NO_CHAT_TEMPLATE; EXPLICIT_REPEAT',
        special_tokens='TOKENIZER_DEFAULT_AS_ORIGINAL_INFERENCE', sequence_length=length)


class PrefillCapture:
    """Hooks save independent CPU snapshots before the caller can mutate state."""

    def __init__(self, root, tokens, config, max_len, record):
        self.root, self.record = Path(root), record
        self.tokens, self.config, self.max_len = cpu(tokens), config, max_len
        self.pre_calls = self.post_calls = 0
        self.handles = []
        self.state_shapes = [list(t.shape) for pair in initial_state(config, max_len, 'cpu') for t in pair]
        self.record.update(inputs=[], outputs=[], intermediates=[],
            input_order='SEMANTIC_NAMES_ONLY; BIND_TO_COMPILED_RUNTIME_NAMES',
            runtime_order_verified=False, source_wrapper_calls=0)

    def tensor(self, folder, name, value):
        value = cpu(value)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'{name} must be a tensor')
        dtype = str(value.dtype).removeprefix('torch.')
        if dtype not in DTYPES:
            raise ValueError(f'unsupported snapshot dtype {dtype}: {name}')
        path = self.root / folder / (name + '.bin')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            value.numpy().astype(DTYPES[dtype], copy=False).tofile(stream)
        return dict(identity(path, self.root), name=name, shape=list(value.shape), dtype=dtype)

    def argument(self, name, value):
        if isinstance(value, torch.Tensor):
            return self.tensor('source_inputs', name, value)
        if isinstance(value, (list, tuple)):
            return [self.argument(f'{name}_{i}', item) for i, item in enumerate(value)]
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        raise TypeError(f'unsupported original wrapper argument {name}: {type(value).__name__}')

    def _equal(self, values, name, expected, dtype=None):
        actual = values.get(name)
        if not isinstance(actual, torch.Tensor):
            raise ValueError(f'original Prefill {name} is not a tensor')
        if dtype is not None and actual.dtype != dtype:
            raise ValueError(f'original Prefill {name} dtype {actual.dtype} differs from {dtype}')
        if actual.shape != expected.shape or not torch.equal(actual, expected):
            raise ValueError(f'original Prefill {name} differs from the declared fresh Prefill contract')

    def before(self, module, args, kwargs):
        self.pre_calls += 1
        if self.pre_calls != 1:
            raise ValueError('expected exactly one wrapper invocation, without Decode')
        values = cpu(bound(module, args, kwargs))
        self.record['source_arguments'] = {name: self.argument(name, value) for name, value in values.items()}
        length = self.tokens.shape[1]
        self._equal(values, 'input_ids', self.tokens, torch.int64)
        positions = torch.arange(length, dtype=torch.int64)
        self._equal(values, 'position_ids', positions[None], torch.int64)
        self._equal(values, 'new_kv_cache_pos', positions, torch.int64)
        self._equal(values, 'output_pos', torch.tensor([length-1], dtype=torch.int32), torch.int32)
        self._equal(values, 'effective_length', torch.tensor([length], dtype=torch.int16), torch.int16)
        if values.get('allQLen') != [length]:
            raise ValueError('original Prefill allQLen must equal [sequence-length]')
        for name, expected in (('use_cache', True), ('topK', None), ('export_flag', False),
                               ('output_attentions', False), ('output_hidden_states', False)):
            if values.get(name) != expected:
                raise ValueError(f'original Prefill requires {name}={expected}')
        mask = torch.full((1, 1, length, self.max_len), float('-inf'), dtype=torch.float32)
        mask[0, 0][torch.arange(self.max_len)[None] <= torch.arange(length)[:, None]] = 0
        self._equal(values, 'attention_mask', mask, torch.float32)
        for name, dtype, shape in (
            ('inputs_embeds', torch.int8, (1, length, self.config.hidden_size)),
            ('embed_scale', torch.float32, (1, length, 1))):
            tensor = values.get(name)
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype or tuple(tensor.shape) != shape:
                raise ValueError(f'original Prefill {name} must have dtype={dtype}, shape={shape}')
            if not torch.isfinite(tensor).all():
                raise ValueError(f'nonfinite original input: {name}')
        pairs = values.get('past_key_values')
        if not isinstance(pairs, (tuple, list)) or len(pairs) != 4 or any(len(pair) != 2 for pair in pairs):
            raise ValueError('expected four original cache pairs')
        states = [tensor for pair in pairs for tensor in pair]
        for name, tensor, shape in zip(STATE_NAMES, states, self.state_shapes):
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float16 or list(tensor.shape) != shape:
                raise ValueError(f'initial state shape/dtype differs: {name}')
            if torch.count_nonzero(tensor).item():
                raise ValueError(f'fresh Prefill requires a zero initial state: {name}')
        # This is a named serialization for the explicit-ConvState CPro model,
        # not a claim about the ordering of its compiled input slots.
        prepared = {name: values[name] for name in INPUT_NAMES}
        prepared['allQLen'] = torch.tensor(values['allQLen'], dtype=torch.int64)
        for name in ('position_ids', 'new_kv_cache_pos'):
            prepared[name] = values[name].to(torch.int32)
        prepared.update(zip(STATE_NAMES, states))
        self.record['inputs'] = [self.tensor('inputs', name, value) for name, value in prepared.items()]
        self.record['serialization'] = {
            'position_ids': 'VALUE_PRESERVING_INT64_TO_INT32',
            'new_kv_cache_pos': 'VALUE_PRESERVING_INT64_TO_INT32',
            'allQLen': 'ORIGINAL_LIST_AS_INT64_TENSOR',
            'other_inputs': 'ORIGINAL_DTYPE_AND_BYTES',
            'linear_update_tensor': 'ABSENT; EXPLICIT_CONVSTATE_MODEL_ONLY',
        }
        self.record['input_contract'] = dict(status='CHECKED', phase='prefill', batch=1,
            sequence_length=length, effective_length=length, cache_capacity=self.max_len,
            initial_state='ALL_ZERO_FP16', mask='STRICT_CAUSAL_ZERO_OR_NEGATIVE_INFINITY',
            output_pos=length-1, token_policy='ALL_FILE_TOKENS_VALID_NO_PADDING_NO_TRUNCATION')
        save_json(self.root/'manifest.json', self.record)

    def after(self, module, args, kwargs, result):
        self.post_calls += 1
        if self.post_calls != 1 or self.pre_calls != 1:
            raise ValueError('unexpected wrapper output invocation')
        if not isinstance(result, (tuple, list)) or len(result) != 9:
            raise ValueError('expected logits and eight state tensors from the four-layer wrapper')
        outputs = cpu(result)
        expected = [[1, 1, self.config.vocab_size]] + self.state_shapes
        for index, (name, value, shape) in enumerate(zip(('logits',) + STATE_NAMES, outputs, expected)):
            row = self.tensor('golden', 'golden_' + name, value)
            row.update(semantic_name=name, source_tuple_index=index)
            self.record['outputs'].append(row)
            if list(value.shape) != shape or not value.is_floating_point():
                raise ValueError(f'original wrapper output {index} has unexpected shape/dtype')
            if index == 0 and value.dtype != torch.float32:
                raise ValueError('original wrapper logits must be FP32')
            if not torch.isfinite(value).all():
                raise ValueError(f'original wrapper output {index} contains nonfinite values')
        self.record['source_wrapper_calls'] = 1
        self.record['source_top1'] = int(outputs[0][0, 0].argmax())
        save_json(self.root/'manifest.json', self.record)

    def _intermediate(self, name, result):
        value = result[0] if isinstance(result, (tuple, list)) else result
        self.record['intermediates'].append(self.tensor('intermediates', name, value))

    def attach(self, wrapper, intermediates=False):
        self.handles.append(wrapper.register_forward_pre_hook(self.before, with_kwargs=True))
        self.handles.append(wrapper.register_forward_hook(self.after, with_kwargs=True))
        if intermediates:
            for index, layer in enumerate(wrapper.model.model.layers):
                self.handles.append(layer.register_forward_pre_hook(
                    lambda module, args, kwargs, i=index: self._intermediate(
                        f'l{i:02d}_input', args[0] if args else kwargs['hidden_states']), with_kwargs=True))
                for part in ('input_layernorm', 'linear_attn', 'self_attn', 'post_attention_layernorm', 'mlp'):
                    submodule = getattr(layer, part, None)
                    if submodule is not None:
                        self.handles.append(submodule.register_forward_hook(
                            lambda module, args, result, name=f'l{index:02d}_{part}': self._intermediate(name, result)))
                self.handles.append(layer.register_forward_hook(
                    lambda module, args, result, name=f'l{index:02d}_output': self._intermediate(name, result)))

    def close(self):
        for handle in self.handles:
            handle.remove()


@contextmanager
def attention_audit(npu, capture, *, live_calls=None):
    """Observe the original call; an independent diagnostic never replaces it."""
    from .layer_probe.attention_reference import reference_math, unpack
    from .layer_probe.common import metrics
    report = capture.record
    report['attention_audit'] = dict(status='UNAVAILABLE', calls=[], scope='ATTENTION_BOUNDARY_ONLY')
    original = getattr(npu, 'adn_fused_infer_attention', None) if npu is not None else None
    if not callable(original):
        report['attention_audit']['reason'] = 'original torch_npu attention entrypoint unavailable'
        yield
        return

    @wraps(original)
    def observed(*args, **kwargs):
        row = dict(status='UNAVAILABLE', call=len(report['attention_audit']['calls']))
        report['attention_audit']['calls'].append(row)
        values = None
        try:
            if args:
                raise ValueError('only the reviewed keyword-bound source call is supported')
            values = cpu(kwargs)
        except Exception as exc:
            row['reason'] = f'{type(exc).__name__}: {exc}'
        result = original(*args, **kwargs)
        try:
            if values is None:
                return result
            allowed = {'query', 'key', 'value', 'actual_seq_lengths_q', 'actual_seq_lengths_kv',
                'block_table', 'num_heads', 'num_key_value_heads', 'block_size', 'input_layout',
                'scale_value', 'inner_precise', 'atten_mask', 'all_seq_lengths_q'}
            if set(values) != allowed or values['input_layout'] != 'BNSD' or values['inner_precise'] != 2:
                raise ValueError('attention arguments differ from the reviewed BNSD source contract')
            for name in ('key', 'value'):
                if not isinstance(values[name], (tuple, list)) or len(values[name]) != 1:
                    raise ValueError(f'{name} must be one paged cache tensor')
            query, mask = values['query'], values['atten_mask']
            length = values['all_seq_lengths_q']
            if isinstance(length, torch.Tensor):
                if length.dtype != torch.int64 or tuple(length.shape) != (1,):
                    raise ValueError('all_seq_lengths_q tensor must be INT64[1]')
                length = length.tolist()
            if (not isinstance(length, (list, tuple)) or len(length) != 1 or type(length[0]) is not int
                    or length[0] != capture.tokens.shape[1] or query.shape[2] != length[0]
                    or values['actual_seq_lengths_q'] != [query.shape[2]]
                    or values['actual_seq_lengths_kv'] != [mask.shape[-1]]):
                raise ValueError('attention lengths differ from this fresh Prefill invocation')
            tensors = dict(query=query, key=values['key'][0], value=values['value'][0],
                           mask=mask, table=values['block_table'])
            folder = f'attention-audit/call{row["call"]}'
            row['inputs'] = [capture.tensor(folder, name, value) for name, value in tensors.items()]
            prepared = reference_math(*(tensors[name].numpy() for name in ('query', 'key', 'value', 'mask', 'table')),
                length[0], heads=values['num_heads'], kv_heads=values['num_key_value_heads'],
                block_size=values['block_size'], scale=values['scale_value'])
            row['reference'] = prepared['evidence']
            actual = cpu(result)
            row['actual'] = capture.tensor(folder, 'original_output', actual)
            if prepared['output'] is None:
                raise ValueError(prepared['evidence']['status'])
            if actual.shape != query.shape or actual.dtype != torch.float16:
                raise ValueError('original attention output is not the reviewed packed FP16 query shape')
            row['independent_reference'] = capture.tensor(folder, 'math_reference_bnsd', torch.from_numpy(prepared['output']))
            row['metrics'] = metrics(unpack(actual.numpy(), values['num_heads']), prepared['output'])
            row['status'] = 'PASSED' if row['metrics']['passed'] else 'FAILED'
            if live_calls is not None:
                live_calls.append((kwargs, values))
        except Exception as exc:
            row['reason'] = f'{type(exc).__name__}: {exc}'
        return result

    npu.adn_fused_infer_attention = observed
    try:
        yield
    finally:
        npu.adn_fused_infer_attention = original
        rows = report['attention_audit']['calls']
        if len(rows) == 1:
            report['attention_audit']['status'] = rows[0]['status']
        else:
            report['attention_audit']['reason'] = 'expected exactly one four-layer full-attention call'
        status = report['attention_audit']['status']
        report['reference_qualification'] = ('ATTENTION_REFERENCE_MISMATCH' if status == 'FAILED'
            else 'ORIGINAL_EXECUTION_ONLY; WHOLE_MODEL_REFERENCE_UNVERIFIED')


def execute_prefill(source, wrapper, tokens, root, record, *, intermediates=False, npu=None,
                    capture_boundaries=False, l3_repeat=None, fused_value_baseline=None, probe_mask=False,
                    probe_mode='value'):
    """Run the original prefill once. No decoder or second golden evaluation."""
    root = Path(root)
    max_len = source.config_data['kv_cache_max_len']
    source.real_seq_len = tokens.shape[1]
    source.input_ids = tokens
    caches = initial_state(wrapper.model.config, max_len, source.device)
    mask = source.create_strict_causal_mask(source.real_seq_len, max_len, 'cpu')[None, None]
    capture = PrefillCapture(root, tokens, wrapper.model.config, max_len, record)
    original = root/'original'
    original.mkdir()
    source.output_path = str(original) + os.sep
    boundaries, replay, error = None, None, None
    live_calls = [] if fused_value_baseline is not None else None
    try:
        capture.attach(wrapper, intermediates)
        if capture_boundaries:
            from .layer_probe.capture import Capture
            if npu is None:
                raise ValueError('boundary capture requires the original torch_npu module')
            boundaries = Capture(root/'capture', layers=[0, 1, 2, 3], calls=1)
            boundaries.report.update(entrypoint='DIRECT_ORIGINAL_SINGLE_PREFILL', reference_model_layers=4,
                placement=dict(policy='CPU_CONSTRUCT_FOUR_QUANTIZE_THEN_MOVE',
                    executed_num_hidden_layers=4, decoder_layer_limit=4, dtype_policy='DEVICE_ONLY_NO_CAST'))
            for item in record.get('sources', []):
                boundaries.source(item['path'])
            owner = sys.modules[type(wrapper).__module__]
            nz = sys.modules['models.modeling_qwen3_5_moe_hiai_nz']
            boundaries.install(npu, nz, owner)
            if l3_repeat is not None:
                if boundaries.quant_audit is None:
                    raise ValueError('original QLinear matmul audit is unavailable for L3 V')
                l3_repeat.module = wrapper.model.model.layers[3].self_attn.v_proj
                boundaries.quant_audit.l3_repeat = l3_repeat
            from .layer_probe.attention_replay_reference import FullAttentionReplay
            replay = FullAttentionReplay(boundaries.root, boundaries.report, npu, layer=3)
            replay.attach(wrapper)
        with torch.no_grad(), working_directory(original), attention_audit(npu, capture, live_calls=live_calls):
            source.prefill(wrapper, tokens, mask, caches, max_len)
        if capture.pre_calls != 1 or capture.post_calls != 1 or len(record['outputs']) != 9:
            raise ValueError('original prefill did not produce one complete four-layer wrapper invocation')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        capture.close()
        if replay is not None:
            replay.close()
        if boundaries is not None:
            boundaries.close(error)
            record['boundary_capture'] = dict(path='capture/manifest.json', status=boundaries.report['status'])
    if l3_repeat is not None:
        l3_repeat.repeat()
    if replay is not None:
        # The independent module replay has its own cloned states. All original
        # capture hooks are now removed; the same-invocation golden stays intact.
        replay.replay()
        boundaries.flush()
        rows = boundaries.report.get('attention_math_replays', [])
        record['attention_module_reference'] = dict(path='capture/manifest.json',
            records=[dict(layer=row.get('layer'), status=row.get('status'), error=row.get('error')) for row in rows])
    if live_calls is not None:
        from .layer_probe.fused_value_control import run_control
        if probe_mask:
            probe_mode = 'mask'
        control = record[f'fused_{probe_mode}_control'] = dict(status='PARTIAL')
        try:
            run_control(npu, live_calls, root, record['attention_audit'], fused_value_baseline, control,
                        probe_mode=probe_mode)
        except Exception as exc:
            control['error'] = f'{type(exc).__name__}: {exc}'
    return record


def generate(source_root, config_path, input_ids, sequence_length, output, *, device='npu:0',
             intermediates=False, capture_boundaries=False, prompt_index=0, prompt_repeat=1, l3_repeat=None,
             probe_fused_value=None, probe_fused_mask=None, probe_fused_rows=None):
    root = new_output(output)
    source_root = Path(source_root).resolve(strict=True)
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = source_root/config_path
    record = dict(artifact_kind='qwen35-target-prefill-fixture-v1', status='STARTING',
        stage='PREPARE', phase='prefill', layers=[0, 1, 2, 3],
        golden_source='ORIGINAL_310P_SAME_INVOCATION', reference_math='ORIGINAL_CALL_UNCHANGED',
        reference_qualification='ORIGINAL_EXECUTION_ONLY; WHOLE_MODEL_REFERENCE_UNVERIFIED',
        numerical_validation='NOT_RUN', model_compilation='NOT_RUN', cpro_device_execution='NOT_RUN',
        reference_quality='ORIGINAL_REFERENCE_OUTPUT_CAPTURE; NOT_INDEPENDENTLY_VALIDATED_WHOLE_MODEL',
        reference_device=device,
        scope='FOUR_LAYERS_PLUS_ORIGINAL_NORM_HEAD; SINGLE_PREFILL; NO_DECODE',
        config_relative_paths='SOURCE_ROOT_AS_ORIGINAL_INFERENCE', steps=[])

    def stage(name):
        record['stage'] = name
        record['steps'].append(name)
        save_json(root/'manifest.json', record)
        print('[target-prefill] ' + name, flush=True)

    try:
        import yaml
        cfg = yaml.safe_load(config_path.read_text())
        for name in ('model_file', 'quanted_pth', 'embedding_weight_path', 'embedding_scale_path'):
            path = Path(cfg[name])
            cfg[name] = str((path if path.is_absolute() else source_root/path).resolve(strict=True))
        max_len = cfg['kv_cache_max_len']
        if type(max_len) is not int or max_len <= 0 or max_len % 32:
            raise ValueError('kv_cache_max_len must be a positive multiple of 32')
        if sequence_length is not None and not 2 <= sequence_length <= min(max_len, 32767):
            raise ValueError('invalid Prefill sequence-length for this cache/int16 length input')
        path = None
        fused_baseline = None
        probes = {name: fixture for name, fixture in (('value', probe_fused_value),
                  ('mask', probe_fused_mask), ('rows', probe_fused_rows)) if fixture is not None}
        probe_mode, probe_fixture = next(iter(probes.items()), ('value', None))
        if probe_fixture is not None:
            from .layer_probe.fused_value_control import previous_call
            if input_ids is not None or len(probes) != 1:
                raise ValueError('Supply one fused-control fixture as the token source')
            input_ids, fused_baseline = previous_call(probe_fixture)
            record[f'fused_{probe_mode}_baseline_fixture'] = str(Path(probe_fixture).resolve())
        if input_ids is not None:
            if prompt_index != 0 or prompt_repeat != 1:
                raise ValueError('prompt options cannot be combined with --input-ids')
            path = Path(input_ids).resolve(strict=True)
            size = path.stat().st_size
            if size % 4 or not 2 <= size // 4 <= min(max_len, 32767):
                raise ValueError('input-ids must contain 2..cache-capacity little-endian INT32 tokens')
            if sequence_length is not None and size != sequence_length * 4:
                raise ValueError('input-ids byte size differs from sequence-length * sizeof(INT32)')
            sequence_length = size // 4
        stage('IMPORT_ORIGINAL_SOURCE')
        source = import_source(source_root)
        import torch_npu as npu
        from .layer_probe.qlinear_backend import runtime
        record['runtime'] = runtime(npu)
        files = [source_root/'inference.py', source_root/'utils.py', config_path,
                 Path(inspect.getfile(source.Qwen3_5MoeForCausalLMWrapper)),
                 source_root/'models/modeling_qwen3_5_moe_hiai_nz.py']
        record['sources'] = [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in files]
        record['resources'] = {name: cfg[name] for name in ('model_file', 'quanted_pth', 'embedding_weight_path', 'embedding_scale_path')}
        source.device, source.quant, source.penalty = device, 'enable', 1.0
        source.config_data = cfg
        tokens = None
        if path is None:
            stage('TOKENIZE_YAML_PROMPT')
            with hf_config_prefix(cfg['model_file'], 4, record), working_directory(source_root):
                tokens, token_source = prompt_tokens(cfg, source.AutoTokenizer, max_len,
                    prompt_index=prompt_index, prompt_repeat=prompt_repeat, sequence_length=sequence_length)
            sequence_length = tokens.shape[1]
        else:
            token_source = dict(source='INPUT_IDS_FILE', original_path=str(path), original_sha256=sha(path),
                                sequence_length=sequence_length)
        record['token_source'] = token_source
        record['sequence_length'] = sequence_length
        print(f'[target-prefill] sequence_length={sequence_length} source={token_source["source"]}', flush=True)
        stage('LOAD_FOUR_LAYERS_ON_CPU')
        with hf_config_prefix(cfg['model_file'], 4, record), working_directory(source_root):
            wrapper = build_four_layers(source, cfg['model_file'], record)
        model_config = wrapper.model.config
        if model_config.kv_cache_max_len != max_len:
            raise ValueError('YAML cache capacity differs from the constructed model/block table')
        if tokens is None:
            tokens = load_tokens(path, sequence_length, max_len, model_config.vocab_size)
        if torch.any(tokens < 0) or torch.any(tokens >= min(model_config.vocab_size, 2**31)):
            raise ValueError('input token is outside the model vocabulary/INT32 serialization range')
        token_path = root/'input_ids.bin'
        with token_path.open('xb') as stream:
            tokens.numpy().astype('<i4').tofile(stream)
        record['tokens'] = dict(**identity(token_path, root), dtype='int32',
                               shape=list(tokens.shape), ids=tokens[0].tolist())
        stage('ORIGINAL_W4_QUANTIZATION')
        source.quant_model_inplace(wrapper, cfg['quanted_pth'], model_config.num_experts,
            start_layer_idx=0, device='cpu', pseudo_quant=False, quant_bits=4)
        stage('MOVE_QUANTIZED_FOUR_LAYERS')
        wrapper.to(device)
        stage('LOAD_ORIGINAL_EMBEDDING')
        vocab, hidden = model_config.vocab_size, model_config.hidden_size
        for key, dtype, width in (('embedding_weight_path', np.int8, hidden), ('embedding_scale_path', np.float32, 1)):
            if Path(cfg[key]).stat().st_size != vocab * width * np.dtype(dtype).itemsize:
                raise ValueError(f'{key} byte size differs from the model vocabulary/hidden size')
        source.quant_embedding_weights = torch.from_numpy(np.fromfile(cfg['embedding_weight_path'], dtype=np.int8).reshape(vocab, hidden)).to(device)
        source.quant_embedding_scale = torch.from_numpy(np.fromfile(cfg['embedding_scale_path'], dtype='<f4').reshape(vocab, 1)).to(device)
        stage('ORIGINAL_PREFILL_ONCE')
        execute_prefill(source, wrapper, tokens.to(device), root, record, intermediates=intermediates,
                        npu=npu, capture_boundaries=capture_boundaries, l3_repeat=l3_repeat,
                        fused_value_baseline=fused_baseline, probe_mode=probe_mode)
        status = 'CAPTURED' if record.get('boundary_capture', {}).get('status', 'CAPTURED') == 'CAPTURED' else 'PARTIAL'
        if capture_boundaries:
            replay_rows = record.get('attention_module_reference', {}).get('records', [])
            if len(replay_rows) != 1 or replay_rows[0]['status'] != 'COMPUTED':
                status = 'PARTIAL'
        if probe_fixture is not None and record[f'fused_{probe_mode}_control']['status'] != 'INSPECTED':
            status = 'PARTIAL'
        record.update(status=status, stage='COMPLETE', reference_execution='EXECUTED',
            next_action='Bind inputs by semantic name to the actual OMC runtime ABI, then compare its matching Prefill outputs.')
    except Exception as exc:
        record.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
    save_json(root/'manifest.json', record)
    return record
