"""Call the supplied inference functions directly, without executing its main."""
from contextlib import contextmanager
import importlib.util
import inspect
import os
from pathlib import Path
import sys

import numpy as np
import torch

from .common import identity, new_output, save_json, sha


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def import_source(root):
    """Import definitions and original imports; __main__ never runs."""
    path = Path(root)/'inference.py'
    spec = importlib.util.spec_from_file_location('_target_original_inference', path)
    source = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = source
    sys.path.insert(0, str(root))
    try:
        with working_directory(root):
            spec.loader.exec_module(source)
    finally:
        sys.path.pop(0)
    for name in ('prefill', 'decoder', 'create_strict_causal_mask', 'quant_model_inplace'):
        if not callable(getattr(source, name, None)):
            raise ValueError(f'original inference.py must expose {name}')
    return source


def build_four_layers(source, model_path, record):
    """Limit construction at the original factory, then restore it immediately."""
    cls = source.Qwen3_5MoeForCausalLMWrapper
    owner = sys.modules[cls.__module__]
    original = owner.load_npu_model
    def load(*args, **kwargs):
        values = inspect.signature(original).bind(*args, **kwargs)
        values.apply_defaults()
        if 'layer_num' not in values.arguments or values.arguments.get('start_layer_idx') != 0:
            raise ValueError('original loader must support layer_num and start_layer_idx=0')
        values.arguments.update(layer_num=4, device='cpu')
        model = original(*values.args, **values.kwargs)
        if len(model.model.layers) != 4 or model.config.num_hidden_layers < 4:
            raise ValueError('original loader did not construct exactly four layers')
        record['checkpoint_layers'] = model.config.num_hidden_layers
        record['constructed_layers'] = 4
        model.config.num_hidden_layers = 4
        return model
    owner.load_npu_model = load
    try:
        return cls(model_path=str(model_path), device='cpu', dtype=torch.float16,
            embedding_config={'embedding_in_omc': False, 'mul_twice': False}, quant_enable=True).eval()
    finally:
        owner.load_npu_model = original


def initial_state(config, max_len, device):
    """Match the supplied four-layer reference's cache ABI, fail on a different model."""
    pattern = ['linear_attention'] * 3 + ['full_attention']
    if list(config.layer_types[:4]) != pattern:
        raise ValueError('original prefill dumps expect three linear layers followed by one full-attention layer')
    conv_dim = 2*config.linear_num_key_heads*config.linear_key_head_dim + config.linear_num_value_heads*config.linear_value_head_dim
    conv = (1, conv_dim, config.linear_conv_kernel_dim)
    state = (1, config.linear_num_value_heads, config.linear_key_head_dim, config.linear_value_head_dim)
    if conv != (1, 8192, 4) or state != (1, 32, 128, 128):
        raise ValueError('model cache dimensions differ from the supplied inference.py reference')
    if max_len % 32:
        raise ValueError('original reference requires kv_cache_max_len divisible by 32')
    kv = (max_len//32, config.num_key_value_heads*config.head_dim//16, 32, 16)
    shapes = [(conv, state)]*3 + [(kv, kv)]
    return [tuple(torch.zeros(s, dtype=torch.float16, device=device) for s in pair) for pair in shapes]


def execute_steps(source, wrapper, input_ids, root, max_token, stage=None):
    """Original prefill -> its actual logits/caches -> original decoder, once each."""
    max_len = source.config_data['kv_cache_max_len']
    if input_ids.shape[1] < 2 or input_ids.shape[1] + max_token > max_len:
        raise ValueError('reference requires at least two prompt tokens and space for every decode step')
    caches = initial_state(wrapper.model.config, max_len, source.device)
    mask = source.create_strict_causal_mask(source.real_seq_len, max_len, 'cpu')[None, None]
    source.input_ids = input_ids  # Original decoder uses this global when saving output_pos.
    for name in ('prefill', 'decode'):
        (root/name).mkdir()
    if stage:
        stage('ORIGINAL_PREFILL')
    source.output_path = str(root/'prefill') + os.sep
    with working_directory(root/'prefill'):
        logits, _, caches = source.prefill(wrapper, input_ids, mask, caches, max_len)
    # Original inference loop chooses argmax(logits), not prefill's auxiliary return.
    first_token = torch.argmax(logits, dim=-1)
    torch.save(dict(logits=logits.detach().cpu(), first_token=first_token.detach().cpu()), root/'prefill-result.pt')
    source.output_path = str(root/'decode') + os.sep
    if stage:
        stage('ORIGINAL_DECODE')
    with working_directory(root/'decode'):
        generated = source.decoder(wrapper, first_token, input_ids.shape[1], mask, caches, max_token, max_len)
    torch.save(generated.detach().cpu(), root/'generated-ids.pt')
    return generated


def run_steps(source_root, config_path, output, *, device='npu:0', max_token=1,
              prompt_index=0, prompt_repeat=32, input_ids_path=None, baseline_only=False):
    if max_token < 1 or prompt_index < 0 or prompt_repeat < 1:
        raise ValueError('max_token/prompt_repeat must be positive; prompt_index must be nonnegative')
    root = new_output(output)
    source_root = Path(source_root).resolve()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = source_root/config_path
    record = dict(status='STARTING', stage='IMPORT_SOURCE', steps=[],
        execution='ORIGINAL_FUNCTIONS_EAGER', observe=not baseline_only,
        scope='FOUR_LAYERS_PREFILL_AND_DECODE; NO_GRAPH_EXPORT',
        backend_policy='ORIGINAL_CALL_NO_API_BRIDGE_NO_GROUP_OVERRIDE',
        numerical_validation='NOT_VERIFIED_AGAINST_EXISTING_GOLDEN')
    capture, npu = None, None
    def stage(name):
        record['stage'] = name
        record['steps'].append(name)
        save_json(root/'steps.json', record)
        print('[target-steps] ' + name, flush=True)
    try:
        stage('IMPORT_SOURCE')
        source = import_source(source_root)
        import torch_npu as npu
        import yaml
        cfg = yaml.safe_load(config_path.read_text())
        def resolve(value):
            path = Path(value)
            return path if path.is_absolute() else source_root/path
        for name in ('model_file', 'quanted_pth', 'embedding_weight_path', 'embedding_scale_path'):
            cfg[name] = str(resolve(cfg[name]))
        files = [source_root/'inference.py', source_root/'utils.py', config_path,
                 Path(inspect.getfile(source.Qwen3_5MoeForCausalLMWrapper)),
                 source_root/'models/modeling_qwen3_5_moe_hiai_nz.py']
        record['sources'] = [dict(path=str(p), sha256=sha(p)) for p in files]
        from .qlinear_backend import runtime
        record['runtime'] = runtime(npu)
        source.device, source.quant, source.penalty = device, 'enable', 1.0
        source.config_data = cfg
        source.tokenizer = source.AutoTokenizer.from_pretrained(cfg['model_file'])
        stage('LOAD_FOUR_LAYERS_ON_CPU')
        with working_directory(source_root):
            wrapper = build_four_layers(source, cfg['model_file'], record)
        if wrapper.model.config.kv_cache_max_len != cfg['kv_cache_max_len']:
            raise ValueError('config cache length differs from constructed model; do not silently change its block table')
        stage('ORIGINAL_W4_QUANTIZATION')
        source.quant_model_inplace(wrapper, cfg['quanted_pth'], wrapper.model.config.num_experts,
            start_layer_idx=0, device='cpu', pseudo_quant=False, quant_bits=4)
        stage('MOVE_QUANTIZED_FOUR_LAYERS')
        wrapper.to(device)  # No dtype cast; only four quantized layers have been constructed.
        vocab = wrapper.model.config.vocab_size
        stage('LOAD_ORIGINAL_EMBEDDING')
        source.quant_embedding_weights = torch.tensor(np.fromfile(cfg['embedding_weight_path'], dtype=np.int8).reshape(vocab, -1), dtype=torch.int8).to(device)
        source.quant_embedding_scale = torch.tensor(np.fromfile(cfg['embedding_scale_path'], dtype=np.float32).reshape(vocab, -1), dtype=torch.float32).to(device)
        stage('PREPARE_INPUT')
        if input_ids_path:
            path = Path(input_ids_path).resolve()
            if path.stat().st_size % 4 or path.stat().st_size == 0:
                raise ValueError('input_ids.bin must contain nonempty INT32 tokens for batch=1')
            ids = torch.from_numpy(np.fromfile(path, dtype=np.int32).astype(np.int64)).unsqueeze(0).to(device)
            source.real_seq_len = ids.shape[1]
            record['input_source'] = dict(path=str(path), sha256=sha(path), dtype='int32', batch=1)
        else:
            # The supplied main overwrites its chat template with prompt * 32.
            text = cfg['prompts'][prompt_index] * prompt_repeat
            tokens = source.tokenizer([text], return_tensors='pt')
            ids = tokens['input_ids'].to(device)
            source.real_seq_len = tokens['attention_mask'].sum().item()
            record['input_source'] = dict(prompt_index=prompt_index, prompt_repeat=prompt_repeat,
                policy='RAW_PROMPT_REPEAT_AS_SUPPLIED_MAIN; NO_CHAT_TEMPLATE_OR_PADDING')
        ids.cpu().to(torch.int32).numpy().tofile(root/'input_ids.bin')
        record['input_ids'] = dict(identity(root/'input_ids.bin', root), shape=list(ids.shape), real_seq_len=source.real_seq_len)
        if not baseline_only:
            from .capture import Capture
            capture = Capture(root/'capture', layers=[0, 1, 2, 3], calls=max_token+1)
            capture.report['entrypoint'] = 'DIRECT_ORIGINAL_PREFILL_DECODER'
            capture.report['reference_model_layers'] = 4
            capture.report['placement'] = dict(policy='CPU_CONSTRUCT_FOUR_QUANTIZE_THEN_MOVE',
                executed_num_hidden_layers=4, decoder_layer_limit=4,
                dtype_policy='DEVICE_ONLY_NO_CAST')
            for path in files:
                capture.source(path)
            owner = sys.modules[type(wrapper).__module__]
            nz = sys.modules['models.modeling_qwen3_5_moe_hiai_nz']
            capture.install(npu, nz, owner)  # Observations only; no group_sizes override.
        stage('ORIGINAL_PREFILL_AND_DECODE')
        originals = root/'original'
        originals.mkdir()
        generated = execute_steps(source, wrapper, ids, originals, max_token, stage)
        record.update(status='EXECUTED', stage='COMPLETE', generated_ids=generated.detach().cpu().tolist(),
            text=source.tokenizer.decode(generated.detach().cpu().tolist(), skip_special_tokens=False))
    except Exception as exc:
        record.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
    finally:
        if capture is not None:
            capture.close(record.get('error'))
            record['capture_status'] = capture.report['status']
            if record['status'] == 'EXECUTED' and record['capture_status'] != 'CAPTURED':
                record['status'] = 'INCOMPLETE'
        if npu is not None:
            try:
                from .qlinear_backend import runtime
                record['runtime_at_end'] = runtime(npu)
            except Exception as exc:
                record['runtime_diagnostic_error'] = str(exc)
        save_json(root/'steps.json', record)
    return record
