"""One complete, source-bound GatedDeltaNet layer with explicit state outputs.

Assemble an existing invocation's boundary records; never recompute its golden.
Private model expressions are extracted at runtime and are not vendored here.
"""
from __future__ import annotations

import ast
import copy
import importlib
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

from .common import checked, sha
from .source_regions import assignment, call, definition, one

PROJECTIONS = ('in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj')


def assemble(manifest):
    records = []
    for core in manifest['regions']:
        if core['point'] != 'gdn':
            continue
        siblings = [r for r in manifest['regions'] if all(r[k] == core[k] for k in ('layer', 'call', 'phase'))]

        def region(point):
            matches = [r for r in siblings if r['point'] == point]
            if len(matches) != 1:
                raise ValueError(f'{core["id"]}: needs one captured {point} from the same invocation')
            return matches[0]

        def tensor(r, direction, name, renamed=None):
            matches = [s for s in r[direction] if s['name'] == name]
            if len(matches) != 1:
                raise ValueError(f'{r["id"]}: missing {direction}/{name}')
            return dict(matches[0], name=renamed or name)

        projections = [region('linear_attn_' + name) for name in PROJECTIONS]
        hidden = tensor(projections[0], 'inputs', 'x', 'hidden_states')
        # All four projections must consume the identical captured layer input.
        for r in projections[1:4]:
            other = tensor(r, 'inputs', 'x')
            if any(hidden[k] != other[k] for k in ('shape', 'dtype', 'bytes', 'sha256')):
                raise ValueError('whole GDN projections do not share one captured hidden input')
        conv = region('convstate_update')
        dependencies = projections + [conv, region('conv1d'), region('decay_beta'), region('linear_attn_norm'), core]
        if any(r.get('sequence_length') != core.get('sequence_length') for r in dependencies):
            raise ValueError('whole GDN capture sequence lengths differ')
        state = tensor(core, 'outputs', 'golden_state_raw', 'golden_recurrent_state_fp32')
        if state['dtype'] != 'float32':
            raise ValueError('whole GDN comparison requires an original FP32 raw-state golden')
        records.append(dict(core, id=core['id'].removesuffix('_gdn') + '_gdn_layer',
            point='gdn_layer', kind='gdn_layer', attrs=dict(core['attrs']),
            inputs=[hidden, tensor(conv, 'inputs', 'old_conv_state'),
                    tensor(core, 'inputs', 'initial_state'), tensor(conv, 'inputs', 'linear_update_tensor'),
                    tensor(core, 'inputs', 'effective_length')],
            outputs=[tensor(projections[-1], 'outputs', 'golden_y', 'golden_hidden'),
                     tensor(conv, 'outputs', 'golden_new_conv_state'), state],
            components=dependencies, provenance='SAME_CAPTURE_LAYER_CALL_PHASE; ORIGINAL_EXECUTED_OUTPUTS'))
    if not records:
        raise ValueError('capture has no GDN invocations for whole-layer replay')
    return records


def single_token_gdr(query, key, value, g, beta, chunk_size, initial_state, step_length,
                     output_final_state, use_qk_l2norm_in_kernel):
    import torch_npu

    if chunk_size != 64 or not output_final_state:
        raise ValueError('single-token Prefill expects the original chunk_size=64 call with final state')
    state = initial_state
    blocks = []
    for t in range(query.shape[1]):
        core, next_state = torch_npu.npu_chunk_gated_delta_rule(
            query[:, t:t+1].contiguous(), key[:, t:t+1].contiguous(), value[:, t:t+1].contiguous(),
            g=g[:, t:t+1].contiguous(), beta=beta[:, t:t+1].contiguous(), chunk_size=1,
            initial_state=state, effective_length=step_length, output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
        blocks.append(core)
        state = next_state.to(initial_state.dtype)
    return torch.cat(blocks, dim=1), state


def adapt_forward(owner, state_dtype='float32', convstate_mode='kernel', gdr_mode='chunk'):
    if state_dtype not in ('float16', 'float32'):
        raise ValueError('whole GDN state must be float16 or float32')
    tree, evidence = definition(owner)
    tree = copy.deepcopy(tree)
    tree.decorator_list = []
    if convstate_mode not in ('kernel', 'reference'):
        raise ValueError('unknown whole GDN ConvState candidate')
    if convstate_mode == 'reference':
        update = call(tree, 'torch.ops.my_ops.linear_attention_convstate_update')
        if ([ast.unparse(a) for a in update.args] != ['mixed_qkv', 'conv_state', 'linear_update_tensor']
                or update.keywords):
            raise ValueError('review changed ConvState update arguments')
        update.func = ast.Name(id='_probe_convstate_reference', ctx=ast.Load())
        update.args = update.args[:2]
        # A name injected only into copied globals cannot be resolved through
        # the original source module by Dynamo's guards. Bind the real helper
        # with a function-local import; leave the source module untouched.
        tree.body.insert(0, ast.ImportFrom(
            module='qwen35_moe_dflash.target.layer_probe.convstate_reference',
            names=[ast.alias(name='tensor_update', asname='_probe_convstate_reference')], level=0))
    # Keep the original FP16 writeback for the installed-kernel candidate.
    state = assignment(tree, 'last_recurrent_state')
    if ast.unparse(state.value) != 'last_recurrent_state.to(torch.float16)':
        raise ValueError('review changed GDN state writeback before applying dtype candidate')
    state.value = ast.parse(f'last_recurrent_state.to(torch.{state_dtype})', mode='eval').body
    op = call(tree, 'torch_npu.npu_chunk_gated_delta_rule')
    initial = [k for k in op.keywords if k.arg == 'initial_state']
    if len(initial) != 1 or ast.unparse(initial[0].value) != 'recurrent_state':
        raise ValueError('review changed GDN initial_state expression')
    if gdr_mode == 'single-token':
        op.func = ast.Name(id='_probe_single_token_gdr', ctx=ast.Load())
        length = one((kw for kw in op.keywords if kw.arg == 'effective_length'), 'GDR effective_length')
        length.arg = 'step_length'
        length.value = ast.parse('self._probe_gdr_step_length', mode='eval').body
        tree.body.insert(0, ast.ImportFrom(
            module=__name__, names=[ast.alias(name='single_token_gdr', asname='_probe_single_token_gdr')], level=0))
    elif gdr_mode != 'chunk':
        raise ValueError('unknown GDR mode')
    # Source ConvState returns NCL, but this DDK infers NC1L for its padded
    # output. Reshape fixes the Conv2D rank in either case; unsqueeze would
    # turn the target's already-4D output into a rejected 5D tensor.
    conv = assignment(tree, 'conv_out')
    if ast.unparse(conv.value) != 'F.conv1d(padded, self.conv1d.weight, groups=self.conv_dim)':
        raise ValueError('review changed whole GDN convolution')
    conv.value = ast.parse('F.conv2d(padded.reshape(batch_size, self.conv_dim, 1, -1), '
                           'self.conv1d.weight.unsqueeze(2), '
                           'groups=self.conv_dim).squeeze(2)', mode='eval').body
    returns = [n for n in tree.body if isinstance(n, ast.Return)]
    if len(returns) != 1 or ast.unparse(returns[0].value) != '(output, past_key_values)':
        raise ValueError('review changed GDN return interface')
    returns[0].value = ast.parse('(output, conv_state, recurrent_state)', mode='eval').body
    module = ast.fix_missing_locations(ast.Module(body=[tree], type_ignores=[]))
    namespace = dict(owner.__globals__)
    exec(compile(module, evidence['file'], 'exec'), namespace)
    evidence.update(adaptations=[('FP16' if state_dtype == 'float16' else 'FP32') + '_RECURRENT_INPUT_AND_WRITEBACK',
                                 'EXPLICIT_CONV2D_SINGLETON_HEIGHT',
                                 'CONVSTATE_PADDED_RESHAPE_NC1L',
                                 'RETURN_HIDDEN_CONV_STATE_RECURRENT_STATE'],
                    conv_input=dict(operation='reshape', shape='[batch_size, conv_dim, 1, -1]',
                                    preserves='ELEMENT_ORDER_AND_DTYPE'),
                    arithmetic_precision='INTERNAL_KERNEL_ACCUMULATION_NOT_ESTABLISHED')
    evidence['convstate_mode'] = convstate_mode
    evidence['gdr_mode'] = gdr_mode
    if gdr_mode == 'single-token':
        evidence['adaptations'].append('SINGLE_TOKEN_GDR_WITH_ORDERED_CORE_CONCAT_AND_RECURRENT_FEEDBACK')
    if convstate_mode == 'reference':
        evidence['adaptations'].append('REFERENCE_CONVSTATE_CONCAT_TAIL; FULL_CAPTURE_LENGTH_ONLY')
        evidence['convstate_binding'] = 'FORWARD_LOCAL_IMPORT; ORIGINAL_SOURCE_MODULE_UNCHANGED'
    return namespace[tree.name], evidence


class WholeGDN(torch.nn.Module):
    def __init__(self, op, state_dtype='float32'):
        super().__init__()
        if state_dtype not in ('float16', 'float32'):
            raise ValueError('whole GDN state must be float16 or float32')
        self.op = op
        self.state_dtype = getattr(torch, state_dtype)
        self.export_adjustments = []

    def forward(self, probe_input_000, probe_input_001, probe_input_002, probe_input_003, probe_input_004):
        conv = probe_input_001.clone()
        state = probe_input_002.to(self.state_dtype).clone()
        return self.op(probe_input_000, past_key_values=(conv, state), export_flag=True,
                       linear_update_tensor=probe_input_003, effective_length=probe_input_004)


def build(record, capture, target, converters, factory, config_path, source_root, root, state_dtype='float32',
          convstate_mode='kernel', gdr_mode='chunk'):
    import yaml
    from .candidates import adapt_qlinear
    from .target_weights import load_weights
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    model_path = Path(config['model_file']).expanduser()
    if not model_path.is_absolute():
        model_path = Path(source_root) / model_path
    wrapper = importlib.import_module('models.export_model_wrapper_qwen3_5_moe')
    _, loader = definition(wrapper._load_text_config)
    if Path(source_root).resolve() not in Path(loader['file']).resolve().parents:
        raise ValueError('whole GDN config loader is outside source-root')
    model_config = wrapper._load_text_config(str(model_path))
    with torch.device('meta'):
        op = target.Qwen3_5MoeGatedDeltaNet(model_config, record['layer'])
    weights, quant = {}, []
    by_point = {r['point']: r for r in record['components']}
    mapping = {**{'linear_attn_' + n: n for n in PROJECTIONS},
               'conv1d': 'conv1d', 'decay_beta': '', 'linear_attn_norm': 'norm'}
    for point, member in mapping.items():
        r = by_point[point]
        captured = torch.load(checked(capture, r['weights']), map_location='cpu', weights_only=True)
        values, evidence = load_weights(r, captured, target, converters, config_path, source_root)
        weights[point] = evidence
        if member in PROJECTIONS:
            if r['kind'] != 'qlinear':
                raise ValueError('whole GDN candidate requires the captured W4 QLinear projections')
            child = factory(W_q=values['W_q'], scale=values['scale'], idx=0)
            setattr(op, member, child)
            folder = root / member
            folder.mkdir()
            quant.append(adapt_qlinear(SimpleNamespace(op=child), folder))
        else:
            parent = op.get_submodule(member) if member else op
            for name, value in values.items():
                if name not in parent._parameters:
                    raise ValueError('unbound whole GDN parameter: ' + point + '.' + name)
                setattr(parent, name, torch.nn.Parameter(value, requires_grad=False))
    if any(t.is_meta for t in (*op.parameters(), *op.buffers())):
        raise ValueError('whole GDN has unbound meta weights')
    forward, source = adapt_forward(target.Qwen3_5MoeGatedDeltaNet.forward, state_dtype, convstate_mode, gdr_mode)
    if gdr_mode == 'single-token':
        # B=1, each unrolled call consumes one valid token. Materialize this
        # before tracing so the X90 graph never has to execute OnesLike.
        op.register_buffer('_probe_gdr_step_length', torch.tensor([1], dtype=torch.int16))
    op.forward = MethodType(forward, op)
    model = WholeGDN(op, state_dtype).eval()
    evidence = dict(status='BOUND_TO_SOURCE', kind='gdn_layer', source=source,
        config_loader=loader, config_sha256=sha(model_path/'config.json'),
        input_scope='CAPTURED_LAYER_HIDDEN_AND_PRE_CALL_STATES; SAME_INVOCATION',
        output_scope='WHOLE_LAYER_HIDDEN_CONV_STATE_AND_RECURRENT_STATE',
        recurrent_state_dtype=state_dtype, golden_state_dtype='float32',
        weight_scope='ORIGINAL_CHECKPOINT_BOUND', weights=weights,
        qlinear_candidates=quant, golden_state_feedback='CAPTURED_310P_TRAJECTORY',
        cpro_state_feedback='NOT_TESTED')
    return model, evidence
