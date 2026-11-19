"""Same-call GDR operands and post-GDR boundaries for segmented physical64/valid19."""
import ast

import numpy as np

from ..layer_probe.common import metrics

OPERANDS = ('query', 'key', 'value', 'g', 'beta')
TAPS = (*OPERANDS, 'core', 'norm_gate', 'out_proj_input')
BASIS = 'SEGMENT_310P_SAME_CALL_GDN_TRACE_VALID19'


def observe_tail(module, snapshot, cleanup):
    from ..layer_probe.capture import cpu

    def once(op, callback):
        def observed(mod, args):
            handle.remove()
            callback(mod, args)
        handle = op.register_forward_pre_hook(observed)
        cleanup.callback(handle.remove)

    def norm(mod, args):
        x, z = args
        snapshot.update(norm_input=cpu(x), norm_gate=cpu(z), gamma=cpu(mod.weight),
                        epsilon=float(mod.variance_epsilon))

    def project(mod, args):
        x, = args
        snapshot['out_proj_input'] = cpu(x)

    once(module.norm, norm)
    once(module.out_proj, project)


def save_trace(root, manifest, layer, snapshot):
    from .segment_gdr import audit_operands
    from .segment_prefix import tensor_record, same_bits
    from .gdn_tail import _gate_norm

    call, = manifest[f'l{layer}_gdr']['calls']
    if 'error' in call or 'gdr' not in snapshot:
        raise ValueError(f'L{layer}: could not capture the GDR operands: {call}')
    data = snapshot['gdr']
    if (call['effective_length'] != [19] or call['physical_q_rows'] != 64
            or data['attrs']['chunk_size'] != 64 or not data['attrs']['output_final_state']
            or not data['attrs']['use_qk_l2norm_in_kernel']):
        raise ValueError(f'L{layer}: unexpected GDR call: {call}')
    core, state = data['outputs']
    audits, ok = audit_operands(data['inputs'], core, state)
    norm_x = snapshot['norm_input'].reshape_as(core)
    if not same_bits(core[:, :19], norm_x[:, :19]):
        raise ValueError(f'L{layer}: valid GDR core differs from the actual gated norm input')
    gate = snapshot['norm_gate'].reshape(core.shape)
    proj = snapshot['out_proj_input'].reshape(core.shape)
    cpu_norm = _gate_norm(core[:, :19].numpy(), gate[:, :19].numpy(), snapshot['gamma'].numpy(),
                         snapshot['epsilon'], np.float16)
    audits['norm_on_captured_core_and_gate'] = metrics(proj[:, :19].numpy(), cpu_norm)
    stages = {**data['inputs'], 'core': core, 'norm_gate': gate, 'out_proj_input': proj}
    specs = {name: tensor_record(root, 'trace_' + name, stages[name].reshape(1, 64, -1)[:, :19])
             for name in TAPS}
    return dict(layer=layer, tensors=specs, gamma=tensor_record(root, 'trace_gamma', snapshot['gamma']),
                epsilon=snapshot['epsilon'], gdr_cpu_passed=ok, cpu_audit=audits)


def save_projection(output, root, trace, cpu_model, prefix, layer, native):
    """Keep only the layer's output projection for fixed-input CPU replay after board recovery."""
    import torch
    from ..layer_probe.common import identity, read_array
    from .cpu_reference import linear

    weight, scale = cpu_model.weights.logical('target.' + prefix + '.out_proj.weight', tuple(layer.out_proj.weight.shape))
    x = read_array(root, trace['tensors']['out_proj_input'])
    golden = native[:, :19]
    result = linear(torch.from_numpy(x), weight, scale).numpy()
    report = dict(cpu_on_captured_input=metrics(result, golden))
    for name, array in (('weight', weight), ('scale', scale)):
        path = output / ('trace_out_proj_' + name + '.bin')
        array.tofile(path)
        report[name] = dict(identity(path, output), dtype=array.dtype.name, shape=list(array.shape))
    return report


def instrument_chunk(forward, gdr):
    """Retain source operands exactly once and keep the source hidden expression intact."""
    from ..layer_probe.whole_gdn import trace_gdn_tail
    trace_gdn_tail(forward, gdr)
    for slot, name in enumerate(OPERANDS):
        if slot < len(gdr.args):
            expr = gdr.args[slot]
        else:
            kw, = [kw for kw in gdr.keywords if kw.arg == name]
            expr = kw.value
        captured = ast.NamedExpr(ast.Name('_trace_' + name, ast.Store()), expr)
        if slot < len(gdr.args):
            gdr.args[slot] = captured
        else:
            kw.value = captured
    ret, = [s for s in forward.body if isinstance(s, ast.Return)]
    if ast.unparse(ret.value) != '(output, past_key_values)':
        raise ValueError('Changed whole GDN return contract')
    # g is computed in FP32 in the source. The CPro converter casts it to FP16.
    expressions = ['_trace_query', '_trace_key', '_trace_value', '_trace_g.to(torch.float16)',
                   '_trace_beta', '_probe_gdr_outputs[0]', '_probe_norm_gate', '_probe_proj_input']
    ret.value.elts.extend(ast.parse(f'{expr}.reshape(1, 64, -1)[:, :19].contiguous()', mode='eval').body
                          for expr in expressions)
