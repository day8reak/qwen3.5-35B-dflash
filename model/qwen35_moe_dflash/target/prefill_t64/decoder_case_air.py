"""Export complete private CPro decoder blocks with named cache and hidden boundaries."""
from contextlib import nullcontext
from pathlib import Path
import json
import os
import sys
import tempfile
from unittest.mock import patch

import torch

from ...common.artifacts import files_under, working_directory
from ...draft.incremental.torch_weight_export import external_weight_export
from ..layer_probe.common import checked, read_array, save_json
from ..layer_probe.replay import export_input_identity
from .chain_air import Head, checkpoint_paths
from .decoder_cases import case_name, cpro_input_records, groups, output_names, state_roles
from .decoder_source import inspect_shared_gates
from .decoder_four_boundary import data_edges
from .decoder_four_silu import inspect_activation_air
from .decode_cache_control import graph_message
from .l0_split import prefill_precision


class DecoderGraph(torch.nn.Module):
    def __init__(self, blocks, head, group, phase, split, valid_rows):
        super().__init__()
        self.blocks = torch.nn.ModuleList(blocks)
        self.head = head
        self.group = group
        self.prefill = phase == 'prefill'
        self.valid_rows = valid_rows
        self.split = split

    def observe_projection(self, module, args, result):
        self.projection = result[0]

    def forward(self, hidden, **feeds):
        outputs = {}
        for i, block in zip(self.group, self.blocks):
            if self.prefill and i == self.split:
                hidden = torch.cat((hidden[:, :self.valid_rows], torch.zeros_like(hidden[:, self.valid_rows:])), dim=1)
            roles = state_roles(i)
            cache = tuple(feeds[key].clone() for key in roles)
            if i % 4 == 3:
                args = dict(position_ids=feeds['position_ids'].int(), attention_mask=feeds['attention_mask'],
                            new_kv_cache_pos=feeds['new_kv_cache_pos'].reshape(1, -1),
                            allQLen=feeds['all_length'])
            else:
                args = dict(effective_length=feeds['effective_length'].short())
            hidden, pair = block(hidden, past_key_values=cache, use_cache=True, export_flag=True, **args)
            if len(self.group) == 1:
                outputs[f'out_proj_{i}'] = self.projection
            outputs.update(zip(roles, pair))
        outputs[f'hidden_{self.group[-1]}'] = hidden
        if self.head is not None:
            outputs['logits'] = self.head(hidden[:, self.valid_rows-1, :])
        return tuple(torch.ops.decoder_compare.readout(outputs[name], name)
                     for name in output_names(self.group, self.head is not None))


def bind_outputs(graph, converter_edges):
    nodes = {op.name: op for op in graph.op}
    net, = [op for op in graph.op if op.type == 'NetOutput']
    by_edge = {edge: name for name, edge in converter_edges.items()}
    result = []
    for edge in data_edges(net):
        producer = edge
        while producer not in by_edge:
            op = nodes[producer.rsplit(':', 1)[0]]
            if op.type not in ('Identity', 'TensorMove') or len(data_edges(op)) != 1:
                raise ValueError(f'Unbound public AIR output {edge}')
            producer, = data_edges(op)
        result.append(dict(name=by_edge[producer], edge=edge))
    if sorted(r['name'] for r in result) != sorted(converter_edges):
        raise ValueError('Public decoder outputs were dropped or repeated')
    return result


def export_case(dest, fixtures, ref, group, final, blocks, head, exporter, edges, gdn_conv):
    rows = 64 if ref['phase'] == 'prefill' else 1
    name = case_name(group)
    linear = sum(i % 4 != 3 for i in group)
    print(f'Export {name}, S={rows}', flush=True)
    module = DecoderGraph(blocks, head, group, ref['phase'], ref['split_layer'], ref['valid_rows'])
    names = ['hidden']
    if linear:
        names.append('effective_length')
    if linear != len(group):
        names.extend(('position_ids', 'attention_mask', 'new_kv_cache_pos', 'all_length'))
    for i in group:
        names.extend(state_roles(i))
    # Example tensors establish the static ABI. Board replay feeds the preceding CPro hidden.
    inputs = cpro_input_records(ref)
    args = {k: torch.from_numpy(read_array(fixtures, inputs[k])) for k in names}
    edges.clear()
    trace = dest/'activations.txt'
    torch._dynamo.reset()
    tap = nullcontext()
    if len(group) == 1:
        mixer = blocks[0].linear_attn if linear else blocks[0].self_attn
        tap = mixer.register_forward_hook(module.observe_projection)
    with (tap, working_directory(dest), patch.dict(os.environ, QWEN35_SILU_SHAPES=str(trace)),
          external_weight_export(), export_input_identity(list(args.values())) as evidence):
        with exporter.cpro_export_contracts(True, chunk_size=rows, gdn_layers=linear, sequence_rows=rows):
            exporter.torchair.dynamo_export(model=module.eval(), export_path='weight', dynamic=False, **args)
    air = dest/'weight/export.air'
    _, graph = graph_message(air.read_bytes(), 'NetOutput')
    audit = prefill_precision(graph, group, final, rows=rows)
    gates = inspect_shared_gates(graph, len(group))
    if audit['mismatches']:
        raise ValueError('\n'.join(audit['mismatches']))
    linear_prefill = (ref['mode'], ref['phase']) == ('linear', 'prefill')
    operators = inspect_activation_air(air, trace, layers=len(group), full_only=linear == 0, softplus=True, rows=rows,
                                       check_convs=True, gdn_conv=gdn_conv,
                                       rms_silu_token_view=linear_prefill, decay_token_view=linear_prefill)
    bindings = {node: list(args)[index] for node, index in evidence['nodes'].items()}
    if (set(bindings.values()) != set(args) or set(bindings) !=
            {n.name for n in graph.op if n.type in ('Data', 'RefData')}):
        raise ValueError(f'{name}: export inputs differ from the actual decoder inputs')
    if set(edges) != set(output_names(group, final)):
        raise ValueError(f'{name}: missing converter output identities')
    save_json(dest/'case.json', dict(layers=group, shared_gates=gates, gdn_convs=operators['gdn_convs'], inputs=bindings, outputs=bind_outputs(graph, edges),
                                     files=files_under(dest, dest)))


@torch.no_grad()
def export(root, fixtures, ref, source, config, prepared, *, gdn_conv='conv2d'):
    plan = groups(ref['mode'])
    paths = checkpoint_paths(source, config, ref)
    sys.path.insert(0, str(prepared))
    try:
        import export_glm_model_qwen3_5_moe as exporter
        from models.export_model_wrapper_qwen3_5_moe import load_npu_model
        from models.modeling_qwen3_5_moe_hiai_cpro import Qwen3_5MoeRMSNorm

        lib = torch.library.Library('decoder_compare', 'DEF')
        lib.define('readout(Tensor x, str role) -> Tensor')
        lib.impl('readout', lambda x, role: torch.empty_like(x), 'Meta')
        edges = {}

        @exporter.torchair.register_fx_node_ge_converter(torch.ops.decoder_compare.readout.default)
        def record_edge(x, role):
            edges[role] = x.tensor
            return x

        for start in range(0, len(plan), 4):
            batch = plan[start:start+4]
            pending = []
            for group in batch:
                case = root/case_name(group)
                if case.exists():
                    for row in json.loads((case/'case.json').read_bytes())['files']:
                        checked(case, row)
                else:
                    pending.append(group)
            if not pending:
                continue
            first, last = pending[0][0], pending[-1][-1]
            model = load_npu_model(str(paths['model_file']), 'cpu', torch.float16, variant='cpro',
                                   layer_num=last-first+1, start_layer_idx=first, quant_enable=True)
            exporter.quant_model_inplace(model, str(paths['quanted_pth']), model.config.num_experts,
                start_layer_idx=first, device='cpu', quant_bits=4, qlinear_cls=exporter.CproQLinear)
            model.eval().to(dtype=torch.float16)
            for module in model.modules():
                if isinstance(module, Qwen3_5MoeRMSNorm):
                    gamma = (module.weight + 1.0).detach().contiguous()
                    module.register_buffer('export_gamma', gamma, persistent=False)
            types = ['full_attention' if i % 4 == 3 else 'linear_attention' for i in range(first, last+1)]
            if model.model.start_layer_idx != first or [m.block_type for m in model.model.layers] != types:
                raise ValueError(f'Incorrect CPro checkpoint slice L{first}–L{last}')
            for group in pending:
                final = group == plan[-1]
                blocks = model.model.layers[group[0]-first:group[-1]-first+1]
                with tempfile.TemporaryDirectory(dir=root.parent, prefix='decoder-air-') as tmp_case:
                    dest = Path(tmp_case)/'case'
                    dest.mkdir()
                    export_case(dest, fixtures, ref, group, final, blocks, Head(model) if final else None,
                                exporter, edges, gdn_conv)
                    dest.rename(root/case_name(group))
            del blocks, model
    finally:
        sys.path.remove(str(prepared))
