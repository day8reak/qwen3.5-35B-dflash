"""Build a selectable per-port CSV from the installed L0 and L3 attention AIRs."""
import csv
from functools import lru_cache
import json
from pathlib import Path
import tempfile

from .attention_slice import CASE_IDS, DTYPES, descriptor, port_info, source_graph
from .decode_cache_trace import _project_input
from .gdr_stage_air import stage_edges
from .qkv_silu import activation_edges


class Catalog:
    def __init__(self, graph):
        self.graph = graph
        self.nodes = {n.name: n for n in graph.op}

        @lru_cache(None)
        def constant(name):
            n = self.nodes[name]
            if n.type in ('Const', 'Constant', 'FileConstant'):
                return True
            return (bool(n.input) and not n.subgraph_name and n.type not in ('Data', 'NetOutput')
                    and all(e and not e.endswith(':-1') and constant(e.rsplit(':', 1)[0]) for e in n.input))
        self.constant = constant

    def spec(self, edge, consumer=None, slot=None):
        d = descriptor(self.nodes, edge)
        declared = port_info(self.graph, d)
        if consumer is not None and slot < len(consumer.input_desc):
            use = port_info(self.graph, consumer.input_desc[slot])
            if not d.dtype:
                declared['dtype'] = use['dtype']
            if not declared['shape']:
                declared['shape'] = use['shape']
        dtype = next((k for k, v in DTYPES.items() if v == declared['dtype']), None)
        return dict(edge=edge, dtype=dtype or declared['dtype'], shape=declared['shape'])

    def row(self, step, target, inputs, description, *, constants=(), manual=False):
        op = self.nodes[target.rsplit(':', 1)[0]]
        missing = [s['edge'] for s in inputs if s['dtype'] not in DTYPES or not s['shape']
                   or any(i <= 0 for i in s['shape'])]
        return dict(step=step, description=description, output_edge=target, op_type=op.type,
                    runtime_inputs=inputs, constant_inputs=list(constants), output=self.spec(target),
                    status='manual_edges' if manual else 'supply_input_abi' if missing else 'ready',
                    incomplete_inputs=missing)

    def node_row(self, step):
        descriptor(self.nodes, step)
        n = self.nodes[step.rsplit(':', 1)[0]]
        inputs, const = {}, []
        for i, edge in enumerate(n.input):
            if not edge or edge.endswith(':-1'):
                continue
            if self.constant(edge.rsplit(':', 1)[0]):
                const.append(edge)
            else:
                inputs[edge] = self.spec(edge, n, i)
        if not inputs or self.constant(n.name) or n.type in ('Data','NetOutput') or n.subgraph_name:
            raise ValueError(f'{step}: no standalone data-dependent operator at this edge')
        return self.row(step, step, list(inputs.values()), f'{n.type} 节点 {n.name} 的输出',
                        constants=const, manual=any(e.endswith(':-1') for e in n.input))

    def preset(self, layer, step):
        net, = [n for n in self.nodes.values() if n.type == 'NetOutput']
        target, = [e for e in net.input if e and not e.endswith(':-1')]
        if step == 'out_proj':
            x = _project_input(self.nodes, target)
            description = '门控后的激活 → DynamicQuant、原 W4 输出投影及输出视图'
        elif layer == 0 and step == 'qkv_conv_silu':
            x, target = activation_edges(self.graph)
            description = '卷积后、Q/K/V 拆分前的 SiLU'
        elif layer == 0 and step in ('gated_rmsnorm', 'gate_silu'):
            edges = stage_edges(self.graph, 'norm')
            if step == 'gated_rmsnorm':
                gdr, = [n for n in self.nodes.values() if n.type == 'ChunkGatedDeltaRule']
                x, target = gdr.name + ':0', edges['rms_output']
                description = 'GDR core → 带 gamma 的 RMSNorm 输出'
            else:
                x, target = edges['gate'], edges['silu_output']
                description = '门控输入 z → SiLU(z)'
        else:
            raise ValueError(f'L{layer}: unknown step {step!r}. Use catalog to select a step.')
        literals = set()
        visited = set()
        def visit(edge):
            if edge == x or edge in visited:
                return
            visited.add(edge)
            n = self.nodes[edge.rsplit(':', 1)[0]]
            if n.type in ('Const','Constant','FileConstant'):
                literals.add(edge)
            else:
                for arg in n.input:
                    if arg and not arg.endswith(':-1'):
                        visit(arg)
        visit(target)
        return self.row(step, target, [self.spec(x)], description, constants=sorted(literals))

    def select(self, layer, step):
        return self.node_row(step) if ':' in step else self.preset(layer, step)

    def rows(self, layer):
        for step in (('qkv_conv_silu','gated_rmsnorm','gate_silu','out_proj') if layer == 0 else ('out_proj',)):
            try:
                yield self.preset(layer, step)
            except (ValueError, KeyError, IndexError) as exc:
                yield dict(step=step, description=str(exc), status='unresolved_preset')
        for n in self.graph.op:
            if n.type in ('Data', 'NetOutput') or n.subgraph_name or self.constant(n.name):
                continue
            for i in range(len(n.output_desc)):
                try:
                    row = self.node_row(f'{n.name}:{i}')
                except ValueError:
                    continue
                yield row


def write_catalog(air, layers, dest):
    fields = ['layer','source_air_sha256','step','description','op_type','runtime_inputs','constant_inputs',
              'output_edge','output','status','incomplete_inputs']
    dest.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        staged = Path(tmp)/'catalog.csv'
        with staged.open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for layer in layers:
                _, original, _, graph = source_graph(air, layer)
                counts[layer] = 0
                for row in Catalog(graph).rows(layer):
                    fields_row = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict,list)) else v
                                  for k,v in row.items()}
                    writer.writerow(dict(fields_row, layer=layer, source_air_sha256=original['air']['sha256']))
                    counts[layer] += 1
        staged.replace(dest)
    print(f'{dest}  rows per layer: {counts}')
