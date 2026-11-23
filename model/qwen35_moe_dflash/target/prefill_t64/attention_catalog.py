"""Build a selectable per-port CSV from the installed L0 and L3 attention AIRs."""
import csv
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import tempfile

from .attention_slice import DTYPES, descriptor, port_info, source_graph
from .attention_steps import AttentionSteps


@lru_cache(None)
def preset_table(layer):
    path = Path(__file__).resolve().parents[4] / 'specs/attention-step-presets.csv'
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return {row['step']: row['步骤说明'] for row in csv.DictReader(stream) if row['layer'] == str(layer)}


class Catalog:
    def __init__(self, graph, bindings=()):
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
        self.steps = AttentionSteps(graph, self.nodes, bindings)

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
                    incomplete_inputs=missing, reason='Original control dependencies require review' if manual else '')

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
        table = preset_table(layer)
        if step not in table:
            raise ValueError(f'L{layer}: unknown step {step!r}. Use catalog to select a step.')
        description = table[step]
        cuts, target = self.steps.boundaries(layer, step)
        if cuts is None:
            row = self.node_row(target)
            return dict(row, step=step, description=description)
        if target in cuts:
            raise ValueError(f'Input and output share {target}; no independent operation in this AIR')
        literals, reached, visited = set(), set(), set()
        manual = False

        def visit(edge):
            nonlocal manual
            if edge in cuts:
                reached.add(edge)
                return
            if edge in visited:
                return
            visited.add(edge)
            op = self.nodes[edge.rsplit(':', 1)[0]]
            if op.type in ('Const', 'Constant', 'FileConstant'):
                literals.add(edge)
                return
            if op.type in ('Data', 'NetOutput') or op.subgraph_name:
                raise ValueError(f'Unexpected dependency {edge} ({op.type}) inside {step}')
            for arg in op.input:
                if arg.endswith(':-1'):
                    manual = True
                elif arg:
                    visit(arg)
        visit(target)
        if reached != set(cuts):
            raise ValueError(f'{step} does not consume {sorted(set(cuts) - reached)}')
        return self.row(step, target, [self.spec(x) for x in cuts], description,
                        constants=sorted(literals), manual=manual)

    def select(self, layer, step):
        return self.node_row(step) if ':' in step else self.preset(layer, step)

    def rows(self, layer, presets_only=False):
        for step, description in preset_table(layer).items():
            try:
                yield dict(self.preset(layer, step), kind='preset')
            except (ValueError, KeyError, IndexError) as exc:
                yield dict(step=step, kind='preset', description=description, reason=str(exc), status='unresolved_preset')
        if presets_only:
            return
        for n in self.graph.op:
            if n.type in ('Data', 'NetOutput') or n.subgraph_name or self.constant(n.name):
                continue
            for i in range(len(n.output_desc)):
                try:
                    row = self.node_row(f'{n.name}:{i}')
                except ValueError:
                    continue
                yield dict(row, kind='ge_port')


def write_catalog(air, layers, dest, *, presets_only=False):
    fields = ['layer','source_air_sha256','kind','step','description','op_type','runtime_inputs','constant_inputs',
              'output_edge','output','status','reason','incomplete_inputs']
    dest.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        staged = Path(tmp)/'catalog.csv'
        with staged.open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for layer in layers:
                _, original, _, graph = source_graph(air, layer)
                counts[layer] = Counter()
                for row in Catalog(graph, original.get('source_inputs', ())).rows(layer, presets_only):
                    fields_row = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict,list)) else v
                                  for k,v in row.items()}
                    writer.writerow(dict(fields_row, layer=layer, source_air_sha256=original['air']['sha256']))
                    counts[layer][row['status']] += 1
        staged.replace(dest)
    print(dest)
    for layer, counts_by_status in counts.items():
        print(f'L{layer}  {dict(counts_by_status)}')
