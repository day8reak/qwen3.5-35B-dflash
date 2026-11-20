"""Compare the two qualified L0 stage dumps against their common measured 310P reference."""
import csv
import json

from ...common.artifacts import share_for_transfer
from ..layer_probe.common import checked


def compare_dumps(chunk, single):
    def scores(dump):
        manifest = json.loads((dump / 'manifest.json').read_text())
        if (manifest['physical_rows'], manifest['valid_rows'], manifest['last_valid_index'],
                manifest['reference_backend'], set(manifest['layers'])) != (64, 19, 18, '310p', {'0'}):
            raise ValueError(f'{dump}: expected the L0 physical64/valid19 dump')
        layer = manifest['layers']['0']
        if not all(g['module_inputs_equal_baseline'] for g in layer['graphs'].values()):
            raise ValueError(f'{dump}: stage inputs differ from the uninstrumented graph')
        reference = {}
        for kind in ('inputs', 'tensors'):
            for name, values in layer[kind].items():
                spec = values['310p']
                checked(dump, spec)
                reference[kind, name] = spec['dtype'], spec['shape'], spec['sha256']
        with (dump / 'summary.csv').open() as f:
            rows = {(r['stage'], r['scope']): r for r in csv.DictReader(f) if r['comparison'] == 'cpro_vs_310p'}
        return manifest, reference, rows

    a, ar, old = scores(chunk)
    b, br, new = scores(single)
    if (a['chunk_size'] != 64 or b['chunk_size'] != 1 or b['cpro_gdr_calls'] != 19
            or b['layers']['0']['chunk_baseline_air_sha256'] != a['layers']['0']['graphs']['module_hidden']['source_air_sha256']
            or a['capture_sha256'] != b['capture_sha256'] or ar != br or old.keys() != new.keys()):
        raise ValueError('Chunk and S=1 dumps differ in baseline AIR, 310P reference, or stage coverage')
    fields = ('relative_l2', 'cosine', 'result', 'reference_fp16_subnormals_with_actual_zero')
    target = single / 'chunk-vs-s1.csv'
    temp = target.with_suffix('.csv.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['stage', 'scope', *(mode+'_'+f for mode in ('chunk64','s1') for f in fields)])
        print('L0                       scope      chunk64 L2/cos          S1 L2/cos')
        for (stage, scope), first in old.items():
            second = new[stage, scope]
            for row in (first, second):
                if row['measurement'] == 'DIAGNOSTIC':
                    row.update(relative_l2='', cosine='', result='DIAGNOSTIC', reference_fp16_subnormals_with_actual_zero='')
            writer.writerow([stage, scope, *(r[f] for r in (first, second) for f in fields)])
            if stage in ('core', 'final_state', 'rms_output', 'silu_output', 'mul_output', 'golden_hidden'):
                values = [f"{float(r['relative_l2']):.6g}/{float(r['cosine']):.6g}"
                          if r['relative_l2'] and r['cosine'] else 'n/a' for r in (first, second)]
                print(f'{stage:24} {scope:9}  {values[0]:22}  {values[1]}')
    temp.replace(target)
    share_for_transfer(target)
    print(target)
