"""Observe the original segmented 310P embedding path, stopping at the first decoder layer."""
from contextlib import ExitStack
import json
import linecache
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from qwen35_moe_dflash.target.layer_probe.capture import bound, cpu
from qwen35_moe_dflash.target.layer_probe.common import read_tensor, save_json
from qwen35_moe_dflash.target.prefill_t64.chain_reference import driver_prompt, run_segment_prefill


def array(t):
    value = cpu(t)
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.numpy()


class EmbeddingTrace(TorchDispatchMode):
    def __init__(self, source, ids, valid, width):
        super().__init__()
        self.source = source
        self.ids = ids.ravel()
        self.valid = valid
        self.width = width
        self.events = []

    def record(self, op, kind, value):
        # Only materialize the selected token rows. The vocabulary table stays on its device.
        frame = sys._getframe(1)
        while frame and not Path(frame.f_code.co_filename).is_relative_to(self.source):
            frame = frame.f_back
        location = None
        if frame:
            filename, line = frame.f_code.co_filename, frame.f_lineno
            location = f'{filename}:{line} {linecache.getline(filename, line).strip()}'
        width = 1 if kind in ('scale', 'mul_scale', 'wrapper_scale') else self.width
        self.events.append((op, kind, location, str(value.dtype), array(value).reshape(-1, width)))

    def lookup(self, op, table, indices, out):
        if not isinstance(indices, torch.Tensor) or indices.dtype not in (torch.int32, torch.int64):
            return
        n = indices.numel()
        if n not in (self.valid, self.ids.size) or not isinstance(out, torch.Tensor):
            return
        # A table indexed along the vocabulary axis is either [vocab, hidden] or scalar per token.
        if table.ndim not in (1, 2) or table.shape[0] <= int(self.ids[:n].max()):
            return
        width = table.shape[1] if table.ndim == 2 else 1
        if width not in (1, self.width) or out.numel() != n * width:
            return
        if not np.array_equal(array(indices).ravel(), self.ids[:n]):
            return
        kind = 'scale' if width == 1 else 'codes' if out.dtype == torch.int8 else 'lookup_rows'
        self.record(f'{op} table={table.dtype}{list(table.shape)}', kind, out)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        op = str(func)
        if op == 'aten.embedding.default':
            self.lookup(op, args[0], args[1], out)
        elif op == 'aten.index_select.default' and args[1] == 0:
            self.lookup(op, args[0], args[2], out)
        elif op in ('aten.index.Tensor', 'aten._unsafe_index.Tensor') and len(args[1]) == 1:
            self.lookup(op, args[0], args[1][0], out)
        elif op == 'aten.mul.Tensor' and isinstance(out, torch.Tensor) and out.ndim in (2, 3) and out.shape[-1] == self.width:
            n = out.numel() // self.width
            if n in (self.valid, self.ids.size):
                for x, scale in ((args[0], args[1]), (args[1], args[0])):
                    if (isinstance(x, torch.Tensor) and isinstance(scale, torch.Tensor)
                            and x.shape[-1:] == (self.width,) and x.numel() == out.numel()
                            and scale.numel() in (1, n)):
                        self.record(op, 'mul_rows', x)
                        self.record(op, 'mul_scale', scale)
                        self.record(op, 'mul_result', out)
                        break
        return out


def capture(source, config, ids, hidden, valid):
    trace = EmbeddingTrace(source, ids, valid, hidden.shape[-1])
    wrapper_inputs = {}

    class AtLayerZero(Exception):
        pass

    with ExitStack() as stack:
        def arm(frame):
            driver = frame
            while driver and 'model2' not in driver.f_locals:
                driver = driver.f_back
            if driver is None:
                raise ValueError('Could not locate the original segmented inference driver')
            _, prompt = driver_prompt(driver.f_locals, frame)
            if not np.array_equal(prompt.numpy().ravel(), ids.ravel()[:valid]):
                raise ValueError('Live YAML prompt differs from the saved prefill input IDs')
            wrapper = frame.f_locals['model_wrapper']

            def enter(mod, args, kw):
                values = bound(mod, args, kw)
                for name in ('input_ids', 'inputs_embeds', 'embed_scale'):
                    val = values.get(name)
                    wrapper_inputs[name] = None if val is None else f'{val.dtype}{list(val.shape)}'
                    if val is None:
                        continue
                    if name == 'input_ids':
                        if not np.array_equal(array(val).ravel(), ids.ravel()):
                            raise ValueError('Live wrapper IDs differ from the retained T64 IDs')
                    else:
                        trace.record('wrapper.' + name, 'wrapper_scale' if name == 'embed_scale' else 'wrapper_rows', val)

            def stop(mod, args, kw):
                actual = bound(mod, args, kw)['hidden_states']
                if actual.device.type != 'npu' or actual.dtype != torch.float16 or not np.array_equal(array(actual), hidden):
                    raise ValueError('Live 310P L0 hidden differs from the saved four-layer fixture')
                raise AtLayerZero()

            stack.callback(wrapper.register_forward_pre_hook(enter, with_kwargs=True).remove)
            # Also observe module outputs when a vendor embedding kernel hides the underlying ATen lookup.
            for name, module in wrapper.named_modules():
                if isinstance(module, torch.nn.Embedding):
                    def embedded(mod, args, kw, out, name=name):
                        index = args[0] if args else kw.get('input')
                        trace.lookup(name, mod.weight, index, out)
                    stack.callback(module.register_forward_hook(embedded, with_kwargs=True).remove)
            stack.callback(wrapper.model.model.layers[0].register_forward_pre_hook(stop, with_kwargs=True).remove)
            stack.enter_context(trace)

        try:
            run_segment_prefill(source/'inference_segment.py', config, arm)
        except AtLayerZero:
            return wrapper_inputs, trace.events
        raise ValueError('Original prefill did not enter L0')


def row_metrics(actual, reference):
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return dict(exact=False, issue='nonfinite')
    delta = a - b
    norm = np.linalg.norm(b)
    return dict(exact=bool(np.array_equal(a, b)), max_abs=float(np.abs(delta).max()),
                relative_l2=float(np.linalg.norm(delta) / norm) if norm else (0. if not delta.any() else None))


def diagnose(root, source, config):
    fixture = root/'export/fixture'
    ref = json.loads((fixture/'reference.json').read_bytes())

    def tensor(role):
        spec = ref['inputs'][role]
        return read_tensor(fixture/spec['path'], spec)

    ids, hidden, valid = tensor('input_ids'), tensor('hidden'), ref['valid_rows']
    width = hidden.shape[-1]
    case = Path(os.getenv('DECODER_EMBEDDING_CASE') or root/'export/package/L00-L03')
    meta = json.loads((case/'case.json').read_bytes())
    # Reuse the recorded fixture identities. Only read the two small, already selected CPro inputs.
    if meta['fixture']['reference_inputs']['embedding'] != ref['inputs']['hidden']['sha256']:
        raise ValueError(f'{case} was prepared from a different 310P entry')

    def feed(role):
        spec = meta['input_files'][role]
        return read_tensor(case/spec['path'], spec).reshape(ids.size, -1)[:valid]

    codes, scale = feed('inputs_embeds'), feed('embed_scale')
    if codes.shape != (valid, width) or codes.dtype != np.int8 or scale.shape != (valid, 1) or scale.dtype != np.float32:
        raise ValueError('Expected CPro INT8 embedding rows and FP32 per-token scales')
    if not np.isfinite(scale).all():
        raise ValueError('Nonfinite CPro embedding scale on a valid token')
    reconstructed = codes.astype(np.float32) * scale
    golden = hidden.reshape(ids.size, width)[:valid]
    wrapper, events = capture(source, config, ids, hidden, valid)
    baseline = row_metrics(reconstructed.astype(golden.dtype), golden)
    print(f'CPro 输入主机重建 vs 实测 310P 入口 {baseline}', flush=True)
    print(f'310P wrapper 实参 {wrapper}', flush=True)
    rows = [dict(token=t, id=int(idx), cpro_scale=float(scale[t, 0]),
                 reconstructed_vs_entry=row_metrics(reconstructed[t].astype(golden.dtype), golden[t]))
            for t, idx in enumerate(ids.ravel()[:valid])]
    summary = []
    matched = []
    for i, (op, kind, location, dtype, values) in enumerate(events):
        values = values[:valid]
        print(f'[{i}] {kind} {op} {dtype}  {location or ""}')
        if kind in ('scale', 'mul_scale', 'wrapper_scale'):
            if values.shape[0] == 1:
                values = np.broadcast_to(values, scale.shape)
            targets = {'cpro_scale': scale}
        elif kind in ('codes', 'mul_rows') or (kind == 'wrapper_rows' and values.dtype == np.int8):
            targets = {'cpro_codes': codes}
        else:
            targets = {'310p_block_entry': golden, 'cpro_reconstructed': reconstructed}
        if (kind in ('lookup_rows', 'wrapper_rows', 'mul_result') and values.dtype.kind == 'f'
                and np.array_equal(values.astype(golden.dtype), golden)):
            detail = kind
            if kind == 'mul_result':
                x = events[i-2][4][:valid]
                for _, earlier, _, _, sample in events[:i-2]:
                    if earlier in ('codes', 'wrapper_rows') and sample.dtype == np.int8 and np.array_equal(sample[:valid], x):
                        detail = 'observed_int8_rows_times_scale'
                        break
            matched.append(dict(event=i, path=detail))
        scores = {}
        for target, other in targets.items():
            if values.shape != other.shape:
                raise ValueError(f'{op} {kind} shape {values.shape} differs from {target} {other.shape}')
            scores[target] = row_metrics(values, other)
            if target == 'cpro_codes':
                scores[target]['equal_fraction'] = float(np.mean(values == other))
            print(f'    vs {target}: {scores[target]}')
            for t, row in enumerate(rows):
                item = row_metrics(values[t], other[t])
                if target == 'cpro_codes':
                    item['different_codes'] = int(np.count_nonzero(values[t] != other[t]))
                if target == 'cpro_scale':
                    item['310p_value'] = float(values[t, 0]) if np.isfinite(values[t, 0]) else None
                row.setdefault(f'{i}_{kind}', {})[target] = item
        summary.append(dict(op=op, kind=kind, source=location, dtype=dtype, comparisons=scores))
    output = root/'l0-embedding/result.json'
    output.parent.mkdir(exist_ok=True)
    save_json(output, dict(status='ENTRY_MATCHED' if matched else 'ENTRY_PATH_UNRESOLVED',
                           matches_live_entry=matched, wrapper_inputs=wrapper, valid_rows=valid,
                           cpro_case=str(case), reconstructed_vs_entry=baseline, events=summary, tokens=rows))
    print(f'转为实测入口 dtype 后逐元素相同的观测点 {matched}', flush=True)
    print(f'逐 token 对照 {output}', flush=True)
    if not matched:
        print('已复现 310P 入口，但观察到的查表或乘法结果不能直接还原它，详见各观测点。')
    return bool(matched)
