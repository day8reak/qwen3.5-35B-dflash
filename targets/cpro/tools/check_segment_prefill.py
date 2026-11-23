#!/usr/bin/env python3
"""Capture the four-layer NZ prefix of the real 19-token segment and check it on CPU."""
import argparse
from contextlib import ExitStack
import json
import runpy
import shlex
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'model'))


def capture(root, entry, config, prior, attention_layers=False, trace_linear=None, gdr_stages=False):
    from qwen35_moe_dflash.common.command_trace import note
    from qwen35_moe_dflash.target.layer_probe.common import save_json
    from qwen35_moe_dflash.target.prefill_t64.segment_prefix import arm_prefix

    manifest = None
    snapshot = {}
    previous = sys.getprofile()
    with ExitStack() as cleanup:
        def entered(frame, event, _):
            nonlocal manifest
            if (event == 'call' and frame.f_code.co_filename == str(entry)
                    and frame.f_code.co_name == 'prefill_chunk_segment1'):
                sys.setprofile(previous)
                manifest = arm_prefix(frame, root, cleanup, attention_layers=attention_layers, trace_linear=trace_linear)
                from trace_segment_attention import observe_l1_gdr_length
                wrapper = frame.f_locals['model_wrapper']
                if gdr_stages:
                    from qwen35_moe_dflash.target.prefill_t64.gdr_stage_capture import observe_norm
                    for i in range(3):
                        snapshot[i] = {}
                        observe_norm(wrapper.model.model.layers[i].linear_attn, snapshot[i], cleanup)
                        observe_l1_gdr_length(wrapper, manifest, cleanup, snapshot[i], layer_index=i)
                elif trace_linear != 1:
                    observe_l1_gdr_length(wrapper, manifest, cleanup)
                if trace_linear is not None:
                    from qwen35_moe_dflash.target.prefill_t64.segment_gdn_trace import observe_tail
                    observe_tail(wrapper.model.model.layers[trace_linear].linear_attn, snapshot, cleanup)
                    observe_l1_gdr_length(wrapper, manifest, cleanup, snapshot, layer_index=trace_linear)

        cleanup.callback(sys.setprofile, previous)
        sys.setprofile(entered)
        sys.argv = [str(entry), '--max_token=0', '--quant_mode=enable', '--quant_bits=4',
                    '--prefill_chunk_size=64', '--config', str(config)]
        sys.path.insert(0, str(entry.parent))
        note(f'[310P driver in Python] cwd={Path.cwd()} argv={shlex.join(sys.argv)}')
        runpy.run_path(str(entry), run_name='__main__')
    if (manifest is None or len(manifest['boundaries']) != 4 or len(manifest['outputs']) != 9
            or len(manifest['conv_history']) != 3 or 'logits' not in manifest['outputs']):
        raise RuntimeError('original segmented prefill did not produce the complete four-layer prefix')
    gdr = manifest['l1_gdr']
    calls = gdr.get('calls', [])
    if (gdr.get('status') != 'OBSERVED' or len(calls) != 1 or calls[0] != dict(
            physical_q_rows=64, effective_length=[19], length_dtype='torch.int16', chunk_size=64)):
        raise ValueError(f"unexpected L1 GDR execution: {gdr}")
    if prior is not None:
        from qwen35_moe_dflash.target.layer_probe.common import read_array, sha
        import numpy as np
        previous_manifest = prior / 'manifest.json'
        known = json.loads(previous_manifest.read_text())
        if (known.get('scope') != 'L1_SEGMENT_GDR_PHYSICAL64_VALID19'
                or known.get('reference_qualification') != 'CPU_CHECKED' or known['status'] != 'CAPTURED'
                or not np.array_equal(read_array(root, manifest['inputs']['input_ids']),
                                      read_array(prior, known['wrapper_ids']))):
            raise ValueError('token IDs or reference status differ from the supplied qualified GDR capture')
        manifest['gdr_capture_sha256'] = sha(previous_manifest)
    if trace_linear is not None:
        from qwen35_moe_dflash.target.prefill_t64.segment_gdn_trace import save_trace
        trace = manifest['linear_trace'] = save_trace(root, manifest, trace_linear, snapshot)
        for name, metric in trace['cpu_audit'].items():
            print(f"L{trace_linear} 310P vs CPU {name}: rel_L2={metric['relative_l2']} passed={metric['passed']}", flush=True)
    if gdr_stages:
        from qwen35_moe_dflash.target.prefill_t64.gdr_stage_capture import capture_stages
        capture_stages(root, manifest, snapshot)
    manifest['status'] = 'CAPTURED'
    save_json(root / 'manifest.json', manifest)
    print(json.dumps(dict(status='CAPTURED', physical_rows=64, valid_rows=19, output_pos=18,
                          conv_history=manifest['conv_history'], manifest=str(root / 'manifest.json'))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, help='private 310P source containing inference_segment.py')
    p.add_argument('--config', type=Path, default=Path('config/qwen3_5_moe.yaml'))
    p.add_argument('--output', type=Path, required=True, help='fresh directory below AI_RUN_DIR')
    offline = p.add_mutually_exclusive_group()
    offline.add_argument('--diagnose-cpu', type=Path, metavar='CAPTURE', help='use saved four-layer inputs and cpu/report.json for fixed-input controls')
    offline.add_argument('--compare-cpu', type=Path, metavar='CAPTURE', help='reuse a completed capture for CPU comparison')
    p.add_argument('--gdr-capture', type=Path, help='match the exact token IDs to the previously qualified GDR case')
    p.add_argument('--decode-after', type=Path, metavar='PREFILL_FIXTURE',
                   help='capture the first T1 decode attention modules after this segmented physical64/valid19 case')
    p.add_argument('--capture-only', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--attention-layers', action='store_true',
                   help='capture standalone attention-module inputs/hidden and compare 310P with fixed-input CPU math')
    p.add_argument('--trace-linear', type=int, choices=range(3), metavar='LAYER',
                   help='observe one whole linear attention module and its chunk GDR boundaries on the same call')
    p.add_argument('--gdr-stages', action='store_true', help='capture native 310P L0-L2 chunk, RMSNorm and SiLU tensors for CPro replay')
    a = p.parse_args()
    if a.gdr_stages:
        if a.decode_after or a.compare_cpu or a.diagnose_cpu or a.trace_linear is not None:
            p.error('--gdr-stages selects the physical64/valid19 prefill capture')
        a.attention_layers = True
    if a.decode_after:
        if a.trace_linear is not None or a.gdr_capture or a.compare_cpu or a.diagnose_cpu:
            p.error('--decode-after selects first decode capture and fixed-input module comparison')
        a.attention_layers = True
    if a.trace_linear is not None:
        if a.compare_cpu or a.diagnose_cpu:
            p.error('--trace-linear is a capture option')
        a.attention_layers = True
    from qwen35_moe_dflash.target.layer_probe.common import new_output
    if a.compare_cpu or a.diagnose_cpu:
        if a.source_root or a.capture_only or a.gdr_capture:
            p.error('CPU replay takes its source from the capture')
        from qwen35_moe_dflash.target.prefill_t64.segment_prefix_cpu import compare, diagnose
        if a.attention_layers and a.compare_cpu:
            from qwen35_moe_dflash.target.prefill_t64.segment_modules import compare_cpu
            result = compare_cpu(a.compare_cpu.resolve(strict=True), new_output(a.output))
        elif a.diagnose_cpu:
            if a.attention_layers:
                p.error('--attention-layers uses --compare-cpu to check saved module outputs')
            result = diagnose(a.diagnose_cpu.resolve(strict=True), new_output(a.output))
        else:
            result = compare(a.compare_cpu.resolve(strict=True), new_output(a.output))
        print(json.dumps(result), flush=True)
        return 0 if result['status'] in ('PASSED', 'INSPECTED') else 1
    if not a.source_root:
        p.error('--source-root is required for capture')
    entry = (a.source_root / 'inference_segment.py').resolve(strict=True)
    config = (entry.parent / a.config).resolve(strict=not a.decode_after)
    prior = a.gdr_capture.resolve(strict=True) if a.gdr_capture else None
    if a.capture_only:
        if a.decode_after:
            from qwen35_moe_dflash.target.prefill_t64.segment_decode import capture_decode
            root = new_output(a.output)
            try:
                capture_decode(root, entry, a.decode_after.resolve(strict=True))
            except Exception as exc:
                from qwen35_moe_dflash.target.layer_probe.common import save_json
                path = root / 'manifest.json'
                report = json.loads(path.read_text()) if path.is_file() else {'phase': 'decode'}
                save_json(path, dict(report, status='FAILED', error=f'{type(exc).__name__}: {exc}'))
                raise
        else:
            capture(new_output(a.output), entry, config, prior, a.attention_layers, a.trace_linear, a.gdr_stages)
        return 0
    # Release both segmented NPU models before loading the independent CPU prefix.
    out = a.output.resolve()
    script = str(Path(__file__).resolve())
    command = [sys.executable, script, '--source-root', str(entry.parent), '--config', str(config),
               '--output', str(out), '--capture-only']
    if prior:
        command.extend(['--gdr-capture', str(prior)])
    if a.decode_after:
        command.extend(['--decode-after', str(a.decode_after.resolve(strict=True))])
    if a.attention_layers:
        command.append('--attention-layers')
    if a.trace_linear is not None:
        command.extend(['--trace-linear', str(a.trace_linear)])
    if a.gdr_stages:
        command.append('--gdr-stages')
    subprocess.run(command, cwd=entry.parent, check=True)
    if a.gdr_stages:
        from qwen35_moe_dflash.common.artifacts import share_for_transfer
        share_for_transfer(out)
        return 0
    cpu_command = [sys.executable, script, '--compare-cpu', str(out)]
    if a.attention_layers:
        cpu_command.append('--attention-layers')
    return subprocess.run([*cpu_command, '--output', str(out / ('module-cpu' if a.attention_layers else 'cpu'))]).returncode


if __name__ == '__main__':
    raise SystemExit(main())
