#!/usr/bin/env python3
"""Capture the four-layer NZ prefix of the real 19-token segment and check it on CPU."""
import argparse
from contextlib import ExitStack
import json
import runpy
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'model'))


def capture(root, entry, config, prior, attention_layers=False):
    from qwen35_moe_dflash.target.layer_probe.common import save_json
    from qwen35_moe_dflash.target.prefill_t64.segment_prefix import arm_prefix

    manifest = None
    previous = sys.getprofile()
    with ExitStack() as cleanup:
        def entered(frame, event, _):
            nonlocal manifest
            if (event == 'call' and frame.f_code.co_filename == str(entry)
                    and frame.f_code.co_name == 'prefill_chunk_segment1'):
                sys.setprofile(previous)
                manifest = arm_prefix(frame, root, cleanup, attention_layers=attention_layers)
                from trace_segment_attention import observe_l1_gdr_length
                observe_l1_gdr_length(frame.f_locals['model_wrapper'], manifest, cleanup)

        cleanup.callback(sys.setprofile, previous)
        sys.setprofile(entered)
        sys.argv = [str(entry), '--max_token=0', '--quant_mode=enable', '--quant_bits=4',
                    '--prefill_chunk_size=64', '--config', str(config)]
        sys.path.insert(0, str(entry.parent))
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
    p.add_argument('--capture-only', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--attention-layers', action='store_true',
                   help='capture standalone attention-module inputs/hidden and compare 310P with fixed-input CPU math')
    a = p.parse_args()
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
    config = (entry.parent / a.config).resolve(strict=True)
    prior = a.gdr_capture.resolve(strict=True) if a.gdr_capture else None
    if a.capture_only:
        capture(new_output(a.output), entry, config, prior, a.attention_layers)
        return 0
    # Release both segmented NPU models before loading the independent CPU prefix.
    out = a.output.resolve()
    script = str(Path(__file__).resolve())
    command = [sys.executable, script, '--source-root', str(entry.parent), '--config', str(config),
               '--output', str(out), '--capture-only']
    if prior:
        command.extend(['--gdr-capture', str(prior)])
    if a.attention_layers:
        command.append('--attention-layers')
    subprocess.run(command, cwd=entry.parent, check=True)
    cpu_command = [sys.executable, script, '--compare-cpu', str(out)]
    if a.attention_layers:
        cpu_command.append('--attention-layers')
    return subprocess.run([*cpu_command, '--output', str(out / ('module-cpu' if a.attention_layers else 'cpu'))]).returncode


if __name__ == '__main__':
    raise SystemExit(main())
