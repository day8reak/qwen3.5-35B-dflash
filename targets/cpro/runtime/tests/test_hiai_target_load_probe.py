"""One explicit target-only init-before-weights experiment; host fake SDK only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

runner = Path(sys.argv[1]).resolve()
base = Path(sys.argv[2]); base.mkdir(parents=True, exist_ok=True)
root = Path(tempfile.mkdtemp(prefix='target-load-', dir=base)).resolve()
model = root / 'model "quoted".omc'
model.write_bytes(b'original-target-load-probe-fixture')
weights = root / 'weight "directory"'; weights.mkdir()


def command(path=model, order='init-before-weights', cleanup='weights-before-manager', version='FAKE-HIAI-X90'):
    return [str(runner), '--target-load-probe', str(path), '--model-sha256',
            hashlib.sha256(path.read_bytes()).hexdigest(), '--weight-directory', str(weights),
            '--load-order', order, '--cleanup-order', cleanup, '--expected-runtime-version', version,
            '--ddk-id', 'fake "ddk"', '--firmware-id', 'fake\\firmware']


def execute(label, fault='', args=None, output=None):
    out = output or root / (label + '.json'); trace = root / (label + '.trace')
    process = subprocess.run((args or command()) + ['--output', str(out)],
        cwd=root, capture_output=True, text=True, timeout=20,
        env=dict(os.environ, FAKE_HIAI_FAILURE=fault, FAKE_HIAI_TRACE=str(trace)))
    events = trace.read_text().splitlines() if trace.exists() else []
    assert 'LEAK' not in process.stderr, process.stderr
    assert not any(event in events for event in ('process', 'load', 'unload', 'v2-run', 'v2-create-tensor'))
    for event in ('v2-init', 'v2-init-weights', 'v2-deinit', 'v2-deinit-weights'):
        assert events.count(event) <= 1, (event, events)  # No fallback or cleanup retry.
    return process, out, events


def report(process, output):
    value = json.loads(output.read_text())
    assert json.loads(process.stdout) == value
    assert value['artifact_kind'] == 'qwen35-hiai-target-load-probe'
    assert value['evidence_kind'] == 'host-fake-hiai'
    assert value['model_name'] == 'qwen35_resident_target'
    assert value['load_order'] == 'init-before-weights' and value['model_load_api'] == 'file-v2'
    assert value['cleanup_order'] == 'weights-before-manager'
    assert value['run_called'] is False and value['process_called'] is False
    assert value['tensor_allocations_requested'] == 0 and value['numerical_validation'] == 'NOT_RUN'
    assert value['lifecycle']['run_attempts'] == value['lifecycle']['successful_runs'] == 0
    assert value['lifecycle']['closed'] is True
    assert value['lifecycle']['weight_init_policy'] == 'EXPERIMENTAL_INIT_BEFORE_WEIGHTS_LOAD_PROBE_ONLY_NO_FALLBACK'
    assert value['lifecycle']['weight_cleanup_policy'] == 'DEINIT_WEIGHTS_BEFORE_DEINIT'
    assert value['lifecycle']['ordering_validation'] == 'NOT_VALIDATED_ON_DEVICE'
    assert value['model_content_unchanged'] is True and value['elapsed_ms'] >= 0
    assert value['weight_directory'] == str(weights) and value['process_id'] > 0
    return value


# Both the production fixture and dedicated probe fixture require Init first.
for fault in ('', 'v2-run', 'v2-allocation', 'v2-shape'):
    process, output, events = execute('success-' + (fault or 'normal'), fault)
    assert process.returncode == 0, process.stderr
    value = report(process, output); life = value['lifecycle']
    assert value['status'] == 'TARGET_LOAD_PROBE_PASSED' and value['runtime_stage'] == 'CLOSED'
    assert value['error'] == life['failure_stage'] == life['cleanup_error'] == ''
    assert life['manager_init_status'] == life['weights_init_status'] == life['weights_deinit_status'] == 0
    assert all(life[key] for key in ('manager_init_attempted', 'manager_initialized', 'weights_init_attempted',
                                   'weights_initialized', 'manager_deinit_called', 'weights_deinit_called'))
    sequence = ['v2-restore-file', 'v2-set-name:qwen35_resident_target', 'v2-create-manager',
                'v2-init', 'v2-init-weights', 'v2-deinit-weights', 'v2-deinit', 'v2-manager-destroy', 'v2-built-model-destroy']
    assert [events.index(event) for event in sequence] == sorted(events.index(event) for event in sequence)


for fault in ('original-init', 'original-init-deinit-throw'):
    process, output, events = execute(fault, fault)
    assert process.returncode == 1
    value = report(process, output); life = value['lifecycle']
    assert value['runtime_stage'] == life['failure_stage'] == 'EXPERIMENTAL_MODEL_MANAGER_INIT_BEFORE_WEIGHTS'
    assert life['manager_init_attempted'] and not life['manager_initialized'] and life['manager_init_status'] == 32
    assert not life['weights_init_attempted'] and life['weights_init_status'] is None
    assert life['manager_deinit_called'] and not life['weights_deinit_called'] and life['weights_deinit_status'] == -1
    assert ('v2-init-weights' not in events) and ('v2-deinit-weights' not in events)
    assert bool(life['cleanup_error']) == (fault == 'original-init-deinit-throw')


for fault in ('original-init-weights', 'original-init-weights-throw', 'original-init-weights-unknown'):
    process, output, events = execute(fault, fault)
    assert process.returncode == 1
    value = report(process, output); life = value['lifecycle']
    assert value['runtime_stage'] == life['failure_stage'] == 'EXPERIMENTAL_INIT_WEIGHTS_AFTER_INIT'
    assert life['manager_initialized'] and life['manager_init_status'] == 0 and life['manager_deinit_called']
    assert life['weights_init_attempted'] and not life['weights_initialized'] and not life['weights_deinit_called']
    assert life['weights_init_status'] == (72 if fault == 'original-init-weights' else None)
    assert life['weights_deinit_status'] == -1 and life['cleanup_error'] == ''
    assert 'v2-deinit-weights' not in events


for fault in ('original-deinit-weights', 'original-deinit-weights-throw', 'original-deinit-weights-unknown', 'original-both-cleanup-failures'):
    process, output, events = execute(fault, fault)
    assert process.returncode == 1
    value = report(process, output); life = value['lifecycle']
    assert value['runtime_stage'] == life['failure_stage'] == 'CLEANUP_FAILED'
    assert life['manager_initialized'] and life['weights_initialized']
    assert life['manager_init_status'] == life['weights_init_status'] == 0
    returned = fault in ('original-deinit-weights', 'original-both-cleanup-failures')
    assert life['manager_deinit_called'] and life['weights_deinit_called'] and life['weights_deinit_status'] == (73 if returned else -1)
    assert life['cleanup_error'] and life['cleanup_error'] in value['error']
    assert events.index('v2-deinit-weights') < events.index('v2-deinit')
    if fault == 'original-both-cleanup-failures':
        assert life['cleanup_error'].startswith('DeinitWeights failed: 73; injected target DeInit exception')


# An unsuccessful Init must stop here without trying weights first or retrying.
old_model = root / 'unchanged-original-fixture.omc'; old_model.write_bytes(b'original-target-fixture')
process, output, events = execute('no-fallback', 'original-init', args=command(old_model))
assert process.returncode == 1
value = report(process, output)
assert value['lifecycle']['manager_init_status'] == 32 and 'v2-init-weights' not in events


process, output, events = execute('wrong-runtime', args=command(version='OTHER'))
assert process.returncode == 1
value = report(process, output)
assert value['runtime_stage'] == 'RUNTIME_IDENTITY'
assert not value['lifecycle']['manager_init_attempted'] and not value['lifecycle']['weights_init_attempted']
assert 'v2-create-built-model' not in events

for label, args in [('wrong-order', command(order='weights-before-init')),
                    ('wrong-cleanup-order', command(cleanup='manager-before-weights')),
                    ('missing-cleanup-order', [arg for arg in command() if arg not in ('--cleanup-order', 'weights-before-manager')]),
                    ('extra-mode', command() + ['--model-io-only', str(model)]),
                    ('wrong-sha', [arg if arg != hashlib.sha256(model.read_bytes()).hexdigest() else '0' * 64 for arg in command()])]:
    process, output, events = execute(label, args=args)
    assert process.returncode != 0 and not output.exists() and not events
old_output = root / 'retained.json'; old_output.write_text('do not replace')
process, output, events = execute('existing-output', output=old_output)
assert process.returncode != 0 and output.read_text() == 'do not replace' and not events
print('PASS target-only load probe: explicit order, zero inference/allocation, exact partial cleanup, no fallback; host fake only')
