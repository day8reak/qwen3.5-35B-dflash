"""Keep portable Draft AIR metadata separate from its original tensor bytes.

TorchAir normally switches Const/FileConstant at (2048 - 200) MiB. A smaller
Draft still needs named FileConstants for compiler snapshots and weight binding.
Override only that decision, in this CLI process, and let the installed exporter
create the graph and write tensors. Never patch the SDK installation or infer
lost parameter names from an already inlined AIR.
"""
from contextlib import contextmanager
import hashlib
import importlib
import inspect
from pathlib import Path
import re
import threading

_LOCK = threading.Lock()
_UPSTREAM = ('https://github.com/Ascend/torchair/blob/'
    'e1118e8b0fc1b95190ab53b732c0872989550e20/python/torchair/_utils/export_utils.py')
_PARAMETERS = ('inputs', 'weight_name', 'export_graph')


@contextmanager
def external_weight_export():
    """Temporarily select TorchAir's native external-weight branch; fail closed."""
    module = importlib.import_module('torchair._utils.export_utils')
    original = getattr(module, '_is_weight_externalized', None)
    if not callable(original) or tuple(inspect.signature(original).parameters) != _PARAMETERS:
        raise ValueError('installed TorchAir lacks the supported external-weight decision hook; '
                         'inspect torchair._utils.export_utils before exporting')
    converter = getattr(module, '_convert_data_to_const', None)
    if (not inspect.isfunction(converter)
            or converter.__globals__.get('_is_weight_externalized') is not original
            or '_is_weight_externalized' not in converter.__code__.co_names):
        raise ValueError('installed TorchAir converter does not use the supported external-weight hook')
    source = Path(inspect.getsourcefile(original))
    evidence = dict(policy='native-torchair-fileconstant-v1', upstream=_UPSTREAM,
                    installed_source=str(source),
                    installed_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    scope='PROCESS_LOCAL_STORAGE_DECISION_ONLY; ORIGINAL_TENSOR_BYTES',
                    calls=0, restored=False, files=[])
    thread_id = threading.get_ident()

    def force_external(inputs, weight_name, export_graph):
        if threading.get_ident() != thread_id or evidence['calls']:
            raise ValueError('Draft export expects one synchronous weight conversion; no segmented/concurrent export')
        observed = original(inputs, weight_name, export_graph)
        selected = [(i, value, weight_name[id(value)]) for i, value in enumerate(inputs)
                    if id(value) in weight_name]
        if (not isinstance(observed, tuple) or len(observed) != 2
                or type(observed[0]) is not bool or type(observed[1]) is not int
                or observed[1] != len(selected) or not selected):
            raise ValueError('installed TorchAir returned an unsupported weight-externalization decision')
        files = []
        for i, value, name in selected:
            if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9.]*', name):
                raise ValueError('unsupported TorchAir parameter name for portable weight files')
            if i >= len(export_graph.op) or export_graph.op[i].type != 'Data':
                raise ValueError('TorchAir weight conversion input is not its original Data node')
            size = value.numel() * value.element_size()
            if not value.is_contiguous() or size <= 0:
                raise ValueError('Draft external weights must be nonempty contiguous tensors')
            files.append(dict(path='weight/' + name.replace('.', '_'),
                node=export_graph.op[i].name, dtype=str(value.dtype),
                shape=list(value.shape), bytes=size))
        if len({r['path'] for r in files}) != len(files):
            raise ValueError('TorchAir parameter names collide after filename normalization')
        evidence.update(calls=1, native_externalized=observed[0], files=files,
                        weight_files=len(files), weight_bytes=sum(r['bytes'] for r in files))
        return True, observed[1]

    if not _LOCK.acquire(blocking=False):
        raise ValueError('another Draft external-weight export is active in this process')
    try:
        module._is_weight_externalized = force_external
        try:
            yield evidence
            if evidence['calls'] != 1:
                raise ValueError('TorchAir did not use the selected external-weight path; AIR not accepted')
        finally:
            module._is_weight_externalized = original
            evidence['restored'] = True
    finally:
        _LOCK.release()


def validate_external_weights(inspection, evidence):
    """Require the native writer to preserve the names and extents it selected."""
    expected = {r['path']: r['bytes'] for r in evidence['files']}
    if (evidence.get('calls') != 1 or not evidence.get('restored') or not expected
            or inspection['weight_files'] != expected):
        raise ValueError('AIR FileConstants differ from the native external-weight selection; retained for diagnosis')
