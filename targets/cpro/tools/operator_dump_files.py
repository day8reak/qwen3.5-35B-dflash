"""Index the observed data_proc filenames; calibrate byte decoding against replay tensors."""
from collections import defaultdict
from dataclasses import dataclass
from math import prod
from pathlib import Path
import re

import numpy as np

from qwen35_moe_dflash.target.layer_probe.common import sized_file


TYPES = {'float': '<f4', 'float32': '<f4', 'fp32': '<f4', 'float16': '<f2',
         'fp16': '<f2', 'half': '<f2', 'int8': 'i1', 'uint8': 'u1',
         'int16': '<i2', 'int32': '<i4', 'int64': '<i8'}
FILENAME = re.compile(r'^_\d+_\d+_(.+)_(' + '|'.join(TYPES) +
                      r')_(\d+(?:_\d+)*)_(\d+)_(\d+)\.dump$', re.I)


@dataclass(frozen=True)
class Dump:
    path: str
    shape: tuple
    dtype: str
    size: int

    @property
    def extra(self):
        return self.size - prod(self.shape) * np.dtype(self.dtype).itemsize

    def views(self, root, air_dtype):
        if self.extra not in (0, 32):
            return
        raw = sized_file(root, dict(path=self.path, bytes=self.size)).read_bytes()
        dtypes = [self.dtype]
        if self.dtype == '<f4' and air_dtype == 'DT_FLOAT16':
            dtypes.append('<f2')
        for dtype in dtypes:
            for offset in ([0, 32] if self.extra else [0]):
                value = np.frombuffer(raw, dtype, count=prod(self.shape), offset=offset)
                yield (dtype, offset), value.reshape(self.shape)


def index_files(files):
    index = defaultdict(list)
    for path, size in sorted(files.items()):
        match = FILENAME.fullmatch(Path(path).name)
        if match and int(match[4]) == size:
            node, tag, dims, _, port = match.groups()
            index[f'{node}:{int(port)}'].append(Dump(path, tuple(map(int, dims.split('_'))),
                                                      TYPES[tag.lower()], size))
    return index


def calibrate(root, index, anchors, air_types):
    """Infer an offset/type only from whole, nonconstant tensors of this invocation."""
    rules, evidence = {}, []
    for edge, target in anchors.items():
        target = np.asarray(target).squeeze()
        if target.size < 2 or not np.isfinite(target).all() or target.min() == target.max():
            continue
        for dump in index.get(edge, ()):
            options = {(dtype, offset) for (dtype, offset), value in dump.views(root, air_types.get(edge))
                       if np.array_equal(value.squeeze(), target)}
            if options:
                key = (edge, dump.dtype, dump.extra, air_types.get(edge))
                rules[key] = rules.get(key, options) & options
                evidence.append(dict(edge=edge, file=dump.path, matching=sorted(options), format=list(key)))
    return rules, evidence


def decoded(root, index, edge, air_type, rules, padding_offset=None):
    """Collapse byte-identical repeats; preserve every distinct plausible result."""
    records = index.get(edge, [])
    unique, unsupported = {}, []
    for dump in records:
        if not prod(dump.shape) or dump.extra not in (0, 32):
            unsupported.append(dump.path)
            continue
        rule = rules.get((edge, dump.dtype, dump.extra, air_type))
        if rule == set():
            unsupported.append(dump.path + ' (conflicting payload matches)')
            continue
        for fmt, array in dump.views(root, air_type):
            if rule and fmt not in rule:
                continue
            if not rule and fmt[0] != dump.dtype:
                continue  # Prefer the parser's declared type unless a replay anchor proves another encoding.
            if dump.extra and padding_offset is not None and fmt[1] != padding_offset:
                continue
            key = (array.shape, array.dtype.str, array.tobytes())
            if key not in unique:
                unique[key] = (array, [])
            unique[key][1].append(dict(path=dump.path, dtype=array.dtype.name, offset=fmt[1],
                                       extra_bytes=dump.extra, calibrated=bool(rule and len(rule) == 1)))
    return list(unique.values()), unsupported
