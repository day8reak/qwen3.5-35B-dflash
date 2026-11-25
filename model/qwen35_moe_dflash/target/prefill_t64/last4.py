"""The two measured L35 exits and the tokenizer for a paired L36–L39 replay."""
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from ...common.artifacts import share_for_transfer
from ..layer_probe.common import checked, identity, read_array, save_json, sha
from .l0_split import CPU_BASIS


SOURCES = ('cpu', '310p')
LAYERS = [36, 37, 38, 39]
BUNDLE = 'last4-inputs'


def publish(run, capture, captured, cpu_hidden):
    capture_sha = sha(capture / 'manifest.json')
    block = captured['blocks'][35]
    if block['layer'] != 35 or block['block_type'] != 'full_attention':
        raise ValueError('Expected the 310P L35 decoder exit')
    measured = read_array(capture, block['hidden'])
    dest = run / BUNDLE
    with tempfile.TemporaryDirectory(dir=run, prefix='last4-prepare-') as tmp:
        staged = Path(tmp) / BUNDLE
        staged.mkdir()
        inputs = {}
        for name, array in zip(SOURCES, (cpu_hidden, measured), strict=True):
            if array.shape != (1, 64, 2048) or array.dtype != np.float16 or not np.isfinite(array).all():
                raise ValueError(f'{name}: expected the finite physical FP16 L35 decoder output')
            path = staged / (name + '.bin')
            array.astype('<f4').tofile(path)
            inputs[name] = dict(identity(path, staged), dtype='float32', shape=list(array.shape))
        vocab = staged / 'token-text.json'
        shutil.copyfile(checked(capture, captured['token_text_map']), vocab)
        manifest = dict(capture_sha256=capture_sha, cpu_basis=CPU_BASIS, source_layer=35, layers=LAYERS,
                        inputs=inputs, token_text_map=identity(vocab, staged), output_pos=18)
        save_json(staged/'manifest.json', manifest)
        if dest.exists():
            if json.loads((dest/'manifest.json').read_bytes()) != manifest:
                raise ValueError(f'{dest}: retained L35 inputs or tokenizer differ from this capture')
            for row in (*inputs.values(), manifest['token_text_map']):
                checked(dest, row)
        else:
            share_for_transfer(staged)
            staged.rename(dest)


def token_result(logits, vocab):
    pieces = vocab['pieces']
    if logits.shape != (1, 1, len(pieces)) or not np.isfinite(logits).all():
        raise ValueError('Expected one finite full-vocabulary logits row for position 18')
    token = int(logits.argmax())
    return dict(token_id=token, text=pieces[token], special=token in vocab['special_ids'])
