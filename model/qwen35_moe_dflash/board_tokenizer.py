"""Pinned original tokenizer assets for board-local UTF-8 text I/O.

Host tokenization is only an admission/parity check. The board re-encodes the
text, checks the admitted IDs, and decodes its own final tokens.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / 'specs/tokenizer.lock.json'
PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
MODES = ('chat', 'raw')


def render(text, mode='chat'):
    if mode not in MODES:
        raise ValueError('prompt format must be chat or raw')
    if not isinstance(text, str) or not text or len(text.encode('utf-8')) > 16384 or '\0' in text:
        raise ValueError('prompt must be nonempty UTF-8 text without NUL, at most 16KiB')
    if mode == 'raw':
        return text
    if text.strip().startswith('<tool_response>') and text.strip().endswith('</tool_response>'):
        raise ValueError('the original chat template does not accept a tool response as its only user query')
    # Exact locked template subset: one user, no tools, enable_thinking=false.
    return '<|im_start|>user\n' + text.strip() + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def checked_sources(directory):
    lock = json.loads(LOCK.read_text())
    directory = Path(directory)
    for name, record in lock['files'].items():
        path = directory / name
        if path.stat().st_size != record['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('tokenizer source differs from locked original model: ' + str(path))
    return lock


def reference(directory, *, verify_sources=True):
    if verify_sources:
        checked_sources(directory)
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise ValueError('text preparation requires the host Python tokenizers package; the board has no Python dependency') from error
    return Tokenizer.from_file(str(Path(directory) / 'tokenizer.json'))


def _byte_decoder():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars = bs[:]
    for byte in range(256):
        if byte not in bs:
            bs.append(byte)
            chars.append(256 + sum(c >= 256 for c in chars))
    return {chr(c): b for b, c in zip(bs, chars)}


def build_bundle(directory, output, *, verify_sources=True):
    lock = checked_sources(directory) if verify_sources else None
    data = json.loads((Path(directory) / 'tokenizer.json').read_text())
    model = data['model']; vocab = model['vocab']; decode = _byte_decoder()
    if (data['normalizer'] != {'type': 'NFC'} or model['type'] != 'BPE'
            or data['pre_tokenizer']['pretokenizers'][0]['pattern'] != {'Regex': PATTERN}
            or len(vocab) != 248044 or set(vocab.values()) != set(range(248044))):
        raise ValueError('unsupported tokenizer structure')
    u32 = lambda n: struct.pack('<I', n)
    blob = lambda b: u32(len(b)) + b
    raw = bytearray(b'QWTOK001')
    raw += blob(PATTERN.encode()) + u32(len(vocab))
    for token, index in sorted(vocab.items(), key=lambda item: item[1]):
        raw += blob(bytes(decode[c] for c in token))
    raw += u32(len(data['added_tokens']))
    for row in data['added_tokens']:
        if any(row.get(k) for k in ('single_word', 'lstrip', 'rstrip', 'normalized')):
            raise ValueError('unsupported added-token normalization')
        raw += u32(row['id']) + u32(int(row['special'])) + blob(row['content'].encode())
    raw += u32(len(model['merges']))
    for pair in model['merges']:
        left, right = pair if isinstance(pair, list) else pair.split(' ')
        raw += u32(vocab[left]) + u32(vocab[right]) + u32(vocab[left + right])
    output = Path(output); output.write_bytes(raw)
    return {'path': str(output.resolve()), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
            'source': lock, 'format': 'QWTOK001', 'normalizer': 'NFC', 'vocab_size': 248070}


def acquire(directory):
    """Explicit small public-asset download, never model weights or device access."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    lock = json.loads(LOCK.read_text())
    for name, record in lock['files'].items():
        path = directory / name
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']:
                continue
            raise ValueError('refusing to overwrite a different tokenizer source: ' + str(path))
        url = f"https://huggingface.co/{lock['repo']}/resolve/{lock['revision']}/{name}"
        with urllib.request.urlopen(url, timeout=90) as stream:
            raw = stream.read(record['bytes'] + 1)
        if len(raw) != record['bytes'] or hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError('downloaded tokenizer identity differs: ' + name)
        path.write_bytes(raw)
    checked_sources(directory)


def prepare_text(directory, text, mode, out, *, verify_sources=True):
    from .board_serial_dflash import _hash_source
    if not verify_sources:
        from .runtime_config import file_record as _hash_source
    formatted = render(text, mode)
    tokenizer = reference(directory, verify_sources=verify_sources)
    ids = tokenizer.encode(formatted, add_special_tokens=False).ids
    if not 1 <= len(ids) <= 63:
        raise ValueError(f'text with template contains {len(ids)} tokens; current S64 draft requires 1..63 before generation')
    out = Path(out)
    prompt = out / 'prompt.txt'; prompt.write_bytes(text.encode('utf-8'))
    bundle = out / 'tokenizer.bin'; build_bundle(directory, bundle, verify_sources=verify_sources)
    shadow = out / 'prompt-admission-ids.bin'; shadow.write_bytes(struct.pack('<' + 'i'*len(ids), *ids))
    return ids, _hash_source(shadow), {'bundle': _hash_source(bundle), 'prompt': _hash_source(prompt),
        'prompt_format': mode, 'source_directory': str(Path(directory).resolve()),
        'source': json.loads(LOCK.read_text()), 'execution': 'BOARD_LOCAL_ENCODE_AND_DECODE',
        'host_scope': 'ADMISSION_AND_REFERENCE_PARITY_ONLY', 'eos_policy': 'FIXED_GENERATION_BUDGET_UNCHANGED'}


def validate_text_result(manifest, loop):
    """Read-only final text check; never supplies text back to the running board."""
    info = manifest['text_io']; observed = loop.get('text_io')
    if observed is None and loop.get('status') == 'FAILED':
        return  # Preserve an earlier native failure; text conversion may not run.
    expected = dict(execution='BOARD_LOCAL_ENCODE_AND_DECODE', prompt_format=info['prompt_format'],
        tokenizer_sha256=info['bundle']['sha256'], prompt_sha256=info['prompt']['sha256'],
        prompt_token_count=len(manifest['prompt_token_ids']), host_admission_ids_matched=True)
    if not isinstance(observed, dict) or any(observed.get(k) != v for k, v in expected.items()):
        raise ValueError('returned text is not bound to this board tokenizer and prompt')
    tokenizer = reference(info['source_directory'])
    ids = loop['generated_tokens']
    unknown = [i for i in ids if i >= tokenizer.get_vocab_size()]
    def decoded(skip):
        # Preserve unknown padded head IDs instead of silently hiding them.
        result = []; group = []
        for i in ids:
            if i in unknown:
                result.extend((tokenizer.decode(group, skip_special_tokens=skip), '\ufffd')); group = []
            else: group.append(i)
        result.append(tokenizer.decode(group, skip_special_tokens=skip))
        return ''.join(result)
    if (observed.get('unmapped_generated_token_ids') != unknown
            or loop.get('generated_text') != decoded(True)
            or loop.get('generated_text_with_special_tokens') != decoded(False)):
        raise ValueError('board text decode differs from the original tokenizer reference')
