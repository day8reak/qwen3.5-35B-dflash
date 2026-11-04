"""Exact prepared FP16 embedding lookup outside the diagnostic model call."""
import math
import re

import numpy as np


def source_contract(prepared):
    from qwen35_moe_dflash.config import Qwen35DFlashConfig
    c = Qwen35DFlashConfig.from_dict(prepared['draft_config'])
    row = prepared['weights']['target.embed_tokens.weight']
    if (prepared.get('precision') != 'float16' or prepared.get('quantization') is not None
            or c.input_embedding_scale != 1.0 or row.get('dtype') != 'float16'
            or row.get('shape') != [c.vocab_size, c.hidden_size]
            or row.get('bytes') != c.vocab_size * c.hidden_size * 2
            or not re.fullmatch('[0-9a-f]{64}', row.get('sha256', ''))):
        raise ValueError('host embedding needs the original hash-bound FP16 table and unit scale')
    return dict(policy='prepared-fp16-block-row-lookup-v1',
                weight={key: row[key] for key in ('dtype', 'shape', 'bytes', 'sha256')},
                mask_token_id=c.mask_token_id, embedding_scale=1.0,
                input='block_embeddings', order='ANCHOR_THEN_K_MASK_ROWS',
                arithmetic='NONE; COPY_ORIGINAL_FP16_BYTES')


def lookup_block(table, anchor, mask_token_id, candidates):
    if (not isinstance(table, np.ndarray) or table.ndim != 2 or table.dtype != np.dtype('<f2')
            or not math.prod(table.shape) or type(candidates) is not int or candidates not in (1, 3, 7, 15)
            or any(type(i) is not int or not 0 <= i < table.shape[0] for i in (anchor, mask_token_id))):
        raise ValueError('invalid host embedding table, token IDs or candidate gear')
    result = np.empty((1, candidates + 1, table.shape[1]), dtype='<f2')
    result[0, 0] = table[anchor]
    result[0, 1:] = table[mask_token_id]
    if not np.isfinite(result).all():
        raise ValueError('nonfinite host embedding rows')
    return result
