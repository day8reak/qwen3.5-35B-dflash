"""Select the original first wrapper's external embedding branch for decoder goldens."""
from ..layer_probe.common import read_tensor, sized_file


def golden_embedding(source, config):
    import numpy as np
    import yaml
    from ..original_quant import load_embedding

    cfg = yaml.safe_load(config.read_text())
    weight, scale = load_embedding(source / cfg['embedding_weight_path'], source / cfg['embedding_scale_path'],
                                   (int(cfg.get('vocab_size', 248320)), 2048))

    def rows(ids):
        if np.any(ids < 0) or np.any(ids >= len(weight)):
            raise ValueError('Token ID outside the configured embedding tables')
        return weight[ids], scale[ids]
    return rows


def retained_embedding(fixture, ref):
    """Replay the operands actually observed by 310P, including physical padding."""
    import numpy as np

    if 'embedding_inputs' not in ref:
        return None  # Earlier fixtures observed the driver's native lookup.
    def read(spec):
        return read_tensor(sized_file(fixture, spec), spec)
    expected = read(ref['inputs']['input_ids'])
    operands = tuple(read(ref['embedding_inputs'][key]) for key in ('inputs_embeds', 'embed_scale'))
    codes, scales = operands
    if (codes.shape != (*expected.shape, 2048) or codes.dtype != np.int8 or
            scales.shape != (*expected.shape, 1) or scales.dtype != np.float32):
        raise ValueError(f'{fixture}: invalid retained embedding operands')

    def rows(ids):
        if not np.array_equal(ids, expected):
            raise ValueError('Live token IDs differ from the retained embedding inputs')
        return operands
    return rows
