"""GE port metadata shared by the TorchAir converter and the single-op AIR."""

GE_TYPE = 'SiluPatch'


def vector_desc(desc, length):
    desc.layout = 'ND'
    desc.device_type = 'NPU'
    desc.shape.dim[:] = [length]
    desc.attr['origin_format'].s = b'ND'
    desc.attr['origin_format_for_int'].i = 2
    desc.attr['format_for_int'].i = 2
    desc.attr['origin_shape_initialized'].b = False


def port_names(op, input_name, output_name):
    for direction, name in (('input', input_name), ('output', output_name)):
        keys = op.attr[f'_{direction}_name_key'].list
        keys.val_type = keys.VT_LIST_STRING
        keys.s[:] = [name.encode()]
        slots = op.attr[f'_{direction}_name_value'].list
        slots.val_type = slots.VT_LIST_INT
        slots.i[:] = [0]
