"""User-declared OMG parameter presets; these do not inspect or validate AIR."""


EXISTING_AIR_PROFILE_NAMES = ("cpro-40block-64-1", "cpro-40block-prefill-64", "cpro-40block-decode-1")
_STATIC_SEQUENCES = {"cpro-40block-prefill-64": 64, "cpro-40block-decode-1": 1}


def get_existing_air_profile(name: str) -> dict:
    """Return a fresh copy of the frozen user-provided compilation contract."""
    if name not in EXISTING_AIR_PROFILE_NAMES:
        raise ValueError("unknown existing AIR profile: " + str(name))
    entries = [("arg1_1", (1, -1, 2048)),
               ("arg113_1", (1, 1, -1, 2048)),
               ("arg107_1", (1, -1))]
    block_pattern = ("linear", "linear", "linear", "full")
    for block in range(40):
        shapes = ((64, 32, 32, 16), (64, 32, 32, 16)) if block_pattern[block % 4] == "full" else (
            (1, 8192, 4), (1, 32, 128, 128))
        entries.extend((f"RefData_{2 * block + index}", shape) for index, shape in enumerate(shapes))
    entries.extend([("arg2_1", (-1,)), ("arg1238_1", (1,)), ("arg0_1", (1, -1, 1)),
                    ("arg114_1", (1,)), ("arg14_1", (128,)), ("arg18_1", (1,))])
    declared_axes = [{"input_name": key, "axis": axis}
                     for key, shape in entries for axis, size in enumerate(shape) if size == -1]
    sequence = _STATIC_SEQUENCES.get(name)
    if sequence is not None:
        entries = [(key, tuple(sequence if size == -1 else size for size in shape)) for key, shape in entries]
    profile = {
        "name": name,
        "input_shape": ";".join(key + ":" + ",".join(map(str, shape)) for key, shape in entries),
        "dynamic_dims": None if sequence is not None else "64,64,64,64,64;1,1,1,1,1",
        "limit_partition_size": False,
        "save_weights_as_external_data": True,
        "declared_model_facts": {
            "source": "USER_PROVIDED_CONTRACT",
            "air_validation": "NOT_PERFORMED",
            "block_count": 40,
            "block_pattern": list(block_pattern),
            "refdata_count": 80,
            "input_count": len(entries),
            "dynamic_axis_order": [{"input_name": key, "axis": axis}
                                   for key, shape in entries for axis, size in enumerate(shape) if size == -1],
            "scope": "Shapes, block roles and compiler settings are user declarations, not facts verified from AIR or OMC.",
        },
    }
    if sequence is not None:
        profile["declared_model_facts"].update(
            profile_mode="STATIC_FIXED_GEAR", fixed_sequence_length=sequence,
            resolved_dynamic_axes=[dict(axis, extent=sequence) for axis in declared_axes],
            scope="Static fixed-gear shape request only; original 80 states and fixed extents retained. AIR shape compatibility, decode semantics and device execution remain unverified.")
    return profile
