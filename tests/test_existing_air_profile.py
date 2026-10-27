"""The frozen existing-AIR contract must preserve the user's complete port order."""
import pytest

from qwen35_moe_dflash.existing_air_profile import EXISTING_AIR_PROFILE_NAMES, get_existing_air_profile


# Independent literal: do not regenerate this expected command from the profile's
# block-pattern loop, since a shared numbering/pattern error would then pass.
EXPECTED_INPUT_SHAPE = """arg1_1:1,-1,2048
arg113_1:1,1,-1,2048
arg107_1:1,-1
RefData_0:1,8192,4
RefData_1:1,32,128,128
RefData_2:1,8192,4
RefData_3:1,32,128,128
RefData_4:1,8192,4
RefData_5:1,32,128,128
RefData_6:64,32,32,16
RefData_7:64,32,32,16
RefData_8:1,8192,4
RefData_9:1,32,128,128
RefData_10:1,8192,4
RefData_11:1,32,128,128
RefData_12:1,8192,4
RefData_13:1,32,128,128
RefData_14:64,32,32,16
RefData_15:64,32,32,16
RefData_16:1,8192,4
RefData_17:1,32,128,128
RefData_18:1,8192,4
RefData_19:1,32,128,128
RefData_20:1,8192,4
RefData_21:1,32,128,128
RefData_22:64,32,32,16
RefData_23:64,32,32,16
RefData_24:1,8192,4
RefData_25:1,32,128,128
RefData_26:1,8192,4
RefData_27:1,32,128,128
RefData_28:1,8192,4
RefData_29:1,32,128,128
RefData_30:64,32,32,16
RefData_31:64,32,32,16
RefData_32:1,8192,4
RefData_33:1,32,128,128
RefData_34:1,8192,4
RefData_35:1,32,128,128
RefData_36:1,8192,4
RefData_37:1,32,128,128
RefData_38:64,32,32,16
RefData_39:64,32,32,16
RefData_40:1,8192,4
RefData_41:1,32,128,128
RefData_42:1,8192,4
RefData_43:1,32,128,128
RefData_44:1,8192,4
RefData_45:1,32,128,128
RefData_46:64,32,32,16
RefData_47:64,32,32,16
RefData_48:1,8192,4
RefData_49:1,32,128,128
RefData_50:1,8192,4
RefData_51:1,32,128,128
RefData_52:1,8192,4
RefData_53:1,32,128,128
RefData_54:64,32,32,16
RefData_55:64,32,32,16
RefData_56:1,8192,4
RefData_57:1,32,128,128
RefData_58:1,8192,4
RefData_59:1,32,128,128
RefData_60:1,8192,4
RefData_61:1,32,128,128
RefData_62:64,32,32,16
RefData_63:64,32,32,16
RefData_64:1,8192,4
RefData_65:1,32,128,128
RefData_66:1,8192,4
RefData_67:1,32,128,128
RefData_68:1,8192,4
RefData_69:1,32,128,128
RefData_70:64,32,32,16
RefData_71:64,32,32,16
RefData_72:1,8192,4
RefData_73:1,32,128,128
RefData_74:1,8192,4
RefData_75:1,32,128,128
RefData_76:1,8192,4
RefData_77:1,32,128,128
RefData_78:64,32,32,16
RefData_79:64,32,32,16
arg2_1:-1
arg1238_1:1
arg0_1:1,-1,1
arg114_1:1
arg14_1:128
arg18_1:1""".replace("\n", ";")


def test_frozen_profile_exact_command_and_user_declared_scope():
    assert EXISTING_AIR_PROFILE_NAMES == ("cpro-40block-64-1",)
    profile = get_existing_air_profile("cpro-40block-64-1")
    assert profile["input_shape"] == EXPECTED_INPUT_SHAPE
    assert profile["dynamic_dims"] == "64,64,64,64,64;1,1,1,1,1"
    assert profile["limit_partition_size"] is False
    assert profile["save_weights_as_external_data"] is True
    facts = profile["declared_model_facts"]
    assert facts["source"] == "USER_PROVIDED_CONTRACT"
    assert facts["air_validation"] == "NOT_PERFORMED"
    assert facts["block_count"] == 40 and facts["refdata_count"] == 80 and facts["input_count"] == 89
    assert facts["block_pattern"] == ["linear", "linear", "linear", "full"]


def test_refdata_shapes_and_dynamic_axes_follow_input_order():
    profile = get_existing_air_profile("cpro-40block-64-1")
    entries = [(name, tuple(map(int, shape.split(","))))
               for name, shape in (entry.split(":") for entry in profile["input_shape"].split(";"))]
    names = [name for name, _ in entries]
    assert len(names) == len(set(names)) == 89
    assert names[3:83] == [f"RefData_{index}" for index in range(80)]
    expected_axes = [("arg1_1", 1), ("arg113_1", 2), ("arg107_1", 1), ("arg2_1", 0), ("arg0_1", 1)]
    dynamic_axes = [(name, axis) for name, shape in entries for axis, size in enumerate(shape) if size == -1]
    assert dynamic_axes == expected_axes
    assert profile["declared_model_facts"]["dynamic_axis_order"] == [
        {"input_name": name, "axis": axis} for name, axis in expected_axes]
    for values, size in zip(profile["dynamic_dims"].split(";"), (64, 1)):
        assert list(map(int, values.split(","))) == [size] * len(expected_axes)
    full_indices = {6, 7, 14, 15, 22, 23, 30, 31, 38, 39, 46, 47, 54, 55, 62, 63, 70, 71, 78, 79}
    for index, (_, shape) in enumerate(entries[3:83]):
        assert shape == ((64, 32, 32, 16) if index in full_indices else
                         (1, 32, 128, 128) if index % 2 else (1, 8192, 4))


def test_profile_results_are_independent():
    changed = get_existing_air_profile("cpro-40block-64-1")
    changed["input_shape"] = "changed"
    changed["declared_model_facts"]["block_pattern"][0] = "changed"
    changed["declared_model_facts"]["dynamic_axis_order"][0]["input_name"] = "changed"
    fresh = get_existing_air_profile("cpro-40block-64-1")
    assert fresh["input_shape"] == EXPECTED_INPUT_SHAPE
    assert fresh["declared_model_facts"]["block_pattern"][0] == "linear"
    assert fresh["declared_model_facts"]["dynamic_axis_order"][0]["input_name"] == "arg1_1"


@pytest.mark.parametrize("name", ["", "cpro-40block-64", "CPRO-40BLOCK-64-1", None])
def test_unknown_profile_is_rejected(name):
    with pytest.raises(ValueError, match="unknown existing AIR profile"):
        get_existing_air_profile(name)
