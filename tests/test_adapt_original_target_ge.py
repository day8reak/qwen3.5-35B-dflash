"""Original YAML -> packed fixture -> typed GE recipe, never a vendor run."""
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest

from test_original_quant_prepare import original
from test_ge_original_quant import contract_document
from qwen35_moe_dflash.config import Qwen35DFlashConfig
from qwen35_moe_dflash.ge_export import configurations
from qwen35_moe_dflash.original_model_config import OriginalModelConfig
from qwen35_moe_dflash.original_quant_prepare import prepare_original_target


@pytest.fixture
def prepared_case(original, tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    source, _, _, _ = original
    config = OriginalModelConfig.from_yaml(source)
    prepared = tmp_path / "prepared"
    # Test sentinel only. This converter does not implement real QPARAM.
    report = prepare_original_target(source, prepared,
        qparam_converter=lambda values: np.full(values.shape, 77, dtype="<i8"))
    contract = tmp_path / "synthetic-bindings.json"
    contract.write_text(json.dumps(contract_document()))
    _, draft = configurations(tiny=True)
    draft.update(hidden_size=256, vocab_size=32, num_target_layers=1)
    draft["dflash_config"]["target_layer_ids"] = [0]
    draft["dflash_config"]["mask_token_id"] = 31
    checked_draft = Qwen35DFlashConfig.from_dict(draft)
    monkeypatch.setattr(Qwen35DFlashConfig, "from_dict", lambda value: checked_draft)
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "targets/cpro/scripts/adapt_original_target.py"))
    args = SimpleNamespace(prepared=str(prepared), quant_bindings=str(contract),
                           fused_bindings=None, sequence_length=1, past_length=0)
    return module, args, config, prepared, report


def test_prepared_yaml_weights_emit_real_recipe_with_synthetic_operator_schema(prepared_case, tmp_path):
    module, args, config, prepared, _ = prepared_case
    report = module["_build_ge"](args, config, tmp_path / "ge")
    assert report["status"] == "GE_SOURCE_GENERATED"
    assert not report["device_ready"] and report["air_export"] == "NOT_RUN"
    assert report["quantization_contract"]["pending_expert_qparam"] == []
    assert (tmp_path / "ge/export.cpp").is_file()
    graph = json.loads((tmp_path / "ge/graph.json").read_text())
    assert [entry["dtype"] for entry in graph["inputs"][:2]] == ["int8", "float32"]
    assert "target.embed_tokens.weight" not in graph["weights"]
    assert (prepared / "manifest.json").is_file()


@pytest.mark.parametrize("tamper", ["yaml", "config", "weight", "dtype", "qparam"])
def test_ge_rejects_changed_source_or_unready_weights_before_writing(prepared_case, tmp_path, tamper):
    module, args, config, prepared, report = prepared_case
    if tamper == "yaml":
        config.source.write_text(config.source.read_text() + "\n# changed\n")
    elif tamper == "config":
        p = config.model_file / "config.json"
        p.write_text(p.read_text() + "\n")
    elif tamper == "weight":
        p = prepared / report["weights"]["target.norm.weight"]["path"]
        p.write_bytes(bytes(p.stat().st_size))
    else:
        if tamper == "dtype":
            report["weights"]["target.norm.weight"]["dtype"] = "int16"
        else:
            report.update(status="QPARAM_PENDING", export_ready=False)
        (prepared / "manifest.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        module["_build_ge"](args, config, tmp_path / "invalid-ge")
    assert not (tmp_path / "invalid-ge").exists()
