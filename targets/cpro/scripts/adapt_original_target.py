#!/usr/bin/env python3
"""Reuse the original model YAML for quantized weights and direct GE construction."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import uuid

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "model"))


def _qparam_converter(module_name):
    if module_name is None:
        return None
    module = importlib.import_module(module_name)
    converter = getattr(module, "convert_qparam", None)
    if not callable(converter):
        raise ValueError("--qparam-module must expose convert_qparam(FP32 ndarray) -> INT64 ndarray")
    return converter


def _build_ge(args, config, output):
    from qwen35_moe_dflash.common.artifacts import checked_file, save_json
    from qwen35_moe_dflash.config import Qwen35DFlashConfig
    from qwen35_moe_dflash.graph.fused.bindings import OperatorBindings
    from qwen35_moe_dflash.graph.fused.export import _structure_graph
    from qwen35_moe_dflash.graph.fused.quant_bindings import QuantOperatorBindings
    from qwen35_moe_dflash.graph.fused.quant_target import build_quantized_target, emit_quantized_cpp
    from qwen35_moe_dflash.common.hashing import sha256_file

    if not args.prepared or not args.quant_bindings:
        raise ValueError("GE construction requires --prepared and --quant-bindings")
    prepared = Path(args.prepared).resolve()
    manifest = prepared / "manifest.json"
    document = json.loads(manifest.read_text())
    recorded = document.get("sources", {})
    if (document.get("config") != config.text_config()
            or any(recorded.get(key) != value for key, value in config.record().items())
            or recorded.get("yaml_sha256") != sha256_file(config.source)
            or recorded.get("text_config_sha256") != sha256_file(config.model_file / "config.json")):
        raise ValueError("prepared weights do not match this YAML configuration and its source paths")
    if not document.get("export_ready") or document.get("status") != "PREPARED":
        raise ValueError("prepared expert scales still need vendor QPARAM conversion before GE export")
    quant = QuantOperatorBindings.from_file(args.quant_bindings)
    fused = OperatorBindings.from_file(args.fused_bindings) if args.fused_bindings else None
    draft = Qwen35DFlashConfig.from_dict(json.loads((REPO / "specs/draft-config.json").read_text()))
    graph = build_quantized_target(document["config"], draft, args.sequence_length,
        quant_bindings=quant, quant_descriptors=document["quant_descriptors"],
        past_length=args.past_length, fused_ops=("gdn",) if fused else (), bindings=fused,
        output_positions=None if getattr(args, "output_pos", None) is None else (args.output_pos,),
        return_logits=getattr(args, "full_logits", False))
    if graph.quantization_contract["pending_expert_qparam"]:
        raise ValueError("prepared expert scales still need vendor QPARAM conversion; FP32 cannot be used as INT64")
    paths = {}
    for key, tensor in graph.weights.items():
        record = document["weights"].get(key)
        if not isinstance(record, dict) or record.get("shape") != list(tensor.shape) or record.get("dtype") != tensor.dtype:
            raise ValueError(f"prepared weight descriptor does not match GE graph: {key}")
        paths[key] = str(checked_file(prepared, record))
    output.mkdir(parents=True, exist_ok=False)
    source = emit_quantized_cpp(graph, paths, recipe_dir=output / "recipe", external_weights=True)
    (output / "export.cpp").write_text(source)
    save_json(output / "graph.json", _structure_graph(graph))
    report = {"schema_version": 1, "status": "GE_SOURCE_GENERATED", "output": str(output),
              "prepared": str(prepared), "prepared_manifest_sha256": sha256_file(manifest),
              "quantization_contract": graph.quantization_contract,
              "output_contract": graph.output_contract,
              "sequence_length": args.sequence_length, "past_length": args.past_length,
              "air_export": "NOT_RUN", "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN",
              "device_ready": False, "sources": config.record()}
    save_json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="original qwen3_5_moe.yaml, kept outside Git")
    parser.add_argument("--stage", choices=("inspect", "prepare", "ge"), default="inspect")
    parser.add_argument("--output", help="new output directory below AI_RUN_DIR")
    parser.add_argument("--qparam-module", help="explicit vendor converter module; no guessed QPARAM encoding")
    parser.add_argument("--prepared", help="previous original-quant preparation directory for GE")
    parser.add_argument("--quant-bindings", help="complete selected-DDK quantized GE schema JSON")
    parser.add_argument("--fused-bindings", help="existing checked GDN binding; Conv and RMS remain base operators")
    parser.add_argument("--sequence-length", type=int, choices=(1, 64), default=64)
    parser.add_argument("--past-length", type=int, default=0)
    parser.add_argument("--output-pos", type=int,
                        help="zero-based position within this input; -1 selects last; omitted selects all")
    parser.add_argument("--full-logits", action="store_true",
                        help="append full-vocabulary FP32 logits for selected positions")
    args = parser.parse_args(argv)
    from qwen35_moe_dflash.common.artifacts import output_path
    from qwen35_moe_dflash.target.original_model_config import OriginalModelConfig

    try:
        run = os.environ.get("AI_RUN_DIR")
        if not run:
            raise ValueError("AI_RUN_DIR is required; source targets/cpro/scripts/env.sh first")
        config = OriginalModelConfig.from_yaml(args.config)
        config.validate_paths()
        output = output_path(args.output or Path(run) / ("original-quant-" + uuid.uuid4().hex[:12]))
        print("original-quant: " + json.dumps(config.record(), ensure_ascii=False), flush=True)
        if args.stage == "ge":
            report = _build_ge(args, config, output)
        else:
            if args.stage == "inspect" and args.qparam_module:
                raise ValueError("inspection does not load or execute a QPARAM converter")
            from qwen35_moe_dflash.target.original_quant_prepare import prepare_original_target
            report = prepare_original_target(args.config, output,
                qparam_converter=_qparam_converter(args.qparam_module), inspect_only=args.stage == "inspect",
                progress=lambda row: print(json.dumps(row, ensure_ascii=False), flush=True))
        print(json.dumps({key: report.get(key) for key in
                         ("status", "output", "device_ready", "model_compilation", "device_execution")},
                         ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        print(json.dumps({"status": "FAILED", "stage": args.stage,
                          "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
