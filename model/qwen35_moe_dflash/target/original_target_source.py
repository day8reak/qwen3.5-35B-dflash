"""Prepare a small, derived copy of the user's original Cpro exporter.

No vendor source, weights, AIR or OMC are shipped in this module. The selected
Cpro decoder loop is patched locally to append real post-layer features. All
original computation statements and the first 81 outputs are retained.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import shlex

from qwen35_moe_dflash.common.artifacts import output_path, save_json


LAYER_IDS = (1, 6, 11, 16, 22, 27, 32, 37)
WRAPPER = "models/export_model_wrapper_qwen3_5_moe.py"
EXPORTER = "export_glm_model_qwen3_5_moe.py"
SOURCE_FILES = (EXPORTER, "utils.py", WRAPPER,
                "models/modeling_qwen3_5_moe_hiai_cpro.py",
                "models/modeling_qwen3_5_moe_hiai_nz.py")
POLICY = "original-cpro-post-decoder-features-v1"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _dump(node):
    return ast.dump(node, include_attributes=False)


def patch_cpro_wrapper(source: str) -> str:
    """Patch one structurally verified original class; reject unknown variants."""
    tree = ast.parse(source)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "Qwen3_5MoeForCausalLMCproWrapper"]
    if len(classes) != 1:
        raise ValueError("expected exactly one original Cpro wrapper class")
    cls = classes[0]
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"]
    if len(methods) != 1:
        raise ValueError("expected exactly one Cpro forward")
    method = methods[0]
    if any(isinstance(n, ast.Name) and n.id.startswith("_dflash_") for n in ast.walk(method)):
        raise ValueError("Cpro wrapper is already patched or uses reserved DFlash names")
    expected_iter = ast.parse("enumerate(self.model.model.layers)", mode="eval").body
    loops = [n for n in method.body if isinstance(n, ast.For) and _dump(n.iter) == _dump(expected_iter)]
    if len(loops) != 1:
        raise ValueError("expected one top-level full original decoder loop")
    loop = loops[0]
    expected_target = ast.parse("for idx, decoder_layer in layers:\n    pass").body[0].target
    if _dump(loop.target) != _dump(expected_target):
        raise ValueError("unknown decoder loop index or layer variable")
    hidden = ast.parse("hidden_states = layer_outputs[0]").body[0]
    captures = [n for n in loop.body if _dump(n) == _dump(hidden)]
    if len(captures) != 1:
        raise ValueError("expected one direct post-decoder hidden-state assignment")
    returns = [n for n in ast.walk(method) if isinstance(n, ast.Return)]
    expected_return = ast.parse("return lm_logits, *kv_caches_out").body[0]
    if len(returns) != 1 or _dump(returns[0]) != _dump(expected_return) or returns[0] not in method.body:
        raise ValueError("original Cpro must return only logits followed by all cache states")
    # These anchors distinguish the full original forward from a miniature or
    # reordered lookalike. No import of torch_npu or loading of weights is needed.
    calls = [n for n in ast.walk(loop) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "decoder_layer"]
    if len(calls) != 1 or not calls[0].args or _dump(calls[0].args[0]) != "Name(id='hidden_states', ctx=Load())":
        raise ValueError("unknown original decoder invocation")
    required_keywords = {"past_key_values", "layer_idx", "new_kv_cache_pos", "use_cache"}
    if not required_keywords <= {n.arg for n in calls[0].keywords}:
        raise ValueError("original decoder state/control arguments are missing")
    cache_extends = [n for n in ast.walk(loop) if isinstance(n, ast.Call)
                     and _dump(n.func) == _dump(ast.parse("kv_caches_out.extend", mode="eval").body)]
    if len(cache_extends) != 1:
        raise ValueError("unknown original cache output collection")
    rows = source.splitlines(keepends=True)
    if any("\t" in rows[n.lineno - 1][:n.col_offset] for n in (loop, captures[0], returns[0])):
        raise ValueError("tab-indented wrappers are not supported")
    base = " " * loop.col_offset
    inner = " " * captures[0].col_offset
    before = (
        f"{base}if len(self.model.model.layers) != 40 or len(past_key_values) != 40:\n"
        f"{base}    raise ValueError('DFlash export requires all 40 target layers and cache pairs')\n"
        f"{base}if not use_cache:\n"
        f"{base}    raise ValueError('DFlash export requires all 80 state outputs')\n"
        f"{base}_dflash_hidden = []\n"
    )
    capture = (
        f"{inner}if idx in {LAYER_IDS!r}:\n"
        f"{inner}    if hidden_states.ndim != 3 or hidden_states.shape[-1] != 2048:\n"
        f"{inner}        raise ValueError('DFlash target hidden must have shape [B,T,2048]')\n"
        f"{inner}    _dflash_hidden.append(hidden_states.to(dtype=torch.float16).clone())\n"
    )
    tail = (
        f"{base}if len(_dflash_hidden) != 8 or len(kv_caches_out) != 80:\n"
        f"{base}    raise ValueError('incomplete target feature or state outputs')\n"
        f"{base}return lm_logits, *kv_caches_out, torch.cat(_dflash_hidden, dim=-1)\n"
    )
    # Insert text at AST-verified statement boundaries. Bytes outside these
    # three changes (including other wrappers and quantization) stay unchanged.
    edits = [(loop.lineno - 1, loop.lineno - 1, before),
             (captures[0].end_lineno, captures[0].end_lineno, capture),
             (returns[0].lineno - 1, returns[0].end_lineno, tail)]
    for first, last, text in sorted(edits, reverse=True):
        rows[first:last] = [text]
    patched = "".join(rows)
    ast.parse(patched)
    return patched


def prepare_original_target_source(source_root, output, *, config=None, sequence_length=55):
    """Copy five small source files and config; never copy model payloads."""
    source_root = Path(source_root).resolve(strict=True)
    output = output_path(output)
    if type(sequence_length) is not int or not 1 <= sequence_length <= 64:
        raise ValueError("export sequence length must be in 1..64")
    config = Path(config).resolve(strict=True) if config else source_root / "config/qwen3_5_moe.yaml"
    payloads = {}
    sources = {}
    for name in SOURCE_FILES:
        path = (source_root / name).resolve(strict=True)
        if source_root not in path.parents or not path.is_file() or path.stat().st_size > 2 * 1024**2:
            raise ValueError(f"source is not a bounded local file: {name}")
        raw = path.read_bytes()
        ast.parse(raw.decode("utf-8"))
        payloads[name] = raw
        sources[name] = {"path": str(path), "bytes": len(raw), "sha256": _sha(raw)}
    if not config.is_file() or config.stat().st_size > 1024**2:
        raise ValueError("config is not a bounded local file")
    config_raw = config.read_bytes()
    sources["config/qwen3_5_moe.yaml"] = {"path": str(config), "bytes": len(config_raw), "sha256": _sha(config_raw)}
    payloads["config/qwen3_5_moe.yaml"] = config_raw
    original = payloads[WRAPPER]
    payloads[WRAPPER] = patch_cpro_wrapper(original.decode("utf-8")).encode("utf-8")
    # The exporter and every caller argument remain original. Unknown source
    # variants must be reviewed before producing an expensive full artifact.
    export_tree = ast.parse(payloads[EXPORTER].decode("utf-8"))
    exports = [n for n in ast.walk(export_tree) if isinstance(n, ast.Call)
               and _dump(n.func) == _dump(ast.parse("torchair.dynamo_export", mode="eval").body)]
    if len(exports) != 1 or not any(n.arg == "model" and isinstance(n.value, ast.Name)
                                  and n.value.id == "model_wrapper" for n in exports[0].keywords):
        raise ValueError("unknown original TorchAir export call")
    air_folder = f"glm_export_air_cpro_40block_{sequence_length}_quant"
    command = ["python3", EXPORTER, "mode=export", "quant_mode=enable", "num_hidden_layers=40",
               f"seq_len={sequence_length}", "topK=16", "config=./config/qwen3_5_moe.yaml"]
    script = ("#!/usr/bin/env bash\nset -euo pipefail\n"
              f"cd {shlex.quote(str(output / 'source'))}\n"
              f"test ! -e {shlex.quote(air_folder)} || {{ echo 'Retained export exists; refusing to overwrite it' >&2; exit 1; }}\n"
              "export PYTHONDONTWRITEBYTECODE=1\n"
              f"exec {shlex.join(command)}\n")
    payloads = {"source/" + name: raw for name, raw in payloads.items()}
    payloads["export.sh"] = script.encode()
    expected = {name: {"bytes": len(raw), "sha256": _sha(raw)} for name, raw in payloads.items()}
    receipt_path = output / "receipt.json"
    if output.exists():
        if not receipt_path.is_file() or receipt_path.is_symlink() or receipt_path.stat().st_size > 1024**2:
            raise ValueError("output exists without a completed source receipt; preserve it and choose a new output")
        retained = json.loads(receipt_path.read_text())
        if (not isinstance(retained, dict) or retained.get("policy") != POLICY or retained.get("source_files") != sources
                or retained.get("generated_files") != expected):
            raise ValueError("retained source/config/patch identity changed; refusing to overwrite")
        for name, row in expected.items():
            target = output / name
            if (target.is_symlink() or output not in target.resolve().parents
                    or not target.is_file() or _sha(target.read_bytes()) != row["sha256"]):
                raise ValueError(f"retained derived source changed: {name}")
        return retained
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in payloads.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
    (output / "export.sh").chmod(0o755)
    receipt = {"schema_version": 1, "artifact_kind": "qwen35-original-target-feature-source",
               "status": "PREPARED_SOURCE_ONLY", "policy": POLICY, "source_files": sources,
               "generated_files": expected, "feature_layer_ids": list(LAYER_IDS),
               "target_layers": 40, "original_output_count": 81, "output_count": 82,
               "feature_output_index": 81, "feature_shape": [1, "T", 16384], "feature_dtype": "float16",
               "export_sequence_length": sequence_length, "export_script": str(output / "export.sh"),
               "expected_air": str(output / "source" / air_folder / "export.air"),
               "model_execution": "NOT_RUN", "air_export": "NOT_RUN", "device_execution": "NOT_RUN",
               "scope": "Original quantized target source plus auxiliary real features; no weights copied, no AIR/OMC execution or accuracy claim"}
    save_json(receipt_path, receipt)
    return receipt
