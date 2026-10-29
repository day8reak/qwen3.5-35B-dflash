"""Header-only preflight for the original-machine PyTorch/TorchAir route.

This keeps the supplied Cpro forward and its original loader/RefData ABI.
SOURCE_READY means that required files and tensor headers exist, not that any
tensor payload, vendor dependency, AIR export or device execution has passed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex

import yaml

from .artifacts import output_path, save_json
from .ge_weights import SafeTensorFiles
from .hashing import sha256_file
from .original_model_config import OriginalModelConfig
from .original_quant import OriginalQuantWeights, canonicalkeys
from .original_quant_prepare import _embedding_record
from .original_target_source import prepare_original_target_source
from .target_weight_shapes import required_weight_shapes


_CONFIG_FIELDS = (
    "num_hidden_layers", "hidden_size", "vocab_size", "layer_types",
    "num_attention_heads", "num_key_value_heads", "head_dim", "num_experts",
    "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size",
    "linear_conv_kernel_dim", "linear_num_key_heads", "linear_key_head_dim",
    "linear_num_value_heads", "linear_value_head_dim", "attention_bias",
    "hidden_act", "rms_norm_eps", "mlp_only_layers", "rope_parameters",
)
_RESOURCES = ("model_file", "quanted_pth", "embedding_weight_path", "embedding_scale_path")


def _stamp(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class _OriginalHFHeaders(SafeTensorFiles):
    """Use the original loader's model*.safetensors scan, independent of index.

    Reuse the existing bounded safetensors header decoder and extent checks.
    The original exporter ignores index.json; its exact model-file scan and
    prefix removal therefore take precedence over an index's weight_map.
    """

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.headers, self.header_records, self.mapping, self.cleaned, self.stamps = {}, {}, {}, {}, {}
        self.discovery = {"mode": "ORIGINAL_MODEL_GLOB_HEADERS", "files": []}
        paths = []
        with os.scandir(self.root) as entries:
            for count, entry in enumerate(entries, 1):
                if count > self.MAX_DISCOVERY_ENTRIES:
                    raise ValueError("original HF directory exceeds bounded entry limit")
                if not (entry.name.startswith("model") and entry.name.endswith(".safetensors")):
                    continue
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    raise ValueError("original HF shard must be a regular file: " + entry.name)
                paths.append(Path(entry.path))
                if len(paths) > self.MAX_DISCOVERY_SHARDS:
                    raise ValueError("original HF shard count exceeds bounded limit")
        if not paths:
            raise ValueError("original loader requires model*.safetensors in model_file")
        total = 0
        for path in sorted(paths):
            before = _stamp(path)
            header, start = self._header(path, max_header_bytes=self.MAX_DISCOVERY_HEADER_BYTES - total)
            total += start - 8
            for name in header:
                if name == "__metadata__":
                    continue
                if name.startswith("model.language_model."):
                    clean = name[len("model.language_model."):]
                elif name.startswith("model.visual."):
                    continue
                elif name.startswith("model."):
                    clean = name[len("model."):]
                else:
                    clean = name
                if clean.startswith("mtp."):
                    continue
                if clean in self.cleaned:
                    raise ValueError("ambiguous original HF tensor after prefix removal: " + clean)
                self.cleaned[clean] = name
                self.mapping[name] = path.name
            if _stamp(path) != before:
                raise ValueError("original HF shard changed while reading its header")
            self.stamps[path] = before
        self.discovery["header_bytes_read"] = total

    def unchanged(self):
        if any(_stamp(path) != before for path, before in self.stamps.items()):
            raise ValueError("original HF shard changed since header inspection")


def inspect_target_pytorch_resources(config_yaml):
    """Read only YAML/config, safetensors headers and embedding file sizes."""
    config_yaml = Path(config_yaml).resolve(strict=True)
    if config_yaml.stat().st_size > 1 << 20:
        raise ValueError("original YAML exceeds 1 MiB")
    yaml_sha = sha256_file(config_yaml)
    config = OriginalModelConfig.from_yaml(config_yaml)
    config.validate_paths()
    text_path = config.model_file / "config.json"
    if text_path.stat().st_size > 1 << 20:
        raise ValueError("original config.json exceeds 1 MiB")
    text_sha = sha256_file(text_path)
    text = config.text_config()
    pinned_path = Path(__file__).resolve().parents[2] / "specs/target-config.json"
    pinned = json.loads(pinned_path.read_text())["text_config"]
    differences = [key for key in _CONFIG_FIELDS if text.get(key) != pinned.get(key)]
    if differences or config.kv_cache_max_len != 2048:
        raise ValueError("original target must retain the complete pinned 40-layer configuration: "
                         + ", ".join(differences or ["kv_cache_max_len"]))
    hf = _OriginalHFHeaders(config.model_file)
    quant = OriginalQuantWeights(config.quanted_pth)
    inventory = required_weight_shapes(text)
    hf_required, quant_required = {}, {}
    for key, shape in sorted(inventory.items()):
        # The original loader first copies every non-expert HF parameter,
        # including dense parameters later replaced by the quant loader.
        if ".mlp.experts." not in key:
            clean = key[len("target."):]
            name = hf.cleaned.get(clean)
            if name is None:
                raise ValueError("original loader HF tensor missing: " + clean)
            path, offset, count, dtype = hf.tensor(name, shape)
            hf_required[key] = {"tensor": name, "shard": path.name, "shape": list(shape),
                                "dtype": dtype, "offset": offset,
                                "bytes": count * (4 if dtype == "F32" else 2)}
        try:
            canonicalkeys(key)
        except ValueError:
            continue
        quant_required[key] = quant.describe(key, shape)
    before_embedding = {key: _stamp(getattr(config, key)) for key in _RESOURCES[2:]}
    embedding = _embedding_record(config, text, inspect_only=True)
    hf.unchanged()
    quant._unchanged(quant.shards)
    if any(_stamp(getattr(config, key)) != before for key, before in before_embedding.items()):
        raise ValueError("original embedding resource changed during size inspection")
    if sha256_file(config_yaml) != yaml_sha or sha256_file(text_path) != text_sha:
        raise ValueError("original configuration changed during header inspection")
    return {
        "schema_version": 1, "artifact_kind": "qwen35-target-pytorch-resources",
        "status": "RESOURCE_HEADERS_READY", "resources": {key: str(getattr(config, key)) for key in _RESOURCES},
        "config_yaml": str(config_yaml), "config_yaml_sha256": yaml_sha,
        "text_config_sha256": text_sha, "pinned_config_sha256": sha256_file(pinned_path),
        "target_layers": 40, "vocab_size": 248320, "experts_per_layer": 256,
        "hf_loader_policy": "model*.safetensors_WITH_ORIGINAL_PREFIX_REMOVAL_INDEX_IGNORED",
        "hf_required_nonexpert_tensors": hf_required, "hf_headers": hf.discovery,
        "quant_required_tensors": quant_required,
        "quant_headers": {name: {"bytes": row["stat"][2], "header_sha256": row["header_sha256"]}
                          for name, row in quant.shards.items()},
        "embedding": embedding, "payload_validation": "NOT_READ",
        "vendor_dependencies": "NOT_VERIFIED", "air_export": "NOT_RUN", "device_execution": "NOT_RUN",
        "scope": "Full required header inventory and embedding sizes only; no tensor-value, upstream-identity or export success claim",
    }


def prepare_target_pytorch(source_root, config_yaml, output, *, sequence_length=55):
    """Prepare a new small on-origin source directory; never execute export."""
    out = output_path(output)
    if out.exists():
        raise ValueError("target-pytorch output exists; preserve it and choose a new output")
    out.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "artifact_kind": "qwen35-target-pytorch-preparation",
              "status": "SOURCE_BLOCKED", "export": "EXPORT_NOT_RUN", "air_export": "NOT_RUN",
              "model_execution": "NOT_RUN", "device_execution": "NOT_RUN", "output": str(out)}
    stage = "resources"
    try:
        resources = inspect_target_pytorch_resources(config_yaml)
        result["resources"] = resources
        # Make the copied YAML relocatable without changing resource choices.
        # Relative input paths use OriginalModelConfig's YAML-directory rule.
        yaml_bytes = Path(resources["config_yaml"]).read_bytes()
        if hashlib.sha256(yaml_bytes).hexdigest() != resources["config_yaml_sha256"]:
            raise ValueError("original YAML changed before derived source preparation")
        raw = yaml.safe_load(yaml_bytes)
        for key, value in resources["resources"].items():
            raw[key] = value
        resolved = out / "resolved-original.yaml"
        with resolved.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(raw, stream, sort_keys=False)
        stage = "source"
        derived = prepare_original_target_source(source_root, out / "target-pytorch-source",
                                                  config=resolved, sequence_length=sequence_length)
        script = out / "export.sh"
        text = ("#!/usr/bin/env bash\nset -euo pipefail\n"
                "# Run on the original machine with the validated original resources.\n"
                "# This command performs the full PyTorch/TorchAir export.\n"
                f"exec bash {shlex.quote(derived['export_script'])}\n")
        with script.open("x", encoding="utf-8") as stream:
            stream.write(text)
        script.chmod(0o755)
        result.update(status="SOURCE_READY", source=derived, on_origin_export_script=str(script),
                      payload_validation="NOT_READ", source_arithmetic="ORIGINAL_CPRO_WITH_AUXILIARY_FEATURE_OUTPUT",
                      runtime_abi="ORIGINAL_REF_DATA_PLUS_REAL_FEATURES",
                      scope="Required resources pass header checks; source prepared only, export is a separate explicit on-origin action")
    except Exception as error:
        result.update(stage=stage, error=f"{type(error).__name__}: {error}")
    save_json(out / "result.json", result)
    return result

