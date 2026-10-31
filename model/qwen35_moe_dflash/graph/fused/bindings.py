"""Explicit vendor GE port contracts; schema checks do not prove installation.

A binding records explicit port tables from converter/registration sources or
the user's interface document, with its provenance retained with the export.
This version supports logical ND tensors and fixed, named ports only.
Document-derived bindings retain a pinned source section and explicit provenance;
constructing them does not verify an installed toolkit or kernel.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from qwen35_moe_dflash.common.artifacts import output_path, save_json
from qwen35_moe_dflash.graph.fused.ir import DTYPES
from qwen35_moe_dflash.graph.fused.recipe import _attribute


_ROLES = {
    "gdn": ({"query", "key", "value", "g", "beta", "initial_state", "effective_length"},
            {"core_attn", "final_state"},
            {"chunk_size": "int", "output_final_state": "bool", "use_qk_l2norm_in_kernel": "bool"}),
    "conv_state": ({"projected", "history", "update_control"}, {"history", "padded"}, {}),
    "rms": ({"x", "gamma"}, {"y", "rstd"}, {"epsilon": "float"}),
}
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]*")


def _name(value):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError(f"invalid GE binding name {value!r}")


def _keys(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() or set(value) - set(required) - set(optional):
        raise ValueError(f"binding fields must be {sorted(required)} (optional {sorted(optional)})")


class OperatorBindings:
    """Checked logical-to-GE wiring, independent of PyTorch and the target SDK.

    Dictionary order of ``outputs`` is the registered output ordinal order.
    Dtypes and optional static shape constraints are checked before insertion.
    ``source`` identifies the caller's schema evidence; this class does not
    assert that the source is installed or valid for the target firmware.
    """
    def __init__(self, document):
        document = deepcopy(document)
        _keys(document, {"schema_version", "target", "ddk_id", "source", "operators"})
        if type(document["schema_version"]) is not int or document["schema_version"] != 1:
            raise ValueError("GE binding schema_version must be 1")
        for key in ("target", "ddk_id"):
            if not isinstance(document[key], str) or not document[key].strip():
                raise ValueError(f"GE binding {key} must identify the selected toolchain")
        _keys(document["source"], {"path", "sha256"})
        if (not isinstance(document["source"]["path"], str) or not document["source"]["path"].strip()
                or not isinstance(document["source"]["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", document["source"]["sha256"])):
            raise ValueError("binding source requires path and lowercase SHA256")
        operators = document["operators"]
        if not isinstance(operators, dict) or not operators or set(operators) - _ROLES.keys():
            raise ValueError("binding operators must select gdn, conv_state and/or rms")
        for role, spec in operators.items():
            _keys(spec, {"op_type", "layout", "inputs", "outputs", "attrs"},
                  {"static_attrs", "history_output_width", "history_keep"})
            _name(spec["op_type"])
            if spec["layout"] != "ND":
                raise ValueError("this GE path requires ND bindings; NZ/paged adapters are not implemented")
            input_names, output_names, logical_attrs = _ROLES[role]
            for direction, expected in (("inputs", input_names), ("outputs", output_names)):
                ports = spec[direction]
                if not isinstance(ports, dict) or set(ports) != expected:
                    raise ValueError(f"{role} {direction} must bind exactly {sorted(expected)}")
                seen = set()
                for logical, desc in ports.items():
                    _keys(desc, {"port", "dtype"}, {"shape"})
                    _name(desc["port"])
                    if desc["port"] in seen:
                        raise ValueError(f"duplicate {role} {direction} port")
                    seen.add(desc["port"])
                    if not isinstance(desc["dtype"], str) or desc["dtype"] not in DTYPES:
                        raise ValueError(f"unsupported dtype at {role}.{logical}")
                    if "shape" in desc:
                        if (not isinstance(desc["shape"], list) or len(desc["shape"]) > 16
                                or any(x is not None and (type(x) is not int or x < 1) for x in desc["shape"])):
                            raise ValueError("binding shape constraints require positive dimensions or null")
            attrs = spec["attrs"]
            if not isinstance(attrs, dict) or set(attrs) != set(logical_attrs):
                raise ValueError(f"{role} attributes must bind exactly {sorted(logical_attrs)}")
            actual_names = set()
            for logical, desc in attrs.items():
                _keys(desc, {"name", "type"})
                _name(desc["name"])
                # Vendor flags may be BOOL or INT. Integer flag encoding is
                # permitted only when explicitly declared in this contract.
                expected = logical_attrs[logical]
                allowed = {expected, "int"} if expected == "bool" else {expected}
                if not isinstance(desc["type"], str) or desc["type"] not in allowed or desc["name"] in actual_names:
                    raise ValueError(f"invalid or duplicate {role} attribute binding")
                actual_names.add(desc["name"])
            static = spec.get("static_attrs", {})
            if not isinstance(static, dict) or actual_names & static.keys():
                raise ValueError("static attributes cannot override bound attributes")
            for key, value in static.items():
                _name(key)
                _keys(value, {"type", "value"})
                _attribute(value)
            if role == "conv_state":
                if (spec.get("history_output_width") not in {"kernel", "kernel_minus_one"}
                        or spec.get("history_keep") not in {"last_kernel_minus_one", "padded_tail"}):
                    raise ValueError("conv_state requires explicit raw history width and last K-1 selection")
            elif "history_output_width" in spec or "history_keep" in spec:
                raise ValueError("history metadata is only valid for conv_state")
        self._document = document
        self.sha256 = None

    @classmethod
    def from_file(cls, path):
        payload = Path(path).read_bytes()
        if len(payload) > 1024 * 1024:
            raise ValueError("GE binding JSON exceeds 1 MiB")

        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate binding JSON key {key!r}")
                result[key] = value
            return result

        result = cls(json.loads(payload, object_pairs_hook=unique_pairs))
        result.sha256 = hashlib.sha256(payload).hexdigest()
        return result

    def manifest(self):
        return deepcopy(self._document)

    def require(self, roles):
        missing = set(roles) - self._document["operators"].keys()
        if missing:
            raise ValueError(f"missing enabled GE operator bindings: {sorted(missing)}")

    def _spec(self, role):
        self.require([role])
        return self._document["operators"][role]

    def tensor_dtype(self, role, direction, logical):
        if direction not in {"inputs", "outputs"}:
            raise ValueError("binding tensor direction must be inputs or outputs")
        return self._spec(role)[direction][logical]["dtype"]

    @property
    def conv_history_slice(self):
        return self._spec("conv_state")["history_keep"]

    def conv_history_size(self, kernel):
        return kernel if self._spec("conv_state")["history_output_width"] == "kernel" else kernel - 1

    @staticmethod
    def _tensor_contract(desc, shape, dtype, label):
        if dtype != desc["dtype"]:
            raise ValueError(f"{label} dtype {dtype} disagrees with binding {desc['dtype']}; no implicit cast")
        constraint = desc.get("shape")
        if constraint is not None and (len(shape) != len(constraint) or any(
                expected is not None and actual != expected for actual, expected in zip(shape, constraint))):
            raise ValueError(f"{label} shape {tuple(shape)} disagrees with binding {constraint}")

    def call(self, graph, role, *, inputs, outputs, attrs, name=None):
        spec = self._spec(role)
        if set(inputs) != set(spec["inputs"]) or set(outputs) != set(spec["outputs"]) or set(attrs) != set(spec["attrs"]):
            raise ValueError(f"{role} call does not match the complete logical contract")
        actual_inputs, actual_outputs = {}, {}
        for logical, desc in spec["inputs"].items():
            value = inputs[logical]
            self._tensor_contract(desc, value.shape, value.dtype, f"{role}.{logical}")
            actual_inputs[desc["port"]] = value
        for logical, desc in spec["outputs"].items():
            shape, dtype = outputs[logical]
            self._tensor_contract(desc, shape, dtype, f"{role}.{logical}")
            actual_outputs[desc["port"]] = (shape, dtype)
        actual_attrs = deepcopy(spec.get("static_attrs", {}))
        for logical, desc in spec["attrs"].items():
            value = attrs[logical]
            expected = _ROLES[role][2][logical]
            if ((expected == "bool" and type(value) is not bool)
                    or (expected == "int" and type(value) is not int)
                    or (expected == "float" and type(value) not in (float, int))):
                raise ValueError(f"invalid logical attribute {role}.{logical}")
            if expected == "bool" and desc["type"] == "int":
                value = int(value)
            actual_attrs[desc["name"]] = {"type": desc["type"], "value": value}
        result = graph.custom(spec["op_type"], actual_inputs, actual_outputs, actual_attrs, name)
        return {logical: result[desc["port"]] for logical, desc in spec["outputs"].items()}


_DOCUMENT_RELATIVE = "docs/reference/qwen35-moe-operators-and-quantization-analysis.md"
_DOCUMENT_PATH = Path(__file__).resolve().parents[4] / _DOCUMENT_RELATIVE
_PORTS_SECTION_SHA256 = "40b972acf196d09b43124d42ddb15afd749c132af2384ba41e33b8afa708592e"


def _source_record():
    payload = _DOCUMENT_PATH.read_bytes()
    start = payload.find("### 4.6 ".encode())
    end = payload.find("## 5. ".encode(), start + 1)
    if start < 0 or end < 0 or hashlib.sha256(payload[start:end]).hexdigest() != _PORTS_SECTION_SHA256:
        raise ValueError("GE port document section changed or missing; review mappings before generating bindings")
    return {"path": _DOCUMENT_RELATIVE, "sha256": hashlib.sha256(payload).hexdigest()}


def _port(name, dtype, shape=None):
    return {"port": name, "dtype": dtype, **({"shape": shape} if shape is not None else {})}


def document_probe_spec(role):
    """NZ single-op diagnostics only; these do not enable production fusion.

    The legacy ND binding file stays unchanged. This independent contract is
    pinned to the same user document and is stored with each new probe ABI.
    """
    from qwen35_moe_dflash.graph.fused.ir import ATTENTION_INPUT_SCHEMA
    source = _source_record()
    cache = [64, 32, 32, 16]
    if role == "cache":
        return {"op_type": "CacheUpdate", "layout": "NZ", "source": source,
                "inputs": {"x": _port("x", "float16", cache),
                           "updates": _port("updates", "float16", [None, 32, 16]),
                           "targetBlock": _port("targetBlock", "int32", [1]),
                           "offsetInBlock": _port("offsetInBlock", "int32", [])},
                "outputs": {"x": _port("x", "float16", cache)}, "attrs": {}}
    if role != "attention":
        raise ValueError("document probe role must be cache or attention")
    return {"op_type": "AdnFusedInferAttention", "layout": "NZ", "source": source,
            "inputs": {"query": _port("query", "float16", [1, 256, None, 16]),
                       "key0": _port("key0", "float16", cache),
                       "value0": _port("value0", "float16", cache),
                       "atten_mask": _port("atten_mask", "float16", [1, 1, None, 2048]),
                       "actual_seq_lengths_q": _port("actual_seq_lengths_q", "int64", [1]),
                       "actual_seq_lengths_kv": _port("actual_seq_lengths_kv", "int64", [1]),
                       "block_table": _port("block_table", "int32", [1, 64]),
                       "all_seq_lengths_q": _port("all_seq_lengths_q", "int64", [1])},
            "outputs": {"attention_out": _port("attention_out", "float16", [1, 256, None, 16])},
            "attrs": {"num_heads": {"type": "int", "value": 16},
                      "num_key_value_heads": {"type": "int", "value": 2},
                      "block_size": {"type": "int", "value": 32},
                      "scale_value": {"type": "float", "value": 256 ** -.5},
                      "input_layout": {"type": "string", "value": "BNSD"},
                      "inner_precise": {"type": "int", "value": 2}},
            "ge_input_schema": [{"index": index, "name": port, "kind": kind,
                                 **({"count": 1} if kind == "dynamic" else {})}
                                for index, (port, kind) in enumerate(ATTENTION_INPUT_SCHEMA)],
            "scope": "Document-derived one-element KV lists; optional slots retained. Target registration, all_seq_lengths_q semantics and NZ kernel compatibility are unverified."}


def document_bindings(*, target, ddk_id, gdn_initial_state_dtype):
    """Construct the documented three-op contract with an explicit state choice.

    The user document reports FP32 initial state for some variants and FP16 at
    the Cpro caller. This argument deliberately has no implicit default.
    """
    if gdn_initial_state_dtype not in {"float16", "float32"}:
        raise ValueError("gdn_initial_state_dtype must explicitly select float16 or float32")
    source = _source_record()
    document = {
        "schema_version": 1, "target": target, "ddk_id": ddk_id, "source": source,
        "operators": {
            "gdn": {
                "op_type": "ChunkGatedDeltaRule", "layout": "ND",
                "inputs": {
                    "query": _port("query", "float16", [1, None, 32, 128]),
                    "key": _port("key", "float16", [1, None, 32, 128]),
                    "value": _port("value", "float16", [1, None, 32, 128]),
                    "g": _port("g", "float16", [1, None, 32]),
                    "beta": _port("beta", "float16", [1, None, 32]),
                    "initial_state": _port("initial_state", gdn_initial_state_dtype, [1, 32, 128, 128]),
                    "effective_length": _port("effective_length", "int16", [1]),
                },
                "outputs": {
                    "core_attn": _port("core_attn", "float16", [1, None, 32, 128]),
                    "final_state": _port("last_recurrent_state", "float32", [1, 32, 128, 128]),
                },
                "attrs": {key: {"name": key, "type": "int"} for key in
                          ("chunk_size", "output_final_state", "use_qk_l2norm_in_kernel")},
            },
            "conv_state": {
                "op_type": "LinearAttentionConvstateUpdate", "layout": "ND",
                "inputs": {
                    "projected": _port("x1", "float16", [1, None, 8192]),
                    "history": _port("x2", "float16", [1, 8192, 3]),
                    "update_control": _port("x3", "int32", [128]),
                },
                "outputs": {
                    "history": _port("y1", "float16", [1, 8192, 4]),
                    "padded": _port("y2", "float16", [1, 8192, None]),
                },
                "attrs": {}, "history_output_width": "kernel", "history_keep": "padded_tail",
            },
            "rms": {
                "op_type": "AdnRmsNorm", "layout": "ND",
                "inputs": {"x": _port("self", "float16"), "gamma": _port("gamma", "float16")},
                "outputs": {"y": _port("y", "float16"), "rstd": _port("rstd", "float32")},
                "attrs": {"epsilon": {"name": "epsilon", "type": "float"}},
            },
        },
    }
    # Use the production validator, including complete logical port sets and
    # the selected convolution history policy, before writing any artifact.
    OperatorBindings(document)
    return document


def _installed_fp16_state_evidence(source):
    """Recognize the observed direct half-buffer ABI; never compile/import source.

    This bounded lexical check is deliberately narrower than a C++ parser. It
    does not evaluate preprocessing, trace aliases or prove the active kernel.
    Unknown or contradictory state-buffer expressions require manual review.
    """
    path = Path(source).resolve()
    if path.name != "chunk_gated_delta_rule.cpp" or not path.is_file():
        raise ValueError("gdn_state_source must be an existing chunk_gated_delta_rule.cpp")
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("GDN state source exceeds 8 MiB")
    payload = path.read_bytes()
    if len(payload) > 8 * 1024 * 1024:
        raise ValueError("GDN state source exceeds 8 MiB")
    text = payload.decode("utf-8")
    # Preserve offsets/line numbers while excluding commented-out examples and
    # strings containing code-looking text from the source evidence.
    ignored = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
    code = ignored.sub(lambda match: re.sub(r"[^\n]", " ", match.group()), text)
    declarations = {}
    for match in re.finditer(r"\bGlobalTensor\s*<\s*([^<>]+?)\s*>\s*(\w+)\s*[;=]", code):
        declarations.setdefault(match.group(2), set()).add(match.group(1).strip())
    states = {"initial_state": [], "last_recurrent_state": []}
    for match in re.finditer(r"\b(\w+)\s*\.\s*SetGlobalBuffer\s*\(([^;]*?)\)\s*;", code):
        receiver, arguments = match.group(1), match.group(2)
        for name, records in states.items():
            if not re.search(r"\b" + name + r"\b", arguments):
                continue
            direct = re.fullmatch(r"\s*reinterpret_cast\s*<\s*__gm__\s+half\s*\*\s*>"
                                  r"\s*\(\s*" + name + r"\s*\)\s*(?:,.*)?", arguments, re.S)
            if declarations.get(receiver) != {"half"} or direct is None:
                raise ValueError(f"GDN {name} buffer has contradictory or unrecognized dtype/address evidence")
            records.append({"line": code.count("\n", 0, match.start()) + 1,
                            "receiver": receiver, "statement": text[match.start():match.end()]})
    if any(not records for records in states.values()):
        raise ValueError("GDN source must show both initial_state and last_recurrent_state direct half buffers")
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
            "state_dtype": "float16", "matches": states,
            "verification_kind": "BOUNDED_SOURCE_PATTERN",
            "scope": "Observed source buffer ABI only; preprocessing, compiled binary and runtime are unverified."}


def write_document_bindings(output, *, target, ddk_id, gdn_initial_state_dtype, gdn_state_source=None):
    """Write a new binding JSON and explicit provenance sidecar under this run."""
    output = output_path(output)
    metadata = output_path(output.with_suffix(".metadata.json"))
    evidence_path = output_path(output.with_suffix(".sources.json")) if gdn_state_source is not None else None
    if output.exists() or metadata.exists() or evidence_path is not None and evidence_path.exists():
        raise FileExistsError("binding or provenance output already exists; use a new path")
    document = document_bindings(target=target, ddk_id=ddk_id,
                                 gdn_initial_state_dtype=gdn_initial_state_dtype)
    document_source = deepcopy(document["source"])
    evidence = None
    if gdn_state_source is not None:
        if gdn_initial_state_dtype != "float16":
            raise ValueError("installed FP16 state override requires gdn_initial_state_dtype=float16")
        evidence = _installed_fp16_state_evidence(gdn_state_source)
        sources = {"schema_version": 1, "artifact_kind": "qwen35-ge-binding-source-evidence",
                   "document": document_source, "source_section_sha256": _PORTS_SECTION_SHA256,
                   "installed_gdn_state": evidence,
                   "override": {"gdn.outputs.final_state.dtype": {"document": "float32", "selected": "float16"}},
                   "compile_validation": "NOT_RUN", "device_validation": "NOT_RUN"}
        save_json(evidence_path, sources)
        document["source"] = {"path": str(evidence_path), "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest()}
        document["operators"]["gdn"]["outputs"]["final_state"]["dtype"] = "float16"
        OperatorBindings(document)
    result = {
        "status": "BINDINGS_FROM_USER_DOCUMENT", "output": str(output),
        "metadata": str(metadata), "source": document["source"],
        "source_section": "4.6 GE 端口信息速查", "source_section_sha256": _PORTS_SECTION_SHA256,
        "source_platform": "Ascend310P3", "requested_target": target, "declared_ddk_id": ddk_id,
        "gdn_initial_state_dtype": gdn_initial_state_dtype,
        "gdn_final_state_dtype": document["operators"]["gdn"]["outputs"]["final_state"]["dtype"],
        "gdn_initial_state_choice": ("documented Cpro caller" if gdn_initial_state_dtype == "float16"
                                     else "documented FP32 interface/other variants"),
        "conv_history_policy": "last K-1 entries of documented y2 history/current concatenation; y1 chronology not assumed",
        "kernel_installation": "UNVERIFIED", "target_compatibility": "UNVERIFIED",
        "device_validation": "NOT_RUN", "schema_origin": "USER_PROVIDED_ANALYSIS",
        "scope": "Port/attribute transcription only; target and DDK labels are declarations, not installation checks.",
        "shape_scope": "Documented full-model GDN/conv dimensions; tiny reduced heads/channels are rejected.",
    }
    if evidence is not None:
        result.update(status="BINDINGS_WITH_INSTALLED_GDN_STATE", schema_origin="USER_DOCUMENT_WITH_SOURCE_STATE_OVERRIDE",
                      document_source=document_source, installed_gdn_state=evidence,
                      gdn_initial_state_choice="explicit installed-source FP16 input/output state ABI",
                      scope="Document ports plus hash-bound observed FP16 GDN state buffers; compiled/runtime ABI unverified.")
    save_json(output, document)
    result["binding_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    save_json(metadata, result)
    return result
