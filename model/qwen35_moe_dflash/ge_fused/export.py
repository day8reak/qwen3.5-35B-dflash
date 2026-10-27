"""Independent stateful target GE source and AIR export; no Torch dependency.

The existing prepared FP16 payload is read and hash checked. This target has a
new explicit-state ABI; its artifacts cannot be handed to the legacy pair
compiler or two-input HIAI runner. Operator bindings describe the selected
installation, but do not establish that the installation supports this graph.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
import struct

from ..artifacts import checked_file, files_under, output_path, save_json
from ..config import Qwen35DFlashConfig
from ..ge_export import _expert_execution, _logged, weight_identity
from ..hashing import sha256_file


FUSION_ROLES = frozenset({"gdn", "conv_state", "rms"})


def selected_fusion(value):
    """Parse the explicit subset; no implicit 'all' or fallback semantics."""
    if isinstance(value, str):
        roles = [] if value == "none" else value.split(",")
    elif isinstance(value, (tuple, list, set, frozenset)):
        roles = list(value)
    else:
        raise ValueError("fused_ops must be none or a subset of gdn,conv_state,rms")
    if (any(not isinstance(role, str) or role not in FUSION_ROLES for role in roles)
            or len(set(roles)) != len(roles)):
        raise ValueError("fused_ops must contain unique roles from gdn,conv_state,rms")
    return tuple(sorted(roles))


def _prepared_graph(prepared_dir, *, sequence_length, past_length, fused_ops,
                    bindings_path, recurrent_dtype, expert_dispatch, gdn_indexed_copy,
                    gdn_state_casts):
    from .target import build_target

    roles = selected_fusion(fused_ops)
    if roles and bindings_path is None:
        raise ValueError("enabled fused_ops require --bindings with the actual GE operator contract")
    if not roles and bindings_path is not None:
        raise ValueError("--bindings supplied with no fused_ops selected")
    bindings = None
    binding_file = None
    if bindings_path is not None:
        from .bindings import OperatorBindings
        binding_file = Path(bindings_path).resolve()
        bindings = OperatorBindings.from_file(binding_file)
        bindings.require(roles)

    prepared = Path(prepared_dir).resolve()
    manifest_path = prepared / "prepared.json"
    manifest_sha = sha256_file(manifest_path)
    plan = json.loads(manifest_path.read_text())
    if plan.get("artifact_kind") != "qwen35-ge-prepared" or plan.get("status") != "PREPARED":
        raise ValueError("expected a successful direct GE preparation")
    if plan.get("precision") != "float16" or plan.get("quantization") is not None:
        raise ValueError("stateful GE export currently requires unquantized FP16 prepared weights")
    prepared_dispatch = plan.get("expert_dispatch", "static")
    if prepared_dispatch not in {"static", "selected"}:
        raise ValueError("invalid prepared expert_dispatch")
    if ("expert_execution" in plan and plan["expert_execution"] !=
            _expert_execution(plan["target_config"], prepared_dispatch)):
        raise ValueError("prepared expert execution metadata differs from configuration")
    dispatch = prepared_dispatch if expert_dispatch is None else expert_dispatch
    sequence_length = plan["sequence_length"] if sequence_length is None else sequence_length
    graph = build_target(plan["target_config"], Qwen35DFlashConfig.from_dict(plan["draft_config"]),
                         sequence_length, past_length=past_length, fused_ops=roles,
                         bindings=bindings, recurrent_dtype=recurrent_dtype,
                         expert_dispatch=dispatch, gdn_indexed_copy=gdn_indexed_copy,
                         gdn_state_casts=gdn_state_casts)
    records = plan["weights"]
    if plan["source"].get("fp16_payload_sha256") != weight_identity(records):
        raise ValueError("prepared source identity differs from FP16 weight records")
    sources = {}
    # Verify the full source payload, including its retained draft weights. Only
    # target weights are copied into this target-only artifact.
    for key, item in records.items():
        rel = Path(item["path"])
        if rel.is_absolute() or len(rel.parts) != 2 or rel.parts[0] != "weights" or ".." in rel.parts:
            raise ValueError(f"invalid prepared weight path: {key}")
        shape = item["shape"]
        if (not isinstance(shape, list) or not shape
                or any(type(dim) is not int or dim <= 0 for dim in shape)
                or item["dtype"] != "float16" or item["bytes"] != math.prod(shape) * 2):
            raise ValueError(f"invalid prepared weight descriptor: {key}")
        sources[key] = checked_file(prepared, item)
    for key, value in graph.weights.items():
        if key not in records or tuple(records[key]["shape"]) != value.shape or value.dtype != "float16":
            raise ValueError(f"prepared weight does not match stateful target: {key}")
    if type(plan.get("external_weights")) is not bool:
        raise ValueError("prepared external_weights must be boolean")
    used = {key: records[key] for key in graph.weights}
    if not plan["external_weights"] and sum(item["bytes"] for item in used.values()) > (1800 << 20):
        raise ValueError("embedded target weights exceed AIR protobuf budget")
    return graph, plan, prepared, manifest_path, manifest_sha, sources, used, bindings, binding_file, dispatch


def _toolchain(ge_root, cxx, cxx11_abi, timeout_seconds):
    if type(cxx11_abi) is not int or cxx11_abi not in (0, 1) or timeout_seconds <= 0:
        raise ValueError("invalid C++ ABI or timeout")
    ge = Path(ge_root).expanduser().resolve()
    include, lib = ge / "include", ge / "lib64"
    for path in (include / "graph/graph.h", include / "graph/operator.h", lib / "libgraph.so"):
        if not path.is_file():
            raise FileNotFoundError(f"matching real GE headers/libraries required: {path}")
    compiler = shutil.which(cxx)
    if compiler is None:
        raise FileNotFoundError(f"C++ compiler not found: {cxx}")
    return ge, include, lib, compiler


def _artifact(prepared_dir, output_dir, *, ge_root=None, sequence_length=None,
              past_length=0, fused_ops=(), bindings_path=None, recurrent_dtype="float16",
              expert_dispatch=None, fp16_reductions="native", fp16_row_mul="native",
              gdn_indexed_copy="native", gdn_state_casts="reject",
              cxx="c++", cxx11_abi=0, timeout_seconds=3600):
    from .emitter import emit_cpp

    if fp16_reductions not in {"native", "matmul"}:
        raise ValueError("fp16_reductions must be native or matmul")
    if fp16_row_mul not in {"native", "compact"}:
        raise ValueError("fp16_row_mul must be native or compact")
    if gdn_indexed_copy not in {"native", "gather"}:
        raise ValueError("gdn_indexed_copy must be native or gather")
    if gdn_state_casts not in {"reject", "allow"}:
        raise ValueError("gdn_state_casts must be reject or allow")
    (graph, plan, prepared, prepared_manifest, prepared_sha, sources, used,
     bindings, binding_file, dispatch) = _prepared_graph(
        prepared_dir, sequence_length=sequence_length, past_length=past_length,
        fused_ops=fused_ops, bindings_path=bindings_path, recurrent_dtype=recurrent_dtype,
        expert_dispatch=expert_dispatch, gdn_indexed_copy=gdn_indexed_copy,
        gdn_state_casts=gdn_state_casts)
    chain = None if ge_root is None else _toolchain(ge_root, cxx, cxx11_abi, timeout_seconds)
    binding_sha = None if bindings is None else bindings.sha256
    root = output_path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    stage = "copy-weights"
    try:
        (root / "weights").mkdir()
        paths = {}
        for key, item in used.items():
            destination = root / item["path"]
            try:
                os.link(sources[key], destination)
            except OSError:
                shutil.copyfile(sources[key], destination)
            paths[key] = str(destination)
        binding_record = None
        if binding_file is not None:
            shutil.copyfile(binding_file, root / "bindings.json")
            binding_record = {"path": "bindings.json", "bytes": (root / "bindings.json").stat().st_size,
                              "sha256": binding_sha}
            checked_file(root, binding_record)
        folder = root / "target"
        folder.mkdir()
        source = folder / "target.cpp"
        stage = "target-lowering"
        reductions, row_mul = {}, {}
        source.write_text(emit_cpp(graph, paths, external_weights=plan["external_weights"],
                                   recipe_dir=folder / "recipe", fp16_reductions=fp16_reductions,
                                   reduction_report=reductions, fp16_row_mul=fp16_row_mul,
                                   row_mul_report=row_mul))
        save_json(folder / "reduction-lowering.json", reductions)
        save_json(folder / "row-mul-lowering.json", row_mul)
        save_json(folder / "gdn-copy-lowering.json", graph.gdn_copy_report)
        save_json(folder / "graph.json", graph.manifest())
        save_json(folder / "state-abi.json", graph.state_abi)
        construction = files_under(folder, root)
        air_record = None
        if chain is not None:
            ge, include, lib, compiler = chain
            env = dict(os.environ)
            env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, (str(lib), env.get("LD_LIBRARY_PATH", ""))))
            binary, air = folder / "target_exporter", folder / "target.air"
            command = [compiler, "-std=c++17", f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}",
                       "-I", str(include), str(source), "-L", str(lib), f"-Wl,-rpath,{lib}", "-lgraph"]
            if (lib / "libgraph_base.so").is_file():
                command.append("-lgraph_base")
            command.extend(["-o", str(binary)])
            save_json(folder / "build-command.json", {"argv": command})
            stage = "target-build"
            _logged(command, folder=folder, env=env, log=folder / "build.log", timeout=timeout_seconds)
            if not binary.is_file() or binary.is_symlink() or not binary.stat().st_size:
                raise RuntimeError("compiler did not produce a nonempty regular exporter")
            command = [str(binary), str(air)]
            save_json(folder / "export-command.json", {"argv": command})
            stage = "target-serialize"
            _logged(command, folder=folder, env=env, log=folder / "export.log", timeout=timeout_seconds)
            if not air.is_file() or air.is_symlink() or not air.stat().st_size:
                raise RuntimeError("exporter did not produce a nonempty regular AIR")
            air_record = {"path": str(air.relative_to(root)), "bytes": air.stat().st_size,
                          "sha256": sha256_file(air)}
        stage = "verify-artifacts"
        for item in construction:
            checked_file(root, item)
        for key, item in used.items():
            checked_file(prepared, item)
            checked_file(root, item)
        if sha256_file(prepared_manifest) != prepared_sha:
            raise ValueError("prepared manifest changed during export")
        if binding_file is not None:
            if sha256_file(binding_file) != binding_sha:
                raise ValueError("operator bindings changed during export")
            checked_file(root, binding_record)
        result = {
            "schema_version": 1,
            "artifact_kind": "qwen35-stateful-ge-air" if chain else "qwen35-stateful-ge-source",
            "status": "EXPORTED" if chain else "SOURCE_READY", "export_backend": "direct-ge",
            "export_root": str(root),
            "pipeline": ["python-numpy-graph", "ge-cpp", "air"], "model_role": "target",
            "state_policy": "explicit", "state_abi": graph.state_abi,
            "legacy_hiai_abi_compatible": False, "fused_ops": list(selected_fusion(fused_ops)),
            "bindings": binding_record, "binding_contract": None if bindings is None else bindings.manifest(),
            "sequence_length": graph.state_abi["sequence_length"], "past_length": past_length,
            "recurrent_dtype": recurrent_dtype, "feature_dtype": "float16", "quantization": None,
            "precision": "float16", "expert_dispatch": dispatch,
            "fp16_reductions": fp16_reductions, "fp16_row_mul": fp16_row_mul,
            "gdn_indexed_copy": gdn_indexed_copy,
            "gdn_state_casts": gdn_state_casts,
            "source": plan["source"], "prepared_manifest_sha256": prepared_sha,
            "target_fp16_payload_sha256": weight_identity(used), "weights": used,
            "external_weights": plan["external_weights"],
            "graph": graph.manifest(), "construction_files": construction, "air": air_record,
            "device_validation": "PENDING", "kernel_registration": "UNVERIFIED",
            "validation_scope": ("GE graph serialization only; no OMG or board execution" if chain else
                                 "Host graph/source generation only; no GE library, OMG or board execution"),
            "limitations": ["New state ABI requires a matching native runner and scheduler.",
                            "Enabled bindings do not prove installed kernel support or numerical agreement.",
                            "Full attention and KV append use primitive GE nodes.",
                            "Packed W4A8 weights and complete DFlash integration are not implemented here."]}
        if chain is not None:
            result.update(ge_root=str(chain[0]), libgraph_sha256=sha256_file(chain[2] / "libgraph.so"),
                          cxx11_abi=cxx11_abi)
        save_json(root / "manifest.json", result)
        return result
    except Exception as error:
        save_json(root / "failure.json", {"status": "FAILED", "stage": stage,
                                          "device_validation": "PENDING", "error": str(error)})
        raise


def plan_target(prepared_dir, output_dir, **options):
    """Generate checked construction artifacts without loading GE libraries."""
    if "ge_root" in options:
        raise ValueError("plan_target does not accept a GE toolchain")
    return _artifact(prepared_dir, output_dir, **options)


def export_target(prepared_dir, output_dir, ge_root, **options):
    """Compile the generated host serializer and write AIR with explicit GE."""
    if ge_root is None:
        raise ValueError("export_target requires an explicit GE root")
    return _artifact(prepared_dir, output_dir, ge_root=ge_root, **options)


_STEP_SOURCE_FIELDS = (
    "model_role", "state_policy", "state_abi", "legacy_hiai_abi_compatible", "fused_ops",
    "binding_contract", "sequence_length", "past_length", "recurrent_dtype", "feature_dtype",
    "precision", "quantization", "expert_dispatch", "fp16_reductions", "fp16_row_mul",
    "gdn_indexed_copy", "gdn_state_casts", "source", "prepared_manifest_sha256",
    "target_fp16_payload_sha256", "external_weights", "graph")
_STEP_AIR_FIELDS = ("air", "construction_files", "weights", "bindings", "export_root",
                    "ge_root", "libgraph_sha256", "cxx11_abi")


def _step_file(root, record):
    from ..compile import _stateful_file
    path = record.get("path", "") if isinstance(record, dict) else ""
    if (not re.fullmatch(r"[A-Za-z0-9_./-]+", path)
            or any(part in {"", ".", ".."} for part in path.split("/"))):
        raise ValueError("step payload requires a canonical ASCII relative path")
    return _stateful_file(root, record)


def _step_record(path, root):
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def _step_contract(omc, air):
    """Verify the compile manifest against its hash-bound AIR manifest.

    Compilation checked the full construction and weight payload. Step staging
    retains that identity without rereading all original weights on every token.
    These are integrity records, not signatures or device correctness evidence.
    """
    from ..compile import _check_stateful_abi, _same_ordered_json
    expected = {"schema_version": 1, "artifact_kind": "qwen35-stateful-omc", "status": "COMPILED",
                "runtime": "hiai", "platform": "kirinx90", "model_role": "target",
                "state_policy": "explicit", "legacy_hiai_abi_compatible": False,
                "precision": "float16", "quantization": None}
    if (any(key not in omc or omc[key] != value for key, value in expected.items())
            or type(omc.get("schema_version")) is not int
            or omc.get("legacy_hiai_abi_compatible") is not False):
        raise ValueError("step preparation requires a COMPILED qwen35-stateful-omc target")
    if (air.get("artifact_kind") != "qwen35-stateful-ge-air" or air.get("status") != "EXPORTED"
            or any(key not in air or not _same_ordered_json(omc.get(key), air[key])
                   for key in _STEP_SOURCE_FIELDS)
            or not _same_ordered_json(omc.get("air_provenance"), {key: air.get(key) for key in _STEP_AIR_FIELDS})):
        raise ValueError("OMC contract differs from its bound AIR manifest")
    for field in ("air_manifest_sha256", "omg_sha256", "prepared_manifest_sha256", "target_fp16_payload_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(omc.get(field, ""))):
            raise ValueError(f"OMC lacks {field}")
    if not isinstance(omc.get("ddk_id"), str) or not omc["ddk_id"].strip():
        raise ValueError("OMC lacks DDK identity")
    _check_stateful_abi(omc)
    graph, abi = omc["graph"], omc["state_abi"]
    if not _same_ordered_json(graph.get("state_abi"), abi):
        raise ValueError("graph and top-level state ABI differ")
    compiled = omc.get("graphs")
    if not isinstance(compiled, list) or len(compiled) != 1:
        raise ValueError("stateful step requires exactly one compiled target")
    compiled = compiled[0]
    if (compiled.get("name") != "target" or compiled.get("graph_name") != graph.get("name")
            or any(not _same_ordered_json(compiled.get(key), value) for key, value in
                   (("inputs", graph["inputs"]), ("outputs", graph["outputs"]), ("state_abi", abi)))):
        raise ValueError("compiled target boundary differs from state ABI")
    # Check complete, ordered layer pairs, including the KV time dimension.
    states = abi["states"]
    if len(states) % 2:
        raise ValueError("state ABI requires complete layer pairs")
    stable_states = []
    for start in range(0, len(states), 2):
        pair, layer = states[start:start + 2], start // 2
        kind = pair[0].get("layer_type")
        roles = {"linear_attention": ("conv_history", "recurrent"), "full_attention": ("key", "value")}.get(kind)
        if roles is None:
            raise ValueError("unknown state layer type")
        for state, role in zip(pair, roles):
            name = f"layer_{layer}_{role}"
            if (type(state.get("layer")) is not int or state["layer"] != layer
                    or state.get("layer_type") != kind or state.get("role") != role
                    or state.get("output") != name + "_out"
                    or state.get("input") != (name if omc["past_length"] else None)):
                raise ValueError("state layer, role or port order mismatch")
            shape = list(state["output_shape"])
            if role in {"key", "value"}:
                if len(shape) != 4 or shape[2] != abi["next_past_length"]:
                    raise ValueError("KV state length differs from next_past_length")
                shape[2] = omc["past_length"]
            elif len(shape) != (3 if role == "conv_history" else 4):
                raise ValueError("invalid recurrent or convolution state rank")
            if omc["past_length"] and state["input_shape"] != shape:
                raise ValueError("incoming state shape differs from current past length")
            if role in {"key", "value"}:
                shape[2] = "past_length"
            stable_states.append({"layer": layer, "layer_type": kind, "role": role,
                                  "shape": shape, "dtype": state["dtype"]})
    identity = {key: omc[key] for key in _STEP_SOURCE_FIELDS
                if key not in {"sequence_length", "past_length", "graph", "state_abi"}}
    identity.update(ddk_id=omc["ddk_id"], omg_sha256=omc["omg_sha256"],
                    weights=graph["weights"], state_descriptors=stable_states,
                    feature_width=graph["outputs"][1]["shape"][-1],
                    abi_policy={key: value for key, value in abi.items() if key not in
                                {"sequence_length", "past_length", "next_past_length", "states", "input_order", "output_order"}},
                    ge_toolchain={key: air[key] for key in ("libgraph_sha256", "cxx11_abi")})
    return compiled, identity


def _step_tsv(model, abi, inputs, outputs, sequence, past):
    rows = ["QWEN35_STATEFUL_STEP\t1", f"model\t{model['path']}\t{model['sha256']}",
            f"state_abi\t{abi['path']}\t{abi['sha256']}", f"tokens\t{sequence}\t{past}"]
    for index, item in enumerate(inputs):
        rows.append("\t".join(("input", str(index), item["name"], item["dtype"],
                               ",".join(map(str, item["shape"])), item["path"], item["sha256"])))
    for index, item in enumerate(outputs):
        rows.append("\t".join(("output", str(index), item["name"], item["dtype"], ",".join(map(str, item["shape"])))))
    return ("\n".join(rows) + "\n").encode("ascii")


def _checked_step(root):
    from ..compile import _stateful_json, _same_ordered_json
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("step manifest must not be a symlink")
    step = _stateful_json(manifest_path)
    if (step.get("artifact_kind") != "qwen35-stateful-step" or step.get("status") != "PREPARED"
            or type(step.get("schema_version")) is not int or step["schema_version"] != 1):
        raise ValueError("previous step must be a prepared stateful step bundle")
    omc = _stateful_json(_step_file(root, step["omc_manifest"]))
    air = _stateful_json(_step_file(root, step["air_manifest"]))
    if step["air_manifest"]["sha256"] != omc["air_manifest_sha256"]:
        raise ValueError("step AIR provenance hash mismatch")
    compiled, identity = _step_contract(omc, air)
    if not _same_ordered_json(step.get("identity"), identity):
        raise ValueError("step model identity mismatch")
    abi_path = _step_file(root, step["state_abi"])
    if step["state_abi"]["path"] != "state-abi.json" or not _same_ordered_json(_stateful_json(abi_path), omc["state_abi"]):
        raise ValueError("step state ABI differs from its model")
    expected_payloads = [{**item, "path": "model/" + item["path"]} for item in compiled["payloads"]]
    if step.get("model_payloads") != expected_payloads:
        raise ValueError("step model payload inventory mismatch")
    for item in expected_payloads:
        _step_file(root, item)
    expected_model = {"path": "model/" + compiled["omc"], "bytes": compiled["bytes"], "sha256": compiled["sha256"]}
    if step.get("model") != expected_model or expected_model not in expected_payloads:
        raise ValueError("step model differs from compiled OMC")
    inputs, graph = step.get("inputs", []), omc["graph"]
    if len(inputs) != len(graph["inputs"]):
        raise ValueError("step input count mismatch")
    seen = set()
    for index, (item, desc) in enumerate(zip(inputs, graph["inputs"])):
        if (item.get("index") != index or type(item.get("index")) is not int
                or any(item.get(key) != desc[key] for key in ("name", "dtype", "shape"))
                or item.get("bytes") != math.prod(desc["shape"]) * (2 if desc["dtype"] == "float16" else 4)
                or item.get("path") in seen):
            raise ValueError("step input descriptor mismatch")
        seen.add(item["path"])
        _step_file(root, item)
    if (step.get("outputs") != graph["outputs"] or step.get("sequence_length") != omc["sequence_length"]
            or step.get("past_length") != omc["past_length"]
            or step.get("next_past_length") != omc["state_abi"]["next_past_length"]
            or type(step.get("host_fake_ancestry")) is not bool):
        raise ValueError("step boundary differs from model")
    plan = _step_file(root, step["step_plan"])
    if step["step_plan"]["path"] != "step.tsv" or plan.read_bytes() != _step_tsv(
            expected_model, step["state_abi"], inputs, graph["outputs"], omc["sequence_length"], omc["past_length"]):
        raise ValueError("step TSV differs from its hash-bound descriptors")
    return step, omc, identity


def prepare_stateful_step(omc_dir, output_dir, token_ids, previous_step=None,
                          previous_result=None, allow_host_fake=False):
    """Stage one complete stateful HIAI call, or continue a verified previous call.

    All sequence positions commit. This does not implement speculative partial
    acceptance/rollback. A new static past-length graph is required for each
    continuation length; no cached state or model metadata is rewritten.
    """
    from ..compile import _stateful_json, _same_ordered_json
    if type(allow_host_fake) is not bool:
        raise ValueError("allow_host_fake must be boolean")
    root = Path(omc_dir).resolve()
    source_path = root / "manifest.json"
    if source_path.is_symlink():
        raise ValueError("OMC manifest must not be a symlink")
    source_hash = sha256_file(source_path)
    omc = _stateful_json(source_path)
    air_path = Path(omc.get("air_source_root", "")) / "manifest.json"
    if (not air_path.is_absolute() or air_path.is_symlink() or not air_path.is_file()
            or sha256_file(air_path) != omc.get("air_manifest_sha256")):
        raise ValueError("original AIR manifest is absent or its hash changed")
    air = _stateful_json(air_path)
    compiled, identity = _step_contract(omc, air)
    payloads, sources = compiled.get("payloads"), {}
    if not isinstance(payloads, list) or not payloads:
        raise ValueError("compiled target requires model payload records")
    for item in payloads:
        source = _step_file(root, item)
        if item["path"] in sources:
            raise ValueError("duplicate compiled model payload")
        sources[item["path"]] = source
    model_record = {"path": compiled["omc"], "bytes": compiled["bytes"], "sha256": compiled["sha256"]}
    if model_record not in payloads:
        raise ValueError("compiled OMC is missing from payload inventory")
    sequence, past = omc["sequence_length"], omc["past_length"]
    vocab = omc["graph"]["weights"]["target.embed_tokens.weight"]["shape"][0]
    if (not isinstance(token_ids, (list, tuple)) or len(token_ids) != sequence
            or any(type(token) is not int or not 0 <= token < min(vocab, 2 ** 31) for token in token_ids)):
        raise ValueError("token_ids must contain sequence_length integer IDs within the model vocabulary")
    incoming, continuation, host_fake_ancestry = {}, None, False
    if past == 0:
        if previous_step is not None or previous_result is not None:
            raise ValueError("past_length=0 starts fresh and must not receive previous state")
    else:
        if previous_step is None or previous_result is None:
            raise ValueError("past_length>0 requires both previous_step and previous_result")
        previous_root = Path(previous_step).resolve()
        old, old_omc, old_identity = _checked_step(previous_root)
        if old.get("host_fake_ancestry") and not allow_host_fake:
            raise ValueError("previous state descends from host-fake-hiai; allow_host_fake is required")
        if not _same_ordered_json(identity, old_identity):
            raise ValueError("continuation source/weights/bindings/precision/DDK/state identity mismatch")
        result_root = Path(previous_result).resolve()
        result_path = result_root / "result.json"
        if result_path.is_symlink():
            raise ValueError("previous result manifest must not be a symlink")
        result_hash = sha256_file(result_path)
        result = _stateful_json(result_path)
        expected = {"schema_version": 1, "status": "EXECUTED", "scope": "stateful-target-step",
                    "state_policy": "explicit", "target_calls": 1, "draft_calls": 0,
                    "ddk_id": old_identity["ddk_id"],
                    "step_plan_sha256": old["step_plan"]["sha256"],
                    "state_abi_sha256": old["state_abi"]["sha256"], "model_sha256": old["model"]["sha256"],
                    "sequence_length": old["sequence_length"], "past_length": old["past_length"],
                    "next_past_length": old_omc["state_abi"]["next_past_length"]}
        if (any(result.get(key) != value for key, value in expected.items())
                or any(type(result.get(key)) is not int for key in ("sequence_length", "past_length", "next_past_length"))
                or result.get("next_past_length") != past):
            raise ValueError("previous result is not a successful execution of this step at the required past length")
        evidence = result.get("evidence_kind")
        if evidence not in {"device-hiai", "host-fake-hiai"} or (evidence == "host-fake-hiai" and not allow_host_fake):
            raise ValueError("previous result requires device-hiai evidence; host-fake-hiai needs allow_host_fake")
        host_fake_ancestry = bool(old.get("host_fake_ancestry")) or evidence == "host-fake-hiai"
        if not isinstance(result.get("runtime_version"), str) or not result["runtime_version"]:
            raise ValueError("previous result lacks runtime version")
        outputs = result.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != len(old["outputs"]):
            raise ValueError("previous result output count mismatch")
        output_files, seen = {}, set()
        for index, (item, desc) in enumerate(zip(outputs, old["outputs"])):
            if (type(item.get("index")) is not int or item["index"] != index
                    or any(item.get(key) != desc[key] for key in ("name", "dtype", "shape"))
                    or item.get("bytes") != math.prod(desc["shape"]) * (2 if desc["dtype"] == "float16" else 4)
                    or item.get("path") in seen):
                raise ValueError("previous result output descriptor mismatch")
            seen.add(item["path"])
            output_files[item["name"]] = (_step_file(result_root, item), item)
        if "inputs" in result:
            if (not isinstance(result["inputs"], list) or len(result["inputs"]) != len(old["inputs"])
                    or any(any(actual.get(key) != expected_input[key] for key in
                               ("index", "name", "dtype", "shape", "bytes", "sha256"))
                           for actual, expected_input in zip(result["inputs"], old["inputs"]))):
                raise ValueError("previous result input inventory mismatch")
        old_states = {(item["layer"], item["role"]): item for item in old_omc["state_abi"]["states"]}
        for state in omc["state_abi"]["states"]:
            prior = old_states[(state["layer"], state["role"])]
            if state["input_shape"] != prior["output_shape"] or state["dtype"] != prior["dtype"]:
                raise ValueError("previous state shape/dtype cannot feed this graph")
            incoming[state["input"]] = output_files[prior["output"]]
        continuation = {"previous_step_manifest_sha256": sha256_file(previous_root / "manifest.json"),
                        "previous_result_sha256": result_hash, "evidence_kind": evidence,
                        "runtime_version": result["runtime_version"], "committed_tokens": past}
    output = output_path(output_dir)
    if output.exists():
        raise FileExistsError(f"step output already exists: {output}")
    air_hash = sha256_file(air_path)
    output.mkdir(parents=True)
    shutil.copyfile(source_path, output / "omc-manifest.json")
    shutil.copyfile(air_path, output / "air-manifest.json")
    for item in payloads:
        destination = output / "model" / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sources[item["path"]], destination)
        _step_file(output, {**item, "path": "model/" + item["path"]})
    save_json(output / "state-abi.json", omc["state_abi"])
    (output / "inputs").mkdir()
    inputs = []
    for index, desc in enumerate(omc["graph"]["inputs"]):
        path = output / "inputs" / f"input_{index:04d}.bin"
        if index == 0:
            path.write_bytes(struct.pack("<" + "i" * sequence, *token_ids))
        else:
            source, item = incoming[desc["name"]]
            shutil.copyfile(source, path)
            _step_file(output, {**item, "path": path.relative_to(output).as_posix()})
        inputs.append({"index": index, **desc, **_step_record(path, output)})
    model = {**model_record, "path": "model/" + model_record["path"]}
    abi_record = _step_record(output / "state-abi.json", output)
    (output / "step.tsv").write_bytes(_step_tsv(model, abi_record, inputs, omc["graph"]["outputs"], sequence, past))
    if (sha256_file(output / "omc-manifest.json") != source_hash or sha256_file(source_path) != source_hash
            or sha256_file(output / "air-manifest.json") != air_hash or air_hash != omc["air_manifest_sha256"]):
        raise ValueError("source manifest changed during step preparation")
    if continuation and sha256_file(result_path) != result_hash:
        raise ValueError("previous result changed during step preparation")
    step = {"schema_version": 1, "artifact_kind": "qwen35-stateful-step", "status": "PREPARED",
            "sequence_length": sequence, "past_length": past, "next_past_length": past + sequence,
            "identity": identity, "omc_manifest": _step_record(output / "omc-manifest.json", output),
            "air_manifest": _step_record(output / "air-manifest.json", output),
            "model": model, "model_payloads": [{**item, "path": "model/" + item["path"]} for item in payloads],
            "state_abi": abi_record, "inputs": inputs, "outputs": omc["graph"]["outputs"],
            "step_plan": _step_record(output / "step.tsv", output), "continuation": continuation,
            "allow_host_fake": allow_host_fake, "host_fake_ancestry": host_fake_ancestry,
            "runtime_placement": "UNVERIFIED",
            "validation_scope": "Input staging only; complete-prefix continuation, no speculative partial commit or device validation."}
    save_json(output / "manifest.json", step)
    return step
