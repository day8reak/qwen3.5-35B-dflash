"""Six-input installed-DDK GDN diagnostic; never a replacement model adapter.

The mathematical oracle uses FP64 arithmetic and FP16 invocation boundaries.
It is independent of kernel chunking/rounding and has no accuracy PASS gate.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from qwen35_moe_dflash.common.artifacts import checked_file, output_path
from qwen35_moe_dflash.common.hashing import sha256_file

KIND = "qwen35-target-gdn-six-input-diagnostic"
INPUTS = ("query", "key", "value", "g", "beta", "initial_state")
OUTPUTS = ("core_attn", "last_recurrent_state")
ATTRS = ("chunk_size", "output_final_state", "use_qk_l2norm_in_kernel")
REGISTRATION = "tools/tools_ascendc/custom_op/ascend310b/config/npu_custom_opinfo.json"
REPORTED_REGISTRATION_SHA256 = "559feea80f2105620039cb1ac3370d1b8b0653510dec880fcc6512cc43f952a4"


def write_json(path, value):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def identity(path, root):
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def mathematical_reference(query, key, value, g, beta, initial_state):
    """Full-prefix recurrence in [B,H,Dk,Dv] state orientation.

    qhat = q / sqrt(sum(q*q)+1e-6) / sqrt(Dk), khat likewise without /sqrt(Dk).
    S' = exp(g)*S; S = S' + beta*khat*(v-khat^T*S'); y = qhat^T*S.
    FP16 inputs are converted to FP64 once; only public outputs round to FP16.
    No per-token FP16 state rounding or partial-valid semantics are inferred.
    """
    arrays = [np.asarray(x) for x in (query, key, value, g, beta, initial_state)]
    q, k, v, decay, gate, state = arrays
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("expected compatible [B,T,H,D] query/key/value")
    if any(d <= 0 for d in q.shape) or v.shape[-1] <= 0:
        raise ValueError("empty GDN dimensions")
    if decay.shape != q.shape[:3] or gate.shape != q.shape[:3]:
        raise ValueError("g/beta must have shape [B,T,H]")
    if state.shape != (q.shape[0], q.shape[2], q.shape[3], v.shape[3]):
        raise ValueError("initial state must have shape [B,H,Dk,Dv]")
    if any(x.dtype != np.dtype("float16") or not np.isfinite(x).all() for x in arrays):
        raise ValueError("six-input diagnostic accepts finite FP16 tensors only")
    q, k, v, decay, gate, state = [x.astype(np.float64) for x in arrays]
    q /= np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6) * math.sqrt(q.shape[-1])
    k /= np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    result = np.empty(v.shape, dtype=np.float64)
    for t in range(q.shape[1]):
        state *= np.exp(decay[:, t, :, None, None])
        prediction = np.einsum("bhk,bhkv->bhv", k[:, t], state)
        residual = (v[:, t] - prediction) * gate[:, t, :, None]
        state += np.einsum("bhk,bhv->bhkv", k[:, t], residual)
        result[:, t] = np.einsum("bhk,bhkv->bhv", q[:, t], state)
    outputs = [result.astype("<f2"), state.astype("<f2")]
    if any(not np.isfinite(x).all() for x in outputs):
        raise ValueError("oracle output is not finite at FP16 boundary")
    return outputs


def fixture(sequence, nonzero_state):
    if sequence not in (1, 64):
        raise ValueError("probe sequence must be 1 or 64")
    rng = np.random.default_rng(35000 + sequence)
    shape = (1, sequence, 32, 128)
    q, k, v = [(rng.standard_normal(shape) * scale).astype("<f2")
               for scale in (0.25, 0.25, 0.2)]
    g = (-0.01 - 0.04 * rng.random(shape[:3])).astype("<f2")
    beta = (0.1 + 0.8 * rng.random(shape[:3])).astype("<f2")
    h, i, j = np.ogrid[:32, :128, :128]
    state = (0.08 * np.sin((h + 1) * 0.13 + i * 0.17 - j * 0.07)
             + 0.025 * (i / 127.0 - 2.0 * j / 127.0))[None].astype("<f2")
    if not nonzero_state:
        state.fill(0)
    return [q, k, v, g, beta, state]


def build_graph(sequence):
    from qwen35_moe_dflash.graph.fused.ir import Graph
    if sequence not in (1, 64):
        raise ValueError("probe sequence must be 1 or 64")
    graph = Graph(f"target_gdn_t{sequence}_six_input")
    qkv, gates, state = (1, sequence, 32, 128), (1, sequence, 32), (1, 32, 128, 128)
    ports = {name: graph.input(name, shape, "float16")
             for name, shape in zip(INPUTS, (qkv, qkv, qkv, gates, gates, state))}
    values = graph.custom("ChunkGatedDeltaRule", ports,
        {"core_attn": (qkv, "float16"), "last_recurrent_state": (state, "float16")},
        {key: {"type": "int", "value": val} for key, val in zip(ATTRS, (sequence, 1, 1))},
        name="gdn")
    graph.outputs = [values[name] for name in OUTPUTS]
    graph.validate()
    return graph


def prepare(output):
    from qwen35_moe_dflash.graph.fused.emitter import emit_cpp
    root = output_path(output)
    root.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "artifact_kind": KIND, "status": "PREPARING",
        "platform": "kirinx90", "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN",
        "numerical_validation": "NOT_RUN", "all_operators_passed": False,
        "scope": "INSTALLED_DDK_SIX_INPUT_ABI_DIAGNOSTIC_NOT_MASTER_MODEL_EQUIVALENCE",
        "contract_source": {"kind": "USER_REPORTED_DDK_REGISTRATION", "path": REGISTRATION,
                            "sha256": REPORTED_REGISTRATION_SHA256},
        "master_differences": ["effective_length input absent", "g is FP16 instead of source FP32",
                               "last_recurrent_state is FP16 instead of source Meta FP32"],
        "reference": "NUMPY_FP64_FULL_PREFIX_RECURRENCE_FP16_PUBLIC_BOUNDARIES; NOT_KERNEL_ROUNDING",
        "preparation_code": {"path": "model/qwen35_moe_dflash/target/gdn_probe.py",
                             "sha256": sha256_file(Path(__file__)), "numpy_version": np.__version__},
        "fixture_seed": "numpy.default_rng(35000 + sequence); both initial-state scenarios share q/k/v/g/beta",
        "validation_policy": {"numerical_gate": "METRICS_ONLY_NO_CALIBRATED_THRESHOLD",
            "warmup": 1, "repetitions": 10, "stability_rule": "ALL_OUTPUT_BYTES_IDENTICAL",
            "comparison_extent": "ALL_ELEMENTS_OF_BOTH_OUTPUTS_ON_EVERY_MEASURED_INVOCATION",
            "feedback": "NOT_TESTED_REPEATED_INVOCATIONS_REUSE_ORIGINAL_STATE",
            "partial_valid": "NOT_SUPPORTED_BY_THIS_SIX_INPUT_PROBE",
            "output_fill_00_a5_5a": "UNRESOLVED_NOT_COVERED_BY_GENERIC_RUNNER",
            "output_initialization_modes": ["zero", "untouched"]},
        "models": {}, "cases": []}
    write_json(root / "manifest.json", result)
    for seq in (64, 1):
        graph = build_graph(seq)
        folder = root / "models" / f"t{seq}"
        folder.mkdir(parents=True)
        source = folder / "probe.cpp"
        source.write_text(emit_cpp(graph, fold_constant_arithmetic=False))
        result["models"][f"t{seq}"] = {"source": identity(source, root), "graph": graph.manifest(),
                                         "model_compilation": "NOT_RUN"}
        for nonzero in (False, True):
            case_id = f"t{seq}_" + ("nonzero_state" if nonzero else "zero_state")
            case = root / "cases" / case_id
            case.mkdir(parents=True)
            inputs = fixture(seq, nonzero)
            golden = mathematical_reference(*inputs)
            plan = ["QWEN35_TENSOR_PROBE 1", "iterations 1 10"]
            records = {}
            for prefix, names, arrays in (("input", INPUTS, inputs), ("golden", OUTPUTS, golden)):
                records[prefix] = []
                for index, (name, array) in enumerate(zip(names, arrays)):
                    path = case / f"{prefix}_{index}.bin"
                    array.tofile(path)
                    records[prefix].append({**identity(path, root), "name": name, "index": index,
                                           "shape": list(array.shape), "dtype": "float16"})
                    plan.append(f'{"output" if prefix == "golden" else "input"} {index} {name} float16 '
                                + ",".join(str(d) for d in array.shape))
            plan_path = case / "plan.txt"
            plan_path.write_text("\n".join(plan) + "\n")
            result["cases"].append({"id": case_id, "model": f"t{seq}", **records,
                                    "plan": identity(plan_path, root)})
    result["status"] = "PREPARED"
    write_json(root / "manifest.json", result)
    return result


def load_prepared(output):
    root = output_path(output)
    result = json.loads((root / "manifest.json").read_text())
    if result.get("artifact_kind") != KIND or result.get("status") != "PREPARED":
        raise ValueError("not a completed GDN diagnostic preparation")
    if set(result["models"]) != {"t64", "t1"} or len(result["cases"]) != 4:
        raise ValueError("incomplete probe matrix")
    for model in result["models"].values():
        checked_file(root, model["source"])
    for case in result["cases"]:
        for item in [case["plan"], *case["input"], *case["golden"]]:
            checked_file(root, item)
    return root, result


def check_registration(ddk):
    path = Path(ddk).resolve() / REGISTRATION
    entry = json.loads(path.read_text())["ChunkGatedDeltaRule"]
    for prefix, names in (("input", INPUTS), ("output", OUTPUTS)):
        slots = {k for k in entry if k.startswith(prefix) and k[len(prefix):].isdigit()}
        if slots != {f"{prefix}{i}" for i in range(len(names))}:
            raise ValueError(f"DDK GDN {prefix} count differs from six-input diagnostic")
        for index, name in enumerate(names):
            port = entry[f"{prefix}{index}"]
            if any(port.get(key) != expected for key, expected in (
                    ("name", name), ("dtype", "float16"), ("format", "ND"), ("paramType", "required"))):
                raise ValueError(f"DDK GDN {prefix}{index} contract differs: {port}")
    if set(entry.get("attr", {}).get("list", "").split(",")) != set(ATTRS):
        raise ValueError("DDK GDN attributes differ")
    if any(entry.get("attr_" + name, {}).get("type") != "int" for name in ATTRS):
        raise ValueError("DDK GDN attributes must be int")
    return {"path": str(path), "sha256": sha256_file(path), "registration": entry,
            "scope": "DECLARATION_ONLY; COMPILED_IMPLEMENTATION_SELECTION_UNVERIFIED"}


def compile_prepared(output, *, ge_root, ddk_root, ddk_id, cxx="c++", cxx11_abi=0, timeout_seconds=600):
    from qwen35_moe_dflash.common.tool_logs import compilation_diagnostics, log_excerpt
    from qwen35_moe_dflash.graph.export import _logged
    from qwen35_moe_dflash.toolchain.existing_air import compile_existing_air
    root, manifest = load_prepared(output)
    if not ge_root or not ddk_root or not ddk_id:
        raise ValueError("compile requires CANN_ROOT, CPRO_DDK_ROOT and CPRO_DDK_ID (or explicit options)")
    ge, ddk = Path(ge_root).resolve(), Path(ddk_root).resolve()
    if any(root == p or p in root.parents or root in p.parents for p in (ge, ddk)):
        raise ValueError("probe output and SDK roots must not contain one another")
    registration = check_registration(ddk)
    library = ge / "lib64"
    for path in (ge / "include/graph/graph.h", library / "libgraph.so"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (root / "compile.json").exists():
        raise FileExistsError("compile.json already exists; retain this diagnostic evidence")
    result = {"artifact_kind": KIND, "status": "COMPILING", "platform": "kirinx90",
              "registration": registration, "models": {}, "model_compilation": "NOT_RUN",
              "device_execution": "NOT_RUN", "numerical_validation": "NOT_RUN",
              "placement": "UNVERIFIED", "manifest_sha256": sha256_file(root / "manifest.json")}
    write_json(root / "compile.json", result)
    for name, model in manifest["models"].items():
        folder = root / "models" / name
        row = {"status": "COMPILING"}
        result["models"][name] = row
        try:
            source = checked_file(root, model["source"])
            air = folder / "air" / "probe.air"
            air.parent.mkdir()
            exporter = folder / "exporter"
            env = dict(os.environ)
            env["LD_LIBRARY_PATH"] = str(library) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
            argv = [cxx, "-std=c++17", f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}", "-I", str(ge / "include"),
                    str(source), "-L", str(library), f"-Wl,-rpath,{library}", "-lgraph"]
            if (library / "libgraph_base.so").is_file():
                argv.append("-lgraph_base")
            argv += ["-o", str(exporter)]
            write_json(folder / "build-command.json", {"argv": argv})
            _logged(argv, folder=folder, env=env, log=folder / "build.log", timeout=timeout_seconds)
            _logged([str(exporter), str(air)], folder=folder, env=env,
                    log=folder / "export.log", timeout=timeout_seconds)
            row["air"] = identity(air, root)
            compiled = compile_existing_air(air, folder / "omc", ddk, ddk_id=ddk_id,
                timeout_seconds=timeout_seconds, use_origin_format=True, save_weights_as_external_data=False)
            if compiled["status"] != "COMPILED":
                raise RuntimeError(str(compiled.get("error", compiled["status"])))
            omc = folder / "omc" / compiled["omc"]["path"]
            row.update(status="COMPILED", omc=identity(omc, root))
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            row.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            if (folder / "omc").exists():
                row["diagnostics"] = compilation_diagnostics(folder / "omc")
            else:
                row["diagnostics"] = {name: log_excerpt(folder / name) for name in ("build.log", "export.log")}
        write_json(root / "compile.json", result)
    result["status"] = ("COMPILED" if all(x["status"] == "COMPILED" for x in result["models"].values())
                        else "PARTIAL" if any(x["status"] == "COMPILED" for x in result["models"].values()) else "FAILED")
    result["model_compilation"] = result["status"]
    result["next"] = ("Run validate_target_attention.py with --reuse-gdn pointing to this directory."
                      if result["status"] == "COMPILED" else
                      "Read models.*.diagnostics for the compiler failure; retain these logs before any retry.")
    write_json(root / "compile.json", result)
    return result


def metrics(actual, golden):
    actual, golden = np.asarray(actual, dtype=np.float64), np.asarray(golden, dtype=np.float64)
    if actual.shape != golden.shape:
        raise ValueError("comparison shape mismatch")
    finite = np.isfinite(actual) & np.isfinite(golden)
    result = {"elements": actual.size, "nonfinite_actual": int((~np.isfinite(actual)).sum()),
              "nonfinite_golden": int((~np.isfinite(golden)).sum()), "allclose_gate": "NOT_DEFINED"}
    if not finite.all():
        result.update(max_abs=None, rmse=None, relative_l2=None, cosine=None)
        return result
    delta = actual - golden
    ng, na = np.linalg.norm(golden.ravel()), np.linalg.norm(actual.ravel())
    result.update(max_abs=float(np.max(np.abs(delta))), rmse=float(np.sqrt(np.mean(delta * delta))),
                  relative_l2=float(np.linalg.norm(delta.ravel()) / ng) if ng else None,
                  cosine=float(np.dot(actual.ravel(), golden.ravel()) / (na * ng)) if na and ng else None,
                  exact_elements=int(np.count_nonzero(actual == golden)))
    return result


def runner_commands(output):
    root, manifest = load_prepared(output)
    compiled = json.loads((root / "compile.json").read_text()) if (root / "compile.json").exists() else {}
    commands = []
    for case in manifest["cases"]:
        model = compiled.get("models", {}).get(case["model"], {})
        path = model.get("omc", {}).get("path", f'<compiled OMC for {case["model"]}>')
        commands.append({"case": case["id"], "model_status": model.get("status", "NOT_COMPILED"),
            "argv": ["<qwen35_tensor_probe>", f"<deployed-probe-root>/{path}", "<runtime-version>",
                     f'<deployed-probe-root>/cases/{case["id"]}', f'<new-device-output-root>/{case["id"]}',
                     "zero"], "alternative_output_initialization": "untouched",
            "note": "Use separate output roots for zero and untouched. This command never runs from this tool."})
    return {"commands": commands, "model_and_fixture_deployment": "COPY_HASH_CHECKED_FILES_KEEP_RELATIVE_PATHS",
            "required_snapshot_count_per_case": 10, "feedback": "NOT_TESTED"}


def compare(output, actual_dir):
    root, manifest = load_prepared(output)
    actual_dir = Path(actual_dir).resolve()
    report = {"artifact_kind": KIND, "status": "DIAGNOSTIC_METRICS_ONLY", "cases": [],
              "numerical_validation": "METRICS_ONLY_NO_CALIBRATED_THRESHOLD", "all_operators_passed": False,
              "provenance": "RUNNER_RECEIPT_IS_NOT_PROOF_OF_DEPLOYED_MODEL_OR_INPUT_HASHES",
              "manifest_sha256": sha256_file(root / "manifest.json"), "device_execution": "UNVERIFIED"}
    for case in manifest["cases"]:
        folder = actual_dir / case["id"]
        row = {"id": case["id"], "status": "NOT_FOUND", "invocations": []}
        report["cases"].append(row)
        if not folder.is_dir():
            continue
        try:
            receipt = json.loads((folder / "result.json").read_text())
            if (receipt.get("status") != "EXECUTED" or receipt.get("repetitions") != 10
                    or receipt.get("warmup") != 1 or len(receipt.get("api_us", [])) != 10
                    or receipt.get("output_initialization") not in ("zero", "untouched")):
                raise ValueError("receipt does not match frozen runner plan")
            if receipt.get("evidence_kind") not in ("device-hiai", "host-fake-hiai"):
                raise ValueError("unknown runner evidence kind")
            golden = [np.fromfile(checked_file(root, item), dtype="<f2").reshape(item["shape"])
                      for item in case["golden"]]
            extent = sum(x.nbytes for x in golden)
            for i in range(10):
                path = folder / f"{i}.bin"
                if path.is_symlink() or not path.is_file() or path.stat().st_size != extent:
                    raise ValueError(f"missing/wrong output extent: {path.name}")
                data = path.read_bytes()
                invocation = {"index": i, "sha256": hashlib.sha256(data).hexdigest(), "outputs": {}}
                offset = 0
                for name, expected in zip(OUTPUTS, golden):
                    actual = np.frombuffer(data, dtype="<f2", count=expected.size, offset=offset).reshape(expected.shape)
                    invocation["outputs"][name] = metrics(actual, expected)
                    offset += expected.nbytes
                row["invocations"].append(invocation)
            row.update(status="METRICS_RECORDED", runner_receipt=receipt,
                       receipt_sha256=sha256_file(folder / "result.json"),
                       output_bytes_stable=len({x["sha256"] for x in row["invocations"]}) == 1)
        except (OSError, ValueError, KeyError, TypeError) as error:
            row.update(status="INVALID_EVIDENCE", error=f"{type(error).__name__}: {error}")
    if all(x["status"] == "METRICS_RECORDED" for x in report["cases"]):
        kinds = {x["runner_receipt"]["evidence_kind"] for x in report["cases"]}
        report["device_execution"] = "REPORTED_BY_RUNNER_UNBOUND_MODEL" if kinds == {"device-hiai"} else "HOST_OR_MIXED_EVIDENCE"
    else:
        report["status"] = "PARTIAL"
    return report
