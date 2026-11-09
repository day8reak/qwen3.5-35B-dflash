"""CacheUpdate physical-layout diagnostic, with whole-cache bit comparison.

This probes the returned tensor, not the master's in-place alias/copy-back ABI.
No server or device access occurs here; deployment identity remains unverified.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from qwen35_moe_dflash.common.artifacts import checked_file, output_path
from qwen35_moe_dflash.common.hashing import sha256_file
# These helpers only persist/hash artifacts; no GDN graph or numerical policy is reused.
from qwen35_moe_dflash.target.gdn_probe import identity, write_json

KIND = "qwen35-target-cache-update-diagnostic"
SHAPE = (64, 32, 32, 16)  # physical block, head/channel group, token offset, channel lane
INPUTS = ("x", "updates", "targetBlock", "offsetInBlock")
DTYPES = ("float16", "float16", "int32", "int32")
REGISTRATION = "tools/tools_ascendc/custom_op/ascend310b/config/npu_custom_opinfo.json"
CASES = {"t64_boundary_identity": (64, 31, False), "t64_boundary_permuted": (64, 31, True),
         "t1_block_end": (1, 63, True), "t1_block_start": (1, 64, True)}


def physical_reference(cache, updates, blocks, offsets):
    """Copy one complete [channel_group, lane] slice for each token.

    output[physical_block, channel_group, token_offset, lane] =
        updates[token, channel_group, lane]. Duplicate destinations are excluded
    because their ordering/atomicity contract is not established for this kernel.
    """
    if cache.shape != SHAPE or cache.dtype != np.dtype("<f2"):
        raise ValueError("cache must be FP16 [64,32,32,16]")
    if updates.ndim != 3 or updates.shape[1:] != (32, 16) or updates.dtype != np.dtype("<f2"):
        raise ValueError("updates must be FP16 [T,32,16]")
    count = updates.shape[0]
    if count not in (1, 64) or any(x.shape != (count,) or x.dtype != np.dtype("<i4") for x in (blocks, offsets)):
        raise ValueError("indices must be INT32 [T], T=1 or 64")
    if np.any((blocks < 0) | (blocks >= 64)) or np.any((offsets < 0) | (offsets >= 32)):
        raise ValueError("cache write index out of bounds")
    if len(set(zip(blocks.tolist(), offsets.tolist()))) != count:
        raise ValueError("duplicate cache destinations are outside this probe contract")
    result = cache.copy()
    for row, (block, offset) in enumerate(zip(blocks, offsets)):
        result[block, :, offset, :] = updates[row]
    return result


def fixture(case_id):
    count, start, permuted = CASES[case_id]
    b, h, t, d = np.ogrid[:64, :32, :32, :16]
    # Finite nonzero FP16 bit patterns encode all four physical axes. Updates
    # have the opposite sign, so every intended write changes its destination.
    cache = (0x0400 + (977*b + 131*h + 37*t + 11*d) % 0x77ff).astype("<u2").view("<f2")
    r, h, d = np.ogrid[:count, :32, :16]
    updates = ((0x0400 + (523*r + 89*h + 7*d) % 0x77ff) | 0x8000).astype("<u2").view("<f2")
    table = ((17*np.arange(64) + 11) % 64 if permuted else np.arange(64)).astype("<i4")
    positions = np.arange(start, start + count, dtype=np.int32)
    return [cache, updates, table[positions // 32], positions % 32], table


def build_graph(sequence):
    from qwen35_moe_dflash.graph.fused.ir import Graph
    if sequence not in (1, 64):
        raise ValueError("CacheUpdate graph sequence must be 1 or 64")
    graph = Graph(f"target_cache_update_t{sequence}")
    shapes = (SHAPE, (sequence, 32, 16), (sequence,), (sequence,))
    inputs = {n: graph.input(n, shape, dtype) for n, shape, dtype in zip(INPUTS, shapes, DTYPES)}
    result = graph.custom("CacheUpdate", inputs, {"x": (SHAPE, "float16")}, name="cache_update")
    graph.outputs = [result["x"]]
    graph.validate()
    return graph


def prepare(output):
    from qwen35_moe_dflash.graph.fused.emitter import emit_cpp
    root = output_path(output)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "artifact_kind": KIND, "status": "PREPARING", "platform": "kirinx90",
        "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN", "numerical_validation": "NOT_RUN",
        "scope": "ISOLATED_RETURNED_CACHE_DIAGNOSTIC_NOT_MASTER_ALIAS_VALIDATION", "all_operators_passed": False,
        "layout": {"cache": "[physical_block, heads*D/16, block_token_offset, 16]",
                   "updates": "[token, heads*D/16, 16]", "logical_head_count": 2, "head_dim": 256,
                   "source": "master Qwen3_5MoeAttention.update: new_k.reshape(B,T,-1,16); block_table[position//32]"},
        "preparation_code_sha256": sha256_file(Path(__file__)),
        "validation_policy": {"numerical_gate": "EXACT_ALL_FP16_BITS_IN_WHOLE_CACHE",
            "untouched_locations": "MUST_EQUAL_ORIGINAL_CACHE_BITS", "warmup": 1, "repetitions": 10,
            "stability_rule": "ALL_OUTPUT_BYTES_IDENTICAL", "deployment_identity": "UNVERIFIED",
            "alias_semantics": "UNVERIFIED_INPUT_OUTPUT_ALIAS_AND_MASTER_COPY_BACK",
            "repetition": "SAME_DESTINATIONS_AND_UPDATES; IDEMPOTENT; NOT_MOVING_STATE_FEEDBACK",
            "excluded": ["duplicate destinations", "out-of-range writes", "input mutation observation"]},
        "models": {}, "cases": []}
    write_json(root / "manifest.json", manifest)
    for sequence in (64, 1):
        graph = build_graph(sequence)
        folder = root / "models" / f"t{sequence}"
        folder.mkdir(parents=True)
        source = folder / "probe.cpp"
        source.write_text(emit_cpp(graph, fold_constant_arithmetic=False))
        manifest["models"][f"t{sequence}"] = {"source": identity(source, root), "graph": graph.manifest()}
    for name, (sequence, start, permuted) in CASES.items():
        folder = root / "cases" / name
        folder.mkdir(parents=True)
        inputs, table = fixture(name)
        expected = physical_reference(*inputs)
        row = {"id": name, "model": f"t{sequence}", "logical_start": start, "logical_end": start+sequence-1,
               "block_table": table.tolist(), "permuted_blocks": permuted, "inputs": []}
        plan = ["QWEN35_TENSOR_PROBE 1", "iterations 1 10"]
        for i, (port, dtype, array) in enumerate(zip(INPUTS, DTYPES, inputs)):
            path = folder / f"input_{i}.bin"; array.tofile(path)
            row["inputs"].append({**identity(path, root), "name": port, "dtype": dtype, "shape": list(array.shape)})
            plan.append(f'input {i} {port} {dtype} ' + ",".join(map(str, array.shape)))
        golden = folder / "golden_0.bin"; expected.tofile(golden)
        row["golden"] = {**identity(golden, root), "dtype": "float16", "shape": list(SHAPE)}
        plan.append("output 0 x float16 64,32,32,16")
        (folder / "plan.txt").write_text("\n".join(plan) + "\n")
        row["plan"] = identity(folder / "plan.txt", root)
        manifest["cases"].append(row)
    manifest["status"] = "PREPARED"
    write_json(root / "manifest.json", manifest)
    return manifest


def load_prepared(output):
    root = output_path(output)
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("artifact_kind") != KIND or manifest.get("status") != "PREPARED"
            or {row["id"] for row in manifest["cases"]} != set(CASES) or set(manifest["models"]) != {"t64", "t1"}):
        raise ValueError("expected complete CacheUpdate diagnostic preparation")
    for record in [m["source"] for m in manifest["models"].values()] + [
            item for row in manifest["cases"] for item in [row["plan"], row["golden"], *row["inputs"]]]:
        checked_file(root, record)
    return root, manifest


def check_registration(ddk):
    path = Path(ddk).resolve() / REGISTRATION
    entry = json.loads(path.read_text())["CacheUpdate"]
    for prefix, names, dtypes in (("input", INPUTS, DTYPES), ("output", ("x",), ("float16",))):
        slots = {key for key in entry if key.startswith(prefix) and key[len(prefix):].isdigit()}
        if slots != {f"{prefix}{i}" for i in range(len(names))}:
            raise ValueError("CacheUpdate registration has different port count")
        for i, (name, dtype) in enumerate(zip(names, dtypes)):
            port = entry[f"{prefix}{i}"]
            if any(port.get(key) != value for key, value in (
                    ("name", name), ("dtype", dtype), ("format", "ND"), ("paramType", "required"))):
                raise ValueError(f"CacheUpdate {prefix}{i} differs from declared diagnostic ABI")
    if entry.get("attr", {}).get("list", ""):
        raise ValueError("CacheUpdate diagnostic declares no attributes")
    return {"path": str(path), "sha256": sha256_file(path), "registration": entry,
            "scope": "DECLARATIONS_ONLY_NOT_KERNEL_SELECTION_OR_ALIAS_PROOF"}


def compile_prepared(output, *, ge_root, ddk_root, ddk_id, cxx="c++", cxx11_abi=0, timeout_seconds=600):
    from qwen35_moe_dflash.common.tool_logs import compilation_diagnostics, log_excerpt
    from qwen35_moe_dflash.graph.export import _logged
    from qwen35_moe_dflash.toolchain.existing_air import compile_existing_air
    root, manifest = load_prepared(output)
    if not all((ge_root, ddk_root, ddk_id)):
        raise ValueError("compile requires CANN_ROOT, CPRO_DDK_ROOT and CPRO_DDK_ID or explicit options")
    ge, ddk = Path(ge_root).resolve(), Path(ddk_root).resolve()
    if any(root == p or p in root.parents or root in p.parents for p in (ge, ddk)):
        raise ValueError("probe output and SDK roots must not contain one another")
    registration = check_registration(ddk)
    library = ge / "lib64"
    for path in (ge / "include/graph/graph.h", library / "libgraph.so"):
        if not path.is_file(): raise FileNotFoundError(path)
    if (root / "compile.json").exists(): raise FileExistsError("retain existing compile.json evidence")
    report = {"artifact_kind": KIND, "status": "COMPILING", "registration": registration, "models": {},
              "manifest_sha256": sha256_file(root / "manifest.json"), "device_execution": "NOT_RUN",
              "model_compilation": "NOT_RUN", "all_operators_passed": False,
              "numerical_validation": "NOT_RUN", "placement_and_alias": "UNVERIFIED"}
    write_json(root / "compile.json", report)
    env = dict(os.environ); env["LD_LIBRARY_PATH"] = str(library) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    for name, model in manifest["models"].items():
        folder = root / "models" / name
        row = report["models"][name] = {"status": "COMPILING"}
        try:
            source = checked_file(root, model["source"])
            exporter, air = folder / "exporter", folder / "air" / "probe.air"
            air.parent.mkdir()
            argv = [cxx, "-std=c++17", f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}", "-I", str(ge / "include"),
                    str(source), "-L", str(library), f"-Wl,-rpath,{library}", "-lgraph"]
            if (library / "libgraph_base.so").is_file(): argv.append("-lgraph_base")
            argv += ["-o", str(exporter)]
            write_json(folder / "build-command.json", {"argv": argv})
            _logged(argv, folder=folder, env=env, log=folder / "build.log", timeout=timeout_seconds)
            _logged([str(exporter), str(air)], folder=folder, env=env, log=folder / "export.log", timeout=timeout_seconds)
            compiled = compile_existing_air(air, folder / "omc", ddk, ddk_id=ddk_id,
                timeout_seconds=timeout_seconds, use_origin_format=True, save_weights_as_external_data=False)
            row.update(air=identity(air, root), compiler_status=compiled["status"])
            if compiled["status"] != "COMPILED": raise RuntimeError(str(compiled.get("error", compiled["status"])))
            row.update(status="COMPILED", omc=identity(folder / "omc" / compiled["omc"]["path"], root))
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            row.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            if (folder / "omc").exists():
                row["diagnostics"] = compilation_diagnostics(folder / "omc")
            else:
                row["diagnostics"] = {name: log_excerpt(folder / name) for name in ("build.log", "export.log")}
        write_json(root / "compile.json", report)
    states = {row["status"] for row in report["models"].values()}
    report["status"] = "COMPILED" if states == {"COMPILED"} else "FAILED" if states == {"FAILED"} else "PARTIAL"
    report["model_compilation"] = report["status"]
    write_json(root / "compile.json", report)
    return report


def runner_commands(output):
    root, manifest = load_prepared(output)
    compiled = json.loads((root / "compile.json").read_text()) if (root / "compile.json").exists() else {}
    commands = []
    for case in manifest["cases"]:
        model = compiled.get("models", {}).get(case["model"], {})
        omc = model.get("omc", {}).get("path", f'<compiled OMC for {case["model"]}>')
        commands.append({"case": case["id"], "model_status": model.get("status", "NOT_COMPILED"),
            "argv": ["<qwen35_tensor_probe>", f"<deployed-probe-root>/{omc}", "<runtime-version>",
                     f'<deployed-probe-root>/cases/{case["id"]}', f'<new-output-root>/{case["id"]}', "zero"]})
    return {"commands": commands, "alternative_initialization": "untouched (use another output root)",
            "deployment": "HASH_CHECK_ALL_MODEL_AND_FIXTURE_FILES; RETAIN_RELATIVE_PATHS",
            "alias_semantics": "UNVERIFIED; OUTPUT_COMPARISON_CANNOT_PROVE_INPUT_MUTATION"}


def compare(output, actual_dir):
    root, manifest = load_prepared(output)
    report = {"artifact_kind": KIND, "status": "DIAGNOSTIC_COMPARISON", "all_operators_passed": False,
              "numerical_gate": "EXACT_ALL_FP16_BITS_IN_WHOLE_CACHE", "device_execution": "UNVERIFIED",
              "provenance": "RUNNER_RECEIPT_DOES_NOT_BIND_DEPLOYED_MODEL_OR_INPUT_HASHES",
              "alias_semantics": "UNVERIFIED", "manifest_sha256": sha256_file(root / "manifest.json"), "cases": []}
    for case in manifest["cases"]:
        row = {"id": case["id"], "status": "NOT_FOUND", "invocations": []}
        report["cases"].append(row)
        folder = Path(actual_dir).resolve() / case["id"]
        if not folder.is_dir(): continue
        try:
            receipt = json.loads((folder / "result.json").read_text())
            if (receipt.get("status") != "EXECUTED" or receipt.get("warmup") != 1 or receipt.get("repetitions") != 10
                    or len(receipt.get("api_us", [])) != 10 or receipt.get("output_initialization") not in ("zero", "untouched")
                    or receipt.get("evidence_kind") not in ("device-hiai", "host-fake-hiai")):
                raise ValueError("runner receipt does not match frozen plan")
            expected = np.fromfile(checked_file(root, case["golden"]), dtype="<u2").reshape(SHAPE)
            blocks, offsets = [np.fromfile(checked_file(root, r), dtype="<i4") for r in case["inputs"][2:]]
            written = np.zeros(SHAPE, dtype=bool); written[blocks, :, offsets, :] = True
            for i in range(10):
                path = folder / f"{i}.bin"
                if path.is_symlink() or not path.is_file() or path.stat().st_size != expected.nbytes:
                    raise ValueError(f"missing/wrong snapshot extent: {i}.bin")
                raw = path.read_bytes(); actual = np.frombuffer(raw, dtype="<u2").reshape(SHAPE)
                different = actual != expected
                row["invocations"].append({"index": i, "sha256": hashlib.sha256(raw).hexdigest(),
                    "mismatched_fp16_elements": int(np.count_nonzero(different)),
                    "mismatched_written_elements": int(np.count_nonzero(different & written)),
                    "mismatched_untouched_elements": int(np.count_nonzero(different & ~written)),
                    "first_mismatch_coordinates": np.argwhere(different)[:8].tolist()})
            exact = all(x["mismatched_fp16_elements"] == 0 for x in row["invocations"])
            row.update(status="ALL_BITS_MATCH" if exact else "NUMERICAL_MISMATCH", runner_receipt=receipt,
                       receipt_sha256=sha256_file(folder / "result.json"),
                       output_bytes_stable=len({x["sha256"] for x in row["invocations"]}) == 1)
        except (OSError, ValueError, KeyError, TypeError) as error:
            row.update(status="INVALID_EVIDENCE", error=f"{type(error).__name__}: {error}")
    states = {row["status"] for row in report["cases"]}
    report["numerical_comparison"] = "ALL_BITS_MATCH" if states == {"ALL_BITS_MATCH"} else "NOT_ALL_MATCH"
    if "NUMERICAL_MISMATCH" in states: report["status"] = "NUMERICAL_MISMATCH"
    elif states != {"ALL_BITS_MATCH"}: report["status"] = "PARTIAL"
    return report
