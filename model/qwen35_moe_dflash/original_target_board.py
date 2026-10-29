"""Run the original target through the user's existing HIAI V2 board tool.

The user supplied the positional input mapping and tool invocation. All byte
sizes and dtype numbers come from this compilation's selected-schema boundary
report. This first invocation does not assert state continuity or accuracy.
"""
from __future__ import annotations

import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import uuid

from .artifacts import output_path, save_json
from .full_target import _recorded_file, read_metadata
from .existing_air import _stat
from .hashing import sha256_file
from .hdc_run import Hdc


_WIDTHS = {"DT_FLOAT": 4, "DT_FLOAT32": 4, "DT_FLOAT16": 2, "DT_INT8": 1, "DT_UINT8": 1,
           "DT_INT16": 2, "DT_UINT16": 2, "DT_INT32": 4, "DT_UINT32": 4,
           "DT_INT64": 8, "DT_UINT64": 8, "DT_BOOL": 1, "DT_BF16": 2}
_PATH = re.compile(r"/[A-Za-z0-9_./-]+\Z")


def _board_path(value):
    if not isinstance(value, str) or not _PATH.fullmatch(value) or ".." in PurePosixPath(value).parts:
        raise ValueError("board path must be absolute with simple path components")
    return PurePosixPath(value)


def original_input_files():
    """Exact positional semantics supplied by the user's original run script."""
    rows = [("inputs_embeds", "inputs_embeds.bin"), ("attention_mask", "attention_mask.bin"),
            ("position_ids", "position_ids.bin")]
    for layer in range(40):
        names = ("past_k", "past_v") if layer % 4 == 3 else ("past_conv_state", "past_recurrent_state")
        rows.extend((f"RefData_{2 * layer + j}", name + ".bin") for j, name in enumerate(names))
    rows.extend((name, name + ".bin") for name in
                ("new_kv_cache_pos", "output_pos", "embed_scale", "allQLen", "conv_state_pos", "effective_length"))
    return rows


def _width(desc):
    names = desc.get("schema_dtype_names", [])
    widths = {_WIDTHS[name] for name in names if name in _WIDTHS}
    if not names or len(widths) != 1 or any(name not in _WIDTHS for name in names):
        raise ValueError("unknown/unsupported selected-schema input/output dtype: " + str(names))
    return next(iter(widths))


def tool_plan(contract, profile, gear, input_dir):
    if type(gear) is not int or gear not in {1, 64}:
        raise ValueError("original target gear must be 64 or 1")
    if (contract.get("artifact_kind") != "qwen35-original-target-contract" or contract.get("status") != "OBSERVED"
            or contract.get("boundaries", {}).get("declared_profile_check", {}).get("status") != "MATCHED_VISIBLE_BOUNDARIES"):
        raise ValueError("board execution requires matching actual compiled original-target boundaries")
    source_dir = _board_path(input_dir)
    boundary = contract["boundaries"]
    actual = {row["name"]: row for row in boundary["inputs_in_serialized_op_order"]}
    entries = [entry.split(":", 1) for entry in profile["input_shape"].split(";")]
    semantics = original_input_files()
    if len(entries) != 89 or len(actual) != 89 or len(semantics) != 89:
        raise ValueError("original target must expose 89 unambiguous root input nodes")
    indices = [actual[name].get("serialized_data_index") for name, _ in entries]
    global_indices = all(type(i) is int for i in indices) and set(indices) == set(range(89))
    if global_indices and indices != list(range(89)):
        raise ValueError("complete compiled input indices contradict the user positional mapping")
    inputs = []
    for index, ((name, shape), (semantic, filename)) in enumerate(zip(entries, semantics)):
        desc = actual[name]
        shape = [gear if int(d) == -1 else int(d) for d in shape.split(",")]
        inputs.append({"index": index, "name": name, "semantic": semantic, "shape": shape,
                       "serialized_data_index": desc.get("serialized_data_index"),
                       "bytes": math.prod(shape) * _width(desc), "schema_dtype_names": desc["schema_dtype_names"],
                       "source": str(source_dir / filename), "file": f"inputs/input_{index:03d}.bin"})
    outputs = boundary["outputs"]
    if len(outputs) != 81 or len({row["netoutput"] for row in outputs}) != 1:
        raise ValueError("original target requires one 81-port output (logits plus 80 states)")
    planned = []
    for index, desc in enumerate(outputs):
        name = "logits" if index == 0 else f"RefData_{index - 1}"
        expected = [1, 1, 248320] if index == 0 else actual[name]["shape"]
        if desc["output_index"] != index or desc["shape"] != expected:
            raise ValueError("original target output shape/order differs from the user contract: " + name)
        if index == 0 and not set(desc["schema_dtype_names"]) <= {"DT_FLOAT", "DT_FLOAT16"}:
            raise ValueError("original logits must have a known FP32/FP16 dtype")
        if index and desc["schema_dtype_names"] != actual[name]["schema_dtype_names"]:
            raise ValueError("returned RefData dtype cannot be fed back exactly: " + name)
        planned.append({"index": index, "name": name, "shape": expected, "bytes": math.prod(expected) * _width(desc),
                        "schema_dtype_names": desc["schema_dtype_names"]})
    return {"schema_version": 1, "artifact_kind": "qwen35-original-target-tool-plan", "gear": gear,
            "inputs": inputs, "outputs": planned,
            "input_shape": ";".join(r["name"] + ":" + ",".join(map(str, r["shape"])) for r in inputs),
            "output_shape": ";".join(r["name"] + ":" + ",".join(map(str, r["shape"])) for r in planned),
            "input_mapping_source": "USER_PROVIDED_SCRIPT_POSITIONAL_MAPPING_WITH_COMPILED_INDEX_SHAPE_DTYPE_CHECKS",
            "serialized_index_check": "MATCHED_GLOBAL_INDICES" if global_indices else "SEPARATE_OR_INCOMPLETE_INDICES_NOT_RUNTIME_SLOTS",
            "runtime_input_order": "USER_DECLARED_UNVERIFIED_BY_DEVICE_REFERENCE",
            "target_features": "NOT_EXPOSED_BY_USER_81_OUTPUT_CONTRACT",
            "state_continuity": "NOT_VALIDATED_SINGLE_TOOL_INVOCATION", "numerical_validation": "NOT_RUN_NO_REFERENCE"}


def _hash_lines(text, root):
    result = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line.rstrip("\r"))
        if match is None:
            raise ValueError("unexpected board checksum response: " + line[:200])
        path = _board_path(match[2])
        relative = path.relative_to(root).as_posix()
        if relative in result:
            raise ValueError("duplicate output checksum path")
        result[relative] = match[1]
    return result


def execute_target_tool(compilation_dir, contract_path, output_dir, *, profile, gear=64,
                        input_dir=None, host_input_dir=None, tool="/data/local/tmp/tools_sysdbg/model_run_tool_internal",
                        board_model_root="/data/local/tmp/qwen3_5_moe", timeout_seconds=7200):
    root, out = Path(compilation_dir).resolve(), output_path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "artifact_kind": "qwen35-original-target-tool-execution",
              "status": "FAILED", "stage": "contract", "device_execution": "NOT_RUN",
              "numerical_validation": "NOT_RUN_NO_REFERENCE", "state_continuity": "NOT_RUN",
              "runtime_placement": "UNVERIFIED", "retrieved": {}, "retrieval_errors": []}
    hdc = None
    attempt = None
    attempted = False
    def collect(remote, local, label):
        _, text = hdc.shell("sha256sum " + shlex.quote(str(remote)), "hash-" + label)
        hashes = _hash_lines(text, remote.parent)
        digest = hashes.get(remote.name)
        if digest is None:
            raise ValueError("missing retrieved-file hash")
        local.parent.mkdir(parents=True, exist_ok=True)
        hdc.receive(remote, local)
        if sha256_file(local) != digest:
            raise ValueError("retrieved output SHA256 mismatch: " + str(remote))
        result["retrieved"][local.relative_to(out).as_posix()] = {"sha256": digest, "bytes": local.stat().st_size}
    try:
        compiled = read_metadata(root / "report.json")
        contract = read_metadata(contract_path)
        if (compiled.get("artifact_kind") != "qwen35-existing-air-compilation" or compiled.get("platform") != "kirinx90"
                or compiled.get("requested_options", {}).get("save_weights_as_external_data") != "true"
                or compiled.get("status") != "COMPILED" or compiled.get("model_compilation") != "PASSED"
                or contract.get("compilation_report_sha256") != sha256_file(root / "report.json")
                or compiled.get("omc") not in compiled.get("generated_artifacts", [])
                or any(contract.get("omc", {}).get(k) != compiled.get("omc", {}).get(k) for k in ("bytes", "sha256"))):
            raise ValueError("actual target contract is not bound to this successful compilation")
        tool, board_root = _board_path(tool), _board_path(board_model_root)
        if bool(input_dir) == bool(host_input_dir):
            raise ValueError("provide exactly one local input directory or existing board input directory")
        digest = sha256_file(root / "report.json")
        cache = board_root / ("codex-target-" + digest[:20])
        attempt = cache / ("run-" + uuid.uuid4().hex)
        plan = tool_plan(contract, profile, gear, str(attempt / "source-inputs") if host_input_dir else input_dir)
        local_inputs = {}
        if host_input_dir:
            source_root = Path(host_input_dir).resolve()
            if not source_root.is_dir():
                raise ValueError("original local input directory is missing")
            for row in plan["inputs"]:
                name = PurePosixPath(row["source"]).name
                path = source_root / name
                if path.is_symlink() or not path.is_file() or path.stat().st_size != row["bytes"]:
                    raise ValueError("original input file size differs from actual compiled dtype/gear: " + str(path))
                before = _stat(path)
                entry = {"path": name, "bytes": before[2], "sha256": sha256_file(path), "stat": before}
                if _stat(path) != before or (name in local_inputs and local_inputs[name] != entry):
                    raise ValueError("original local input changed or has conflicting slot descriptors")
                local_inputs[name] = entry
            plan["host_input_root"] = str(source_root)
            plan["host_inputs"] = list(local_inputs.values())
        save_json(out / "plan.json", plan)
        sn, bip = os.environ.get("SN"), os.environ.get("BIP")
        if (not sn or sn.startswith("-") or not bip or not re.fullmatch(r"[^\s:]+:[0-9]+", bip)
                or not 0 < int(bip.rsplit(":", 1)[1]) < 65536):
            raise ValueError("the configured SN and HDC server BIP are required")
        hdc = Hdc(os.environ.get("HDC", "hdc"), sn, bip, out, timeout_seconds)
        result.update(stage="device-preflight", serial=sn, hdc_server=bip, gear=gear,
                      plan_sha256=sha256_file(out / "plan.json"), contract_sha256=sha256_file(contract_path))
        _, probe = hdc.shell("test -x " + shlex.quote(str(tool)) + " && command -v sha256sum && command -v find && sha256sum "
                             + shlex.quote(str(tool)) + " && param get const.ohos.fullname", "device-preflight")
        result["device_probe"] = "ONLINE_TOOL_PRESENT"
        result["device_preflight_observation"] = probe
        result.update(board_model_cache=str(cache), board_attempt=str(attempt), stage="model-cache")
        qcache = shlex.quote(str(cache))
        owner = shlex.quote(str(cache / ".compilation-sha256"))
        hdc.shell("mkdir -p " + shlex.quote(str(board_root)) + " && if mkdir " + qcache + "; then printf '%s\\n' " + shlex.quote(digest) + " > " + owner
                  + "; else test -d " + qcache + " && test \"$(cat " + owner + ")\" = " + shlex.quote(digest) + "; fi", "own-model-cache")
        uploads = []
        for row in compiled.get("generated_artifacts", []):
            path = _recorded_file(root, row, "stat")
            relative = PurePosixPath(row["path"]).relative_to("compile-work/generated")
            remote = cache / "payload" / relative
            _board_path(str(remote))
            uploads.append((row, path, remote))
        if not uploads or len(uploads) > 8192:
            raise ValueError("target generated payload inventory is missing or excessive")
        hdc.shell("mkdir -p " + " ".join(shlex.quote(str(p)) for p in sorted({r.parent for _, _, r in uploads})), "model-directories")
        manifest = out / "model-files.tsv"
        manifest.write_text("".join(row["sha256"] + " " + str(remote) + "\n" for row, _, remote in uploads))
        hdc.send(manifest, cache / "model-files.tsv")
        _, text = hdc.shell("while read -r expected path; do if test -f \"$path\" && test \"$(sha256sum \"$path\" | cut -d ' ' -f 1)\" = \"$expected\"; then printf 'OK %s\\n' \"$path\"; else printf 'SEND %s\\n' \"$path\"; fi; done < "
                            + shlex.quote(str(cache / "model-files.tsv")), "inspect-model-cache")
        decisions = {}
        for line in text.splitlines():
            match = re.fullmatch(r"(OK|SEND) (/[A-Za-z0-9_./-]+)", line.strip())
            if not match or match[2] in decisions:
                raise ValueError("incomplete or ambiguous remote cache inventory")
            decisions[match[2]] = match[1]
        if set(decisions) != {str(r) for _, _, r in uploads}:
            raise ValueError("remote model cache inventory does not cover every compiler file")
        for row, local, remote in uploads:
            if decisions[str(remote)] == "SEND":
                print("target-board: uploading " + remote.name + " (" + str(row["bytes"]) + " bytes)", flush=True)
                hdc.send(local, remote)
                _, hashed = hdc.shell("sha256sum " + shlex.quote(str(remote)), "verify-upload")
                if _hash_lines(hashed, remote.parent).get(remote.name) != row["sha256"]:
                    raise ValueError("target model upload checksum mismatch")
        result["model_payloads"] = [{"path": str(remote), "sha256": row["sha256"], "bytes": row["bytes"],
                                     "upload": decisions[str(remote)]} for row, _, remote in uploads]
        result["stage"] = "input-snapshot"
        hdc.shell("mkdir " + shlex.quote(str(attempt)) + " && mkdir " + shlex.quote(str(attempt / "inputs"))
                  + " " + shlex.quote(str(attempt / "outputs")) + " " + shlex.quote(str(attempt / "source-inputs")), "attempt-directories")
        for row in local_inputs.values():
            source = _recorded_file(source_root, row, "stat")
            hdc.send(source, attempt / "source-inputs" / row["path"])
        input_script = out / "prepare-inputs.sh"
        lines = ["#!/system/bin/sh", "set -eu"]
        for row in plan["inputs"]:
            source = shlex.quote(row["source"])
            dest = shlex.quote(str(attempt / row["file"]))
            lines.extend(["test -f " + source, "test \"$(wc -c < " + source + ")\" -eq " + str(row["bytes"]),
                          "cp " + source + " " + dest, "sha256sum " + source, "sha256sum " + dest])
        input_script.write_text("\n".join(lines) + "\n")
        hdc.send(input_script, attempt / "prepare-inputs.sh")
        _, input_text = hdc.shell("sh " + shlex.quote(str(attempt / "prepare-inputs.sh")), "prepare-inputs")
        input_hashes = []
        for line in input_text.splitlines():
            match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line.rstrip("\r"))
            if match is None:
                raise ValueError("unexpected private input checksum response")
            input_hashes.append(match.groups())
        if len(input_hashes) != 178:
            raise ValueError("not every original input and private copy produced a checksum")
        result["inputs"] = []
        for index, row in enumerate(plan["inputs"]):
            before, copied = input_hashes[2 * index:2 * index + 2]
            if (before[1] != row["source"] or copied[1] != str(attempt / row["file"]) or before[0] != copied[0]):
                raise ValueError("original input changed while copying to its private runtime slot")
            if local_inputs and before[0] != local_inputs[PurePosixPath(row["source"]).name]["sha256"]:
                raise ValueError("original host input differs from the uploaded board bytes")
            result["inputs"].append(dict(row, sha256=before[0]))
        save_json(out / "input-evidence.json", {"inputs": result["inputs"]})
        primary = cache / "payload" / PurePosixPath(compiled["omc"]["path"]).relative_to("compile-work/generated")
        argv = [str(tool), "--model=" + str(primary), "--input=" + ",".join(str(attempt / row["file"]) for row in plan["inputs"]),
                "--times=1", "--modelmanager_itf=V2", "--weight_dir=" + str(primary.parent), "--enable_item=1",
                "--input_shape=" + plan["input_shape"], "--output_shape=" + plan["output_shape"],
                "--output_dir=" + str(attempt / "outputs")]
        save_json(out / "command.json", {"argv": argv, "tool_sha256_observation": probe})
        command = ("cd " + shlex.quote(str(attempt)) + " && LD_LIBRARY_PATH=" + shlex.quote(str(tool.parent))
                   + " " + shlex.join(argv) + " > tool.stdout 2> tool.stderr; rc=$?; printf '%s\\n' \"$rc\" > "
                   + shlex.quote(str(attempt / "tool.exitcode")) + "; rmdir "
                   + shlex.quote(str(cache / ".execution-lock")) + "; exit \"$rc\"")
        hdc.shell("mkdir " + shlex.quote(str(cache / ".execution-lock")), "acquire-execution-lock")
        result["remote_execution_lock"] = str(cache / ".execution-lock")
        result["timeout_scope"] = "HDC wait only; remote cancellation unverified. Retained lock prevents automatic overlapping execution."
        result.update(stage="execute", device_execution="ATTEMPTED")
        attempted = True
        print(f"target-board: original 40-layer target, gear={gear}, HIAI V2, one tool invocation", flush=True)
        status, _ = hdc.shell(command, "execute-original-target", allow_failure=True)
        result["remote_exitcode"] = status
        if status != 0:
            raise RuntimeError("original target tool exited with status " + str(status))
        result["stage"] = "retrieve-outputs"
        _, inventory = hdc.shell("find " + shlex.quote(str(attempt / "outputs")) + " -type f -exec sha256sum {} \\;", "output-inventory")
        hashes = _hash_lines(inventory, attempt / "outputs")
        if not hashes or len(hashes) > 1024:
            raise ValueError("target tool produced no outputs or exceeded the retrieval budget")
        for relative, digest in hashes.items():
            remote = attempt / "outputs" / relative
            local = out / "outputs" / relative
            local.parent.mkdir(parents=True, exist_ok=True)
            hdc.receive(remote, local)
            if sha256_file(local) != digest:
                raise ValueError("target output retrieval hash mismatch: " + relative)
            result["retrieved"]["outputs/" + relative] = {"sha256": digest, "bytes": local.stat().st_size}
        result.update(status="EXECUTED_UNVALIDATED", device_execution="EXECUTED_OUTPUTS_RETRIEVED",
                      state_continuity="NOT_VALIDATED_SINGLE_TOOL_INVOCATION", output_mapping="UNVERIFIED_TOOL_FILENAMES",
                      native_run_count="UNVERIFIED_TOOL_INTERNAL_WARMUP_POLICY", performance="NOT_MEASURED",
                      target_features="NOT_EXPOSED_BY_USER_81_OUTPUT_CONTRACT")
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        if attempted:
            result["device_execution"] = "FAILED"
    finally:
        if attempted:
            for name in ("tool.stdout", "tool.stderr", "tool.exitcode"):
                try:
                    collect(attempt / name, out / name, name.replace(".", "-"))
                except Exception as error:
                    result["retrieval_errors"].append({"path": name, "error": f"{type(error).__name__}: {error}"})
            try:
                saved = (out / "tool.exitcode").read_text().strip()
                if not re.fullmatch(r"0|[1-9][0-9]{0,2}", saved) or int(saved) != result.get("remote_exitcode"):
                    raise ValueError("retrieved target exit code differs from the verified invocation")
                if result["retrieval_errors"]:
                    raise ValueError("target execution log retrieval is incomplete")
            except Exception as error:
                result.update(status="FAILED", device_execution="FAILED", evidence_error=f"{type(error).__name__}: {error}")
        result["hdc_invocations"] = hdc.count if hdc else 0
        save_json(out / "result.json", result)
    return result
