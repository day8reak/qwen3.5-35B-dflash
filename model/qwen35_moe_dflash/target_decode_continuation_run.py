"""Execute one explicitly prepared same-model decode state-feedback experiment."""
from __future__ import annotations

from pathlib import Path

from .artifacts import output_path, save_json
from .existing_air_profile import get_existing_air_profile
from .target_compile_failure import _relative, _safe
from .target_decode_output_inventory import _descriptors
from .target_prefill_output_inventory import _hash_retrieved, _logged_slots, _numerical


def inspect_continuation_outputs(plan, board_root, observed):
    """Bind the executed input hashes and all explicitly logged child outputs."""
    root = Path(board_root)
    if (observed.get("status") != "EXECUTED_UNVALIDATED" or observed.get("remote_exitcode") != 0
            or observed.get("device_execution") != "EXECUTED_OUTPUTS_RETRIEVED"
            or observed.get("retrieval_errors") != []
            or observed.get("lock_state") != "NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT"
            or observed.get("source_preservation") != "UNCHANGED_CONTINUATION_PROOF_AND_ALL89_PREPARED_INPUT_SHA256"):
        raise ValueError("child decode requires a complete successful locked V2 execution receipt")
    actual = observed.get("inputs")
    if not isinstance(actual, list) or len(actual) != 89 or len(plan["inputs"]) != 89:
        raise ValueError("child decode must report every one of89 actual input byte hashes")
    for expected, row in zip(plan["inputs"], actual):
        if (any(row.get(key) != expected.get(key) for key in
                ("index", "name", "runtime_tensor_name", "dtype", "shape", "bytes", "file", "sha256", "source_filename"))
                or row.get("host_source") != expected["source"]):
            raise ValueError("executed child input differs from its proven parent-state/original-row preparation")
    feedback = plan.get("state_feedback_bindings")
    if (not isinstance(feedback, list) or len(feedback) != 80
            or {row.get("source_refdata") for row in feedback} != {f"RefData_{i}" for i in range(80)}
            or len({row.get("runtime_input_index") for row in feedback}) != 80
            or len({row.get("runtime_output_index") for row in feedback}) != 80):
        raise ValueError("prepared continuation lacks all80 independently proven state feedback bindings")
    descriptors = _descriptors(plan)
    retrieved = observed.get("retrieved", {})
    files, log, log_identities = [], None, []
    for name, row in retrieved.items():
        if name.startswith("outputs/"):
            identity, _ = _hash_retrieved(_relative(root, name), row)
            files.append(dict(identity, path=name))
    if len(files) != 82:
        raise ValueError("child decode must retrieve exactly82 data files")
    for name in ("tool.stdout", "tool.stderr", "tool.exitcode", "tool.pid"):
        if name not in retrieved:
            raise ValueError("child decode execution log is missing: " + name)
        identity, captured = _hash_retrieved(_relative(root, name), retrieved[name], capture=True)
        log_identities.append(dict(identity, path=name))
        if name == "tool.stdout":
            log = captured
        elif name == "tool.exitcode" and captured.strip() != b"0":
            raise ValueError("child decode retained exit code is not zero")
        elif name == "tool.pid" and captured.strip() != str(observed.get("tool_process_id")).encode():
            raise ValueError("child decode retained PID differs from its execution receipt")
    mapping = _logged_slots(log, observed, descriptors, files)
    if mapping["status"] != "BOUND_TO_COMPLETE_EXPLICIT_TOOL_LOG" or mapping.get("unlogged_retrieved_files"):
        raise ValueError("child output slots lack a complete explicit tool log binding: " + str(mapping.get("reason")))
    by_path = {row["path"]: row for row in files}
    slots = {row["runtime_index"]: dict(by_path[row["path"]], **row) for row in mapping["entries"]}
    excluded = {plan["feature_runtime_output_index"], plan["logits_runtime_output_index"]}
    if {row["runtime_output_index"] for row in feedback} != set(range(82)) - excluded:
        raise ValueError("state feedback outputs must exclude the actual feature and logits slots")
    chain = []
    for binding in feedback:
        index, output_index = binding["runtime_input_index"], binding["runtime_output_index"]
        if type(index) is not int or not 0 <= index < 89 or type(output_index) is not int or not 0 <= output_index < 82:
            raise ValueError("state feedback runtime slot is outside the frozen model ABI")
        parent = binding["output_file"]
        if (actual[index]["name"] != binding["source_refdata"] or actual[index]["sha256"] != parent["sha256"]
                or actual[index]["bytes"] != parent["bytes"] or slots[output_index]["bytes"] != parent["bytes"]):
            raise ValueError("actual child input bytes differ from the individually mapped parent output")
        child = slots[output_index]
        chain.append({"source_refdata": binding["source_refdata"], "runtime_input_index": index,
            "runtime_output_index": output_index, "parent_output": parent,
            "executed_input_sha256": actual[index]["sha256"], "child_output": child,
            "child_bytes_changed": child["sha256"] != parent["sha256"]})
    tensors = {}
    for kind, key in (("logits", "logits_runtime_output_index"), ("feature", "feature_runtime_output_index")):
        index = plan[key]
        tensors[kind] = dict(slots[index], statistics=_numerical(
            _relative(root, slots[index]["path"]), descriptors[index], slots[index]))
    for row in files + log_identities:
        if _safe(_relative(root, row["path"]))[1] != row["stat"]:
            raise ValueError("child output or execution log changed during state-chain inspection")
    return {"status": "ALL80_PARENT_OUTPUT_BYTES_USED_BY_CHILD_INPUTS",
        "feedback_input_count": 80, "bound_child_output_count": 82,
        "changed_state_output_count": sum(row["child_bytes_changed"] for row in chain),
        "states": chain, "logged_slot_mapping": mapping, "bound_tensors": tensors,
        "state_continuity": "BYTE_CHAIN_OBSERVED_CONTROL_SEMANTICS_AND_NUMERICAL_CONTINUITY_NOT_VALIDATED",
        "numerical_validation": "NOT_RUN_NO_REFERENCE",
        "scope": "One same-model parent-output to executed-child-input byte chain; changed hashes and finite tensors do not establish correct control semantics, logical history, greedy sequence correctness or accuracy"}


def run_decode_continuation(prepared, output, *, apply=False, board_model_root="/data/local/tmp/qwen3_5_moe",
                            timeout_seconds=7200, lock_wait_seconds=1800):
    from .target_decode_continuation import validate_decode_continuation
    out = output_path(output)
    out.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "artifact_kind": "qwen35-target-decode-continuation-job",
        "status": "FAILED", "mode": "APPLY" if apply else "PLAN_ONLY", "stage": "validate-prepared",
        "output": str(out), "prepared": str(Path(prepared).resolve()), "device_execution": "NOT_RUN",
        "model_compilation": "NOT_RUN", "numerical_validation": "NOT_RUN_NO_REFERENCE",
        "state_continuity": "NOT_VALIDATED", "state_feedback": "NOT_RUN"}
    try:
        plan = validate_decode_continuation(prepared)
        save_json(out / "plan.json", plan)
        result.update(status="DECODE_CONTINUATION_PLANNED", plan=str(out / "plan.json"),
            input_count=89, output_count=82, execution_scope=plan["execution_scope"],
            next="Use this same prepared directory with a separate new output and --apply for one experimental continuation")
        if apply:
            from .original_target_board import execute_target_tool
            compilation = Path(plan["compilation"])
            observed = execute_target_tool(compilation, compilation.parent / "target-contract/report.json", out / "board",
                profile=get_existing_air_profile("cpro-40block-decode-1"), gear=1,
                host_input_dir=plan["input_dir"], target_features=True,
                runtime_continuation_prepared=prepared, tool=plan["tool_binding"]["path"],
                board_model_root=board_model_root, timeout_seconds=timeout_seconds, lock_wait_seconds=lock_wait_seconds)
            result.update(observed, artifact_kind="qwen35-target-decode-continuation-job", output=str(out),
                mode="APPLY", prepared=str(Path(prepared).resolve()), plan=str(out / "plan.json"),
                board_result=str(out / "board/result.json"))
            if observed["status"] != "EXECUTED_UNVALIDATED":
                raise RuntimeError(observed.get("error", "child decode did not complete"))
            result["stage"] = "bind-child-state-chain"
            chain = inspect_continuation_outputs(plan, out / "board", observed)
            save_json(out / "state-chain.json", chain)
            if validate_decode_continuation(prepared) != plan:
                raise ValueError("continuation preparation changed during child-output inspection")
            result.update(status="DECODE_CONTINUATION_EXECUTED_UNVALIDATED", stage="complete",
                state_feedback=chain["status"], state_continuity=chain["state_continuity"],
                feedback_input_count=chain["feedback_input_count"], bound_child_output_count=chain["bound_child_output_count"],
                state_chain=str(out / "state-chain.json"),
                logits=chain["bound_tensors"]["logits"]["statistics"], feature=chain["bound_tensors"]["feature"]["statistics"],
                next="Retain this experimental byte-chain observation; validate original control semantics and ordinary decode history before numerical or multi-round DFlash acceptance")
    except Exception as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
    save_json(out / "result.json", result)
    return result
