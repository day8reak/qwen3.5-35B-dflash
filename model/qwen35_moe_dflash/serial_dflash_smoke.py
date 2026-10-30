"""One bounded entry point for the existing decode1 and S64 draft artifacts.

Prompt ingestion uses the same decode1 model from zero state, followed by
serial greedy verification. Numerical acceptance and acceleration are separate.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
import shutil
import struct

from .artifacts import output_path, save_json
from .incremental_smoke import SmokeContract, run_incremental_smoke
from .original_embedding_lookup import OriginalEmbeddingLookup, _current


def _backends():
    from .original_target_smoke_backend import OriginalTargetSmokeBackend
    from .existing_draft_smoke_backend import ExistingDraftSmokeBackend
    return OriginalTargetSmokeBackend, ExistingDraftSmokeBackend


def _prompt(path, length):
    if type(length) is not int or not 1 <= length <= 63:
        raise ValueError("prompt length must be1..63 explicit tokens")
    path, before = _current(path)
    if not 4 * length <= before["bytes"] <= 16384 or before["bytes"] % 4:
        raise ValueError("prompt source must contain enough little-endian INT32 token IDs")
    raw = path.read_bytes()
    if _current(path)[1] != before:
        raise ValueError("prompt source changed during read")
    values = list(struct.unpack("<" + str(length) + "i", raw[:4 * length]))
    if any(not 0 <= token < 248320 for token in values):
        raise ValueError("prompt contains an out-of-vocabulary token ID")
    return values, {"path": str(path), "bytes": before["bytes"], "sha256": hashlib.sha256(raw).hexdigest(),
        "identity": before, "selected_prefix_tokens": length,
        "policy": "NEW_SHORT_PROMPT_FROM_EXPLICIT_SOURCE_ID_PREFIX_NOT_CONTINUATION_OF_OLD64_RUN",
        "tokenizer_provenance": "USER_SUPPLIED_NOT_REVALIDATED"}


def run_serial_dflash_smoke(*, decode_job, runtime_probe, state_lineage, source_torchair_pb2,
        draft_bundle, weight_table, scale_table, prompt_input_ids, output,
        prompt_length=8, max_new_tokens=8, max_proposals_per_round=2,
        minimum_draft_rounds=2, apply=False, board_model_root="/data/local/tmp/qwen3_5_moe",
        timeout_seconds=7200, lock_wait_seconds=1800):
    """Prepare, or execute with --apply, the unchanged complete model pair."""
    out = output_path(output)
    out.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "artifact_kind": "qwen35-existing-models-serial-dflash-job",
        "status": "FAILED", "stage": "host-preflight", "mode": "APPLY" if apply else "PLAN_ONLY",
        "output": str(out), "device_execution": "NOT_RUN", "model_compilation": "NOT_RUN",
        "numerical_validation": "NOT_EVALUATED_BY_EXECUTION_MILESTONE", "performance": "NOT_EVALUATED",
        "execution_scope": "SHORT_PROMPT_DECODE1_BOOTSTRAP_AND_S64_DRAFT_SERIAL_VERIFY",
        "target_control_semantics": "EXPERIMENTAL_NOT_NUMERICALLY_VALIDATED",
        "automatic_retry": "NONE_RETAIN_FAILED_ATTEMPT_AND_LOCK_EVIDENCE"}
    target = draft = None
    try:
        if (type(apply) is not bool or any(type(value) is not int or value < 1 for value in
                (max_new_tokens, max_proposals_per_round, minimum_draft_rounds, timeout_seconds))
                or not 1 <= max_proposals_per_round <= 15
                or type(lock_wait_seconds) is not int or not 0 <= lock_wait_seconds <= 7200
                or timeout_seconds > 7200 or prompt_length + max_new_tokens > 64):
            raise ValueError("serial smoke requires positive bounded options and complete prefix<=64")
        prompt, source = _prompt(prompt_input_ids, prompt_length)
        # Retain all step inputs and outputs. This is a conservative file-space
        # estimate, not a memory estimate or a performance assertion.
        calls = prompt_length + max_new_tokens
        required = calls * (160 << 20) + (256 << 20)
        free = shutil.disk_usage(out).free
        if free < required:
            raise ValueError(f"retained serial steps need about{required} host bytes; available{free}")
        lookup = OriginalEmbeddingLookup(weight_table, scale_table)
        Target, Draft = _backends()
        target = Target(decode_job, runtime_probe, state_lineage, lookup, out / "target",
            source_torchair_pb2=source_torchair_pb2, board_model_root=board_model_root,
            timeout_seconds=timeout_seconds, lock_wait_seconds=lock_wait_seconds)
        target_preflight = target.preflight(prompt)
        draft = Draft(draft_bundle, out / "draft", runtime_version=target.runtime_version,
            firmware_id=target.firmware_id, timeout_seconds=timeout_seconds,
            lock_wait_seconds=lock_wait_seconds)
        plan = {"schema_version": 1, "artifact_kind": "qwen35-existing-models-serial-dflash-plan",
            "prompt_token_ids": prompt, "prompt_source": source,
            "max_new_tokens": max_new_tokens, "max_proposals_per_round": max_proposals_per_round,
            "minimum_draft_rounds": minimum_draft_rounds, "sequence_capacity": 64,
            "target": target_preflight, "draft": draft.preflight(),
            "models": {"target": asdict(target.identity), "draft": asdict(draft.identity)},
            "embedding_tables": lookup.initialization,
            "host_disk_budget": {"observed_free_bytes": free, "estimated_retained_bytes": required},
            "target_prefill": "PROMPT_TOKENS_SEQUENTIALLY_ADVANCED_FROM80_ZERO_STATES_WITH_SAME_DECODE1_MODEL",
            "draft_policy": "FULL_PREFIX_RECOMPUTE_RIGHT_PADDED_TO64_ORIGINAL15_CANDIDATES",
            "verification_policy": "GREEDY_COMPARE_BEFORE_ADVANCE_COMMIT_ONLY_TARGET_SELECTED_TOKENS",
            "eos_policy": "DISABLED_FOR_BOUNDED_EXECUTION_MILESTONE",
            "model_compilation": "NOT_RUN", "accuracy_gate": "DEFERRED_BY_USER_DIRECTIVE"}
        if _current(prompt_input_ids)[1] != source["identity"]:
            raise ValueError("prompt source changed during backend setup")
        save_json(out / "plan.json", plan)
        result["plan"] = str(out / "plan.json")
        if not apply:
            result.update(status="DFLASH_SERIAL_SMOKE_PLANNED", stage="complete",
                next="Use the same arguments with a separate new --output and --apply")
        else:
            result["stage"] = "execute-serial-loop"
            print(f"dflash: unchanged models; ingest{len(prompt)} prompt tokens with decode1, "
                  f"generate{max_new_tokens}, verify up to{max_proposals_per_round} candidates per round", flush=True)
            loop = run_incremental_smoke(target, draft, prompt, contract=SmokeContract(),
                max_new_tokens=max_new_tokens, max_proposals_per_round=max_proposals_per_round,
                minimum_draft_rounds=minimum_draft_rounds, eos_token_ids=(), target_next_anchor=True)
            save_json(out / "loop.json", loop)
            summaries = {"target": target.execution_summary(), "draft": draft.execution_summary()}
            result.update(loop=str(out / "loop.json"), generated_token_ids=loop["generated_token_ids"],
                counts=loop["counts"], native_executions=summaries,
                counts_scope="PROTOCOL_OBSERVATIONS_PREFILL_IS_COMPOSITE_NATIVE_COUNTS_SEPARATE",
                multi_round_observed=loop["multi_round_observed"],
                final_state_sha256=loop["final_state_sha256"],
                final_feature_prefix_sha256=loop["final_feature_prefix_sha256"],
                acceptance_rate=(loop["counts"]["accepted_proposals"] /
                    max(1, loop["counts"]["accepted_proposals"] + loop["counts"]["mismatched_proposals"])),
                acceptance_rate_scope="ONLY_CONSIDERED_CANDIDATES_UP_TO_K_NOT_ALL15_NATIVE_PROPOSALS")
            if (any(row.get("validated_native_calls") is not True for row in summaries.values())
                    or summaries["target"].get("native_calls") != len(prompt) + len(loop["generated_token_ids"])
                    or summaries["draft"].get("native_calls") != loop["counts"]["draft_calls"]
                    or loop["evidence_kinds"] != ["device-hiai"]):
                raise ValueError("actual native execution receipts or counts do not cover the entire loop")
            result.update(status=("DFLASH_SERIAL_LOOP_EXECUTED_UNVALIDATED" if loop["multi_round_observed"]
                                  else "DFLASH_SERIAL_LOOP_INCOMPLETE_MILESTONE"),
                stage="complete", device_execution="EXECUTED_OUTPUTS_RETRIEVED",
                receipt_authenticity="ADAPTER_VALIDATED_NATIVE_INPUT_OUTPUT_HASHES",
                state_continuity="ACTUAL_COMMITTED_TOKEN_BYTE_CHAIN_CONTROL_SEMANTICS_UNVALIDATED",
                next="Preserve loop/step outputs; numerical and ordinary-baseline validation follow this execution milestone")
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        for name, backend in (("target", target), ("draft", draft)):
            if backend is not None:
                try:
                    result.setdefault("native_executions", {})[name] = backend.execution_summary()
                except Exception as summary_error:
                    result[name + "_summary_error"] = f"{type(summary_error).__name__}: {summary_error}"
        if apply and result["stage"] == "execute-serial-loop":
            result["device_execution"] = "INCOMPLETE_REVIEW_RETAINED_STEP_RECEIPTS"
        result["next"] = "Inspect the retained failed step; do not automatically rerun model processes or remove locks"
    save_json(out / "result.json", result)
    return result
