#!/usr/bin/env python3
"""Prepare real target features for one full draft step; --apply runs it once."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--inspection", help="completed target prefill output inspection; host preparation only")
    source.add_argument("--prepared", help="previously prepared real-feature draft directory; requires --apply")
    parser.add_argument("--draft-bundle", help="unchanged successful full draft bundle; required with --inspection")
    parser.add_argument("--output", required=True, help="new evidence directory below AI_RUN_DIR")
    parser.add_argument("--apply", action="store_true", help="run one previously prepared draft step on the board")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--lock-wait-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    if args.inspection and (not args.draft_bundle or args.apply):
        parser.error("--inspection requires --draft-bundle and prepares host inputs only; use --prepared with --apply afterwards")
    if args.prepared and (not args.apply or args.draft_bundle):
        parser.error("--prepared requires --apply and cannot use --draft-bundle")
    if not 1 <= args.timeout_seconds <= 7200 or not 0 <= args.lock_wait_seconds <= 7200:
        parser.error("timeout must be 1..7200 seconds and lock wait 0..7200 seconds")
    try:
        if args.inspection:
            from qwen35_moe_dflash.draft.real_feature_draft import prepare_real_feature_draft
            result = prepare_real_feature_draft(args.inspection, args.draft_bundle, args.output)
        else:
            from qwen35_moe_dflash.draft.real_feature_draft_board import execute_real_feature_draft
            result = execute_real_feature_draft(args.prepared, args.output,
                timeout_seconds=args.timeout_seconds, lock_wait_seconds=args.lock_wait_seconds)
        fields = ("status", "stage", "mode", "output", "prepared", "device_execution", "model_compilation",
            "numerical_validation", "state_continuity", "execution_scope", "sequence_length", "valid_length",
            "anchor_position", "anchor_token_id", "proposal_positions", "feature_source", "model_source",
            "model_cache", "missing_model_upload_bytes", "remote_exitcode", "native_process_id", "lock_state",
            "draft_top1", "hidden_statistics", "output_numerics", "error", "source_error", "evidence_error", "next")
        shown = {key: result[key] for key in fields if key in result}
        token = result.get("token_provenance", {})
        shown.update({key: token[key] for key in ("valid_length", "anchor_position", "anchor_token_id") if key in token})
        if not args.apply:
            shown["proposal_positions"] = [64, 78]
            shown["feature_source"] = result.get("feature_provenance", {}).get("source")
            shown["next"] = "After REAL_FEATURE_DRAFT_STEP_PREPARED, run --prepared with a separate new --output and --apply; no model recompilation"
        shown["report"] = str(Path(args.output).resolve() / ("result.json" if args.apply else "manifest.json"))
        print(json.dumps(shown, ensure_ascii=False, allow_nan=False), flush=True)
        success = result.get("status") in {"REAL_FEATURE_DRAFT_STEP_PREPARED", "REAL_FEATURE_DRAFT_EXECUTED_UNVALIDATED"}
        if args.apply and not success:
            from qwen35_moe_dflash.common.tool_logs import log_excerpt
            for name in ("draft.stderr", "draft.stdout", "tool.stderr", "tool.stdout"):
                path = Path(args.output) / name
                if path.is_file():
                    excerpt = log_excerpt(path)
                    if excerpt:
                        print(f"real-feature-draft: {name}\n{excerpt}", flush=True)
        return 0 if success else 1
    except Exception as error:
        print(json.dumps({"status": "FAILED", "device_execution": "NOT_RUN_OR_UNVERIFIED",
            "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
