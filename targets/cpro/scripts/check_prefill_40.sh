#!/usr/bin/env bash
# Use the 310P, AIR and DDK conda environments for reference, air and board respectively.
set -euo pipefail
repo="${REPO_DIR:?}"
run="${PREFILL40_RUN:-${AI_RUN_DIR:?}/target/tests/prefill-40l-t64-valid19-v1024-01}"
tool="$repo/targets/cpro/tools/prefill_40_layers.py"

case "${1:-}" in
  split-air)
    old="$AI_RUN_DIR/target/tests/prefill-40l-t64-valid19-v1024-01"
    single="$AI_RUN_DIR/target/tests/prefill40-single-t64-valid19-v1024-qlate-01"
    air="${PREFILL40_AIR:-$old/air}"
    if [[ ! -f "$air/manifest.json" ]]; then
      ref="${PREFILL40_REFERENCE:-$old/reference}"
      if [[ ! -f "$ref/manifest.json" && -z "${PREFILL40_REFERENCE:-}" ]]; then
        ref="$single/reference"
      fi
      if [[ ! -f "$ref/manifest.json" && -z "${PREFILL40_REFERENCE:-}" ]]; then
        capture_run="${PREFILL40_RUN:-$AI_RUN_DIR/target/tests/prefill-40l-t64-valid19-v1024-qlate-01}"
        ref="$capture_run/reference"
        if [[ ! -f "$ref/manifest.json" ]]; then
          conda run --no-capture-output -n qwen3_5_w4 python "$repo/targets/cpro/tools/prefill_40_single.py" \
            capture --run "$capture_run" --source-root "${TARGET_SOURCE:?}" --config "${NZ_CONFIG:-config/qwen3_5_moe.yaml}"
        fi
      fi
      python "$tool" export --source-root "${CPRO_SOURCE:?}" --config "${TARGET_CONFIG:-config/qwen3_5_moe.yaml}" \
        --reference "$ref" --output "$air"
    fi
    python "$tool" check-air --air "$air"
    python "$repo/targets/cpro/scripts/pull_artifacts.py" --prepare "$air"
    ;;
  compile-run)
    run="${PREFILL40_RUN:-$AI_RUN_DIR/target/tests/prefill-40l-t64-valid19-v1024-qlate-01}"
    air="${PREFILL40_AIR:-$AI_RUN_DIR/target/tests/prefill-40l-t64-valid19-v1024-01/air}"
    single="$AI_RUN_DIR/target/tests/prefill40-single-t64-valid19-v1024-qlate-01"
    # Match the kernel used for the measured whole-model logits, including on resume.
    digest="${PREFILL40_GDR_SHA256:-}"
    if [[ -z "$digest" ]]; then
      receipt="$single/board/attempt.json"
      if [[ ! -f "$receipt" ]]; then receipt="$single/omc/manifest.json"; fi
      if [[ -f "$receipt" ]]; then
        digest=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["gdr_source_sha256"])' "$receipt")
        echo "Using GDR SHA-256 from $receipt: $digest"
      else
        gdr="${GDR_SOURCE:?set to the installed allops GDR source when the whole-model receipt is unavailable}"
        read -r digest _ < <(sha256sum -- "$gdr")
        echo "Previous GDR receipt unavailable. Pinning current allops source $gdr: $digest"
      fi
    fi
    python "$tool" compile-run --run "$run" --remote-air "$air" --gdr-sha256 "$digest" \
      --config "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}" --ddk-root "${CPRO_DDK_ROOT:?}" \
      --ddk-id "${CPRO_DDK_ID:-100.520.020.010}" --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}"
    ;;
  reference)
    python "$tool" capture --source-root "${TARGET_SOURCE:?}" --config "${TARGET_CONFIG:?}" \
      --baseline "${PREFILL40_BASELINE:-$AI_RUN_DIR/target/tests/gdr-stages-t64-valid19-fresh01/reference}" \
      --output "$run/reference"
    ;;
  air)
    python "$tool" export --source-root "${CPRO_SOURCE:?}" --config "${TARGET_CONFIG:?}" \
      --reference "$run/reference" --output "$run/air"
    ;;
  compile|board)
    python "$tool" compile --run "$run" --ddk-root "${CPRO_DDK_ROOT:?}" --ddk-id "${CPRO_DDK_ID:-}" \
      --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}"
    if [[ "$1" == compile ]]; then exit 0; fi
    python "$tool" run --compiled "$run/omc" --ddk-root "${CPRO_DDK_ROOT:?}" \
      --config "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}" --output "$run/board"
    ;;
  *) echo 'usage: bash check_prefill_40.sh reference|air|compile|board|split-air|compile-run' >&2; exit 2 ;;
esac
