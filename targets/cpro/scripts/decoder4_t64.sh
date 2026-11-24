#!/usr/bin/env bash
# Export on 78, then compile and run one four-layer OMC on 53.
set -euo pipefail
export REPO_DIR="$(cd -- "$(dirname -- "$0")/../../.." && pwd)"
export QWEN35_TRACE_COMMANDS=1
cd "$REPO_DIR"
tests="${AI_RUN_DIR:?}/target/tests"
run="${DECODER4_RUN:-$tests/decoder4-single-t64-valid19-01}"
tool=targets/cpro/tools/decoder4_prefill.py
original="${ORIGINAL4_RUN:-$tests/decoder4-original-t64-01}"
golden="${ORIGINAL4_GOLDEN:-$original/golden}"
compare_tool=targets/cpro/tools/compare_decoder4.py
in_env() {
    local name="$1"
    shift
    if [[ "${CONDA_DEFAULT_ENV:-}" == "$name" ]]; then "$@"
    else conda run --no-capture-output -n "$name" "$@"; fi
}
is_four_capture() {
    [[ -s "$1/manifest.json" ]] && python -c \
        'import json,sys; m=json.load(open(sys.argv[1])); sys.exit(m.get("status") != "CAPTURED" or m.get("layers") != [0,1,2,3] or "four_layer_top1" not in m)' \
        "$1/manifest.json"
}
golden_78() {
    local capture="${ORIGINAL4_CAPTURE:-}"
    if [[ -s "$golden/manifest.json" && -z "${ORIGINAL4_CAPTURE:-}" ]]; then return; fi
    if [[ -z "$capture" ]]; then
        local dir
        for dir in "$tests/prefill-4l-t64-valid19" "$tests"/gdr-stages-t64-valid19*/reference "$tests"/attention4-t64-valid19*/reference; do
            if is_four_capture "$dir"; then capture="$dir"; break; fi
        done
    fi
    if [[ -z "$capture" || ! -s "$capture/manifest.json" ]]; then
        if [[ -n "${ORIGINAL4_CAPTURE:-}" ]]; then echo "Missing 310P capture at $capture" >&2; return 1; fi
        mkdir -p "$original"
        local work
        work="$(mktemp -d "$original/capture-XXXXXXXX")"
        capture="$work/310p"
        local nz="${TARGET_SOURCE:?set TARGET_SOURCE to the 310P source containing inference_segment.py}"
        local cfg="${NZ_CONFIG:-config/qwen3_5_moe.yaml}"
        (cd "$nz" && in_env qwen3_5_w4 python -u "$REPO_DIR/targets/cpro/tools/check_segment_prefill.py" \
            --source-root "$PWD" --config "$cfg" --output "$capture" --capture-only)
    fi
    in_env qwen3_5_w4 python -u "$compare_tool" golden --capture "$capture" --output "$golden"
}
compare_53() {
    local local_ref="$golden"
    if [[ ! -d "$local_ref" && -z "${ORIGINAL4_GOLDEN:-}${ORIGINAL4_REMOTE_GOLDEN:-}" ]] && is_four_capture "$tests/prefill-4l-t64-valid19"; then
        local_ref="$tests/prefill-4l-t64-valid19"
    fi
    in_env custom_pypto python -u "$compare_tool" compare --original "${ORIGINAL4_REPLAY:-$original}" \
        --golden "$local_ref" --remote-golden "${ORIGINAL4_REMOTE_GOLDEN:-$golden}" "$@"
}
set -x
variant=()
if (( $# > 1 )); then
    if [[ $# != 2 || ! $1 =~ ^original-(air|compile|board|compare|log)$ || $2 != --silu-patch ]]; then
        echo 'Use --silu-patch with original-air, original-compile, original-board, original-compare or original-log.' >&2
        exit 2
    fi
    variant=(--silu-patch)
fi
case "${1:-}" in
    aligned-diagnose|aligned-diagnose-report|aligned-diagnose-next)
        args=(--run "${ALIGNED4_RUN:-${original}-aligned-silu-patch}")
        if [[ "$1" == aligned-diagnose-report ]]; then
            args+=(--report)
        else
            args+=(--ddk "${CPRO_DDK_ROOT:?}" --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}")
            if [[ "$1" == aligned-diagnose-next ]]; then args+=(--consumer); fi
        fi
        if [[ "$1" != aligned-diagnose && -n "${SILU_DIAGNOSIS:-}" ]]; then
            args+=(--diagnostic-dir "$SILU_DIAGNOSIS")
        fi
        in_env custom_pypto python -u targets/cpro/tools/diagnose_silu_format.py \
            "${args[@]}"
        ;;
    original-golden) golden_78 ;;
    aligned-air|aligned-compile|aligned-board|aligned-compare|aligned-log|original-air|original-compile|original-board|original-compare|original-log)
        run="$original"
        remote_air="${ORIGINAL4_REMOTE_AIR:-$run/air}"
        if [[ $1 == aligned-* ]]; then
            variant=(--aligned)
            run="${ALIGNED4_RUN:-${original}-aligned-silu-patch}"
            remote_air="${ALIGNED4_REMOTE_AIR:-$run/air}"
        elif (( ${#variant[@]} )); then
            run="${run}-silu-patch"
            remote_air="${SILU_PATCH_REMOTE_AIR:-$run/air}"
        fi
        tool=targets/cpro/tools/decoder4_original.py
        result_run="$run"
        if [[ $1 == aligned-* ]]; then result_run="$run/no-platform"; fi
        action="${1#*-}"
        if [[ "$action" == log ]]; then
            in_env custom_pypto python -u "$tool" log --run "$run" "${variant[@]}"
        elif [[ "$action" == compare ]]; then
            args=()
            if [[ ! -d "$result_run" ]]; then result_run="$run"; fi
            if (( ${#variant[@]} )); then args+=(--modified "${MODIFIED4_REPLAY:-$result_run}"); fi
            compare_53 "${args[@]}"
        elif [[ "$action" == air ]] && (( ${#variant[@]} )); then
            args=(--baseline "$original")
            if [[ -n "${CPRO_SOURCE:-}" ]]; then args+=(--source "$CPRO_SOURCE"); fi
            if [[ -n "${TARGET_CONFIG:-}" ]]; then args+=(--config "$TARGET_CONFIG"); fi
            if [[ -n "${ORIGINAL4_CONV_CONTROL:-}" ]]; then args+=(--conv-control "$ORIGINAL4_CONV_CONTROL"); fi
            in_env qwen3_5_cpro python -u "$tool" air --run "$run" "${args[@]}" "${variant[@]}"
            golden_78
        elif [[ "$action" == air ]]; then
            ref="${ORIGINAL4_REFERENCE:-${DECODER_REFERENCE:-${PREFILL40_REFERENCE:-}}}"
            if [[ -z "$ref" ]]; then
                for dir in "$tests/prefill-40l-t64-valid19-v1024-01" "$tests/prefill40-single-t64-valid19-v1024-qlate-01" \
                           "${PREFILL40_RUN:-$tests/prefill-40l-t64-valid19-v1024-qlate-01}" "$original"; do
                    if [[ -s "$dir/reference/manifest.json" ]] && python -c \
                        'import json,sys; m=json.load(open(sys.argv[1])); sys.exit((m.get("reference"), m.get("status")) != ("MEASURED_310P_40_DECODER_BLOCKS_T64_VALID19", "CAPTURED"))' \
                        "$dir/reference/manifest.json"; then ref="$dir/reference"; break; fi
                done
            fi
            if [[ -z "$ref" ]]; then
                in_env qwen3_5_w4 python -u targets/cpro/tools/prefill_40_single.py capture --run "$run" \
                    --source-root "${TARGET_SOURCE:?set to the 310P source for reference capture}" \
                    --config "${NZ_CONFIG:-config/qwen3_5_moe.yaml}"
                ref="$run/reference"
            fi
            args=(--reference "$ref")
            if [[ -n "${ORIGINAL4_CONV_CONTROL:-}" ]]; then args+=(--conv-control "$ORIGINAL4_CONV_CONTROL"); fi
            in_env qwen3_5_cpro python -u "$tool" air --run "$run" \
                --source "${CPRO_SOURCE:-${TARGET_SOURCE:?set CPRO_SOURCE to the original project}}" \
                --config "${TARGET_CONFIG:-config/qwen3_5_moe.yaml}" "${args[@]}"
        else
            args=(--run "$run" --ddk "${CPRO_DDK_ROOT:?}" --remote-air "$remote_air"
                  --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}")
            if [[ "$action" == board ]]; then
                args+=(--runtime-config "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}")
                if (( ${#variant[@]} )); then args+=(--baseline "${ORIGINAL4_REPLAY:-$original}"); fi
                if [[ -n "${ORIGINAL_INPUT_DIR:-}" ]]; then args+=(--inputs "$ORIGINAL_INPUT_DIR"); fi
                if [[ -n "${ORIGINAL_INPUT_MAP:-}" ]]; then args+=(--input-map "$ORIGINAL_INPUT_MAP"); fi
            fi
            in_env custom_pypto python -u "$tool" "$action" "${args[@]}" "${variant[@]}"
            if [[ "$action" == board ]] && (( ${#variant[@]} )); then compare_53 --modified "$result_run"; fi
        fi
        ;;
    review|air)
        source_root="${CPRO_SOURCE:-${TARGET_SOURCE:?set CPRO_SOURCE to the private CPro source directory}}"
        review="${DECODER4_PREPARED_SOURCE:-$run/review}"
        if [[ "$1" == review ]]; then
            in_env qwen3_5_cpro python "$tool" review --source-root "$source_root" --output "$review"
            exit 0
        fi
        args=()
        if [[ -d "$review" || -n "${DECODER4_PREPARED_SOURCE:-}" ]]; then args+=(--prepared-source "$review"); fi
        ref="${DECODER_REFERENCE:-${PREFILL40_REFERENCE:-}}"
        if [[ -z "$ref" ]]; then
            for dir in "$tests/prefill-40l-t64-valid19-v1024-01" "$tests/prefill40-single-t64-valid19-v1024-qlate-01" \
                       "${PREFILL40_RUN:-$tests/prefill-40l-t64-valid19-v1024-qlate-01}" "$run"; do
                if [[ -s "$dir/reference/manifest.json" ]] && python -c \
                    'import json,sys; m=json.load(open(sys.argv[1])); sys.exit(m.get("status") != "CAPTURED" or len(m.get("blocks", ())) != 40)' \
                    "$dir/reference/manifest.json"; then ref="$dir/reference"; break; fi
            done
        fi
        if [[ -z "$ref" ]]; then
            in_env qwen3_5_w4 python -u targets/cpro/tools/prefill_40_single.py capture \
                --run "$run" --source-root "${TARGET_SOURCE:?set to the 310P source for reference capture}" \
                --config "${NZ_CONFIG:-config/qwen3_5_moe.yaml}"
            ref="$run/reference"
        fi
        in_env qwen3_5_cpro python -u "$tool" air --run "$run" --source-root "$source_root" \
            --config "${TARGET_CONFIG:-${NZ_CONFIG:-config/qwen3_5_moe.yaml}}" \
            --reference "$ref" "${args[@]}"
        ;;
    compile|board)
        if [[ ! -s "$run/air/manifest.json" ]]; then
            in_env custom_pypto python -u "$tool" pull --run "$run" --remote-air "${DECODER4_REMOTE_AIR:-$run/air}"
        fi
        if [[ ! -s "$run/omc/manifest.json" ]]; then
            in_env custom_pypto python -u "$tool" compile --run "$run" --ddk-root "${CPRO_DDK_ROOT:?}" \
                --gdr-source "${GDR_SOURCE:-$CPRO_DDK_ROOT/ddk/ascendc/ops/impl/custom/chunk_gated_delta_rule.cpp}" \
                --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}"
        fi
        if [[ "$1" == board ]]; then
            in_env custom_pypto python -u "$tool" board --run "$run"
        fi
        ;;
    *) echo "Usage: bash $0 {review|air|compile|board|original-golden|aligned-{air,compile,board,compare,log,diagnose,diagnose-report,diagnose-next}} or original-{air,compile,board,compare,log} [--silu-patch]" >&2; exit 2 ;;
esac
