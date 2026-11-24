#!/usr/bin/env bash
# Export on 78, then compile and run one four-layer OMC on 53.
set -euo pipefail
export REPO_DIR="$(cd -- "$(dirname -- "$0")/../../.." && pwd)"
export QWEN35_TRACE_COMMANDS=1
cd "$REPO_DIR"
tests="${AI_RUN_DIR:?}/target/tests"
run="${DECODER4_RUN:-$tests/decoder4-single-t64-valid19-01}"
tool=targets/cpro/tools/decoder4_prefill.py
in_env() {
    local name="$1"
    shift
    if [[ "${CONDA_DEFAULT_ENV:-}" == "$name" ]]; then "$@"
    else conda run --no-capture-output -n "$name" "$@"; fi
}
set -x
case "${1:-}" in
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
            in_env custom_pypto python -u "$tool" board --run "$run" \
                --runtime-config "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}"
        fi
        ;;
    *) echo "Usage: bash $0 {review|air|compile|board}" >&2; exit 2 ;;
esac
