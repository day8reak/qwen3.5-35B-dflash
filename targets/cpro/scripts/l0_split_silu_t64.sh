#!/usr/bin/env bash
# 78: air (310P golden + original CPro one-layer export). 53: board (compile, replay and compare).
set -euo pipefail
repo=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
cd "$repo"
export QWEN35_TRACE_COMMANDS=1
tests="${AI_RUN_DIR:?}/target/tests"
run="${L0_SPLIT_RUN:-$tests/l0-split-silupatch-t64-valid19-02}"
baseline="${ORIGINAL4_RUN:-$AI_RUN_DIR/target/tests/decoder4-original-t64-01}"
tool="$repo/targets/cpro/tools/l0_silu_omcs.py"
in_env() {
    local env_name="$1"
    shift
    if [[ "${CONDA_DEFAULT_ENV:-}" == "$env_name" ]]; then "$@"
    else conda run --no-capture-output -n "$env_name" "$@"; fi
}
case "${1:-}" in
    air)
        mkdir -p "$run"
        capture="${L0_SPLIT_CAPTURE:-}"
        if [[ -z "$capture" ]]; then
            for f in "$run"/capture-*/310p/manifest.json "$tests"/l0-split-silupatch-t64-valid19-01/capture-*/310p/manifest.json; do
                if [[ -s "$f" ]] && python -c 'import json,sys; m=json.load(open(sys.argv[1])); sys.exit(m.get("status") != "CAPTURED" or "l0_logits" not in m)' "$f"; then
                    capture="${f%/manifest.json}"; break
                fi
            done
        fi
        if [[ -z "$capture" ]]; then
            capture="$(mktemp -d "$run/capture-XXXXXXXX")/310p"
            (cd "${TARGET_SOURCE:?set TARGET_SOURCE to the 310P source with inference_segment.py}" &&
                in_env qwen3_5_w4 python -u "$repo/targets/cpro/tools/check_segment_prefill.py" \
                    --source-root "$PWD" --config "${NZ_CONFIG:-config/qwen3_5_moe.yaml}" \
                    --l0-readout --capture-only --output "$capture")
        fi
        args=(--baseline "$baseline" --capture "$capture")
        if [[ -n "${CPRO_SOURCE:-}" ]]; then args+=(--source "$CPRO_SOURCE"); fi
        if [[ -n "${TARGET_CONFIG:-}" ]]; then args+=(--config "$TARGET_CONFIG"); fi
        in_env qwen3_5_cpro python -u "$tool" air --run "$run" "${args[@]}"
        ;;
    board)
        in_env custom_pypto python -u "$tool" board --run "$run" --ddk "${CPRO_DDK_ROOT:?}" \
            --remote-air "${L0_SPLIT_REMOTE_AIR:-$run/air}" \
            --diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}" \
            --runtime-config "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}" "${@:2}"
        ;;
    compare)
        args=()
        if [[ -n "${L0_SPLIT_REPLAY:-}" ]]; then args+=(--replay "$L0_SPLIT_REPLAY"); fi
        in_env custom_pypto python -u "$tool" compare --run "$run" "${args[@]}"
        ;;
    *) echo "Usage: bash $0 {air|board|compare}" >&2; exit 2 ;;
esac
