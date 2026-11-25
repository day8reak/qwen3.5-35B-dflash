#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/../../.."
tool=targets/cpro/tools/gdn_layer_compare.py
action=${1:-plan}
if (( $# )); then shift; fi
case "$action" in
    -h|--help) echo "78: bash $0 air   |   35: bash $0 board"
        exec python "$tool" --help;;
    plan) exec python "$tool" "$action" "$@";;
    air)
        if (( $# )); then echo "Usage: bash $0 air (set GDN_LAYER_RUN to choose the output directory)" >&2; exit 2; fi
        bash targets/cpro/scripts/split_prefill_t64.sh air --mode linear
        action=prepare;;
    prepare|board|compare) ;;
    *) echo "Usage: bash $0 {air|plan|prepare|board|compare}" >&2; exit 2;;
esac
tests="${AI_RUN_DIR:?}/target/tests"
args=(--run "${GDN_LAYER_RUN:-$tests/l0-gdn-ops-softpluspatch-t64-valid19-01}" --l0-run "${L0_SPLIT_RUN:-$tests/l0-split-softpluspatch-t64-valid19-01}")
env_name=custom_pypto
case "$action" in
    prepare) env_name=qwen3_5_w4
        args+=(--source "${TARGET_SOURCE:?}" --config "${NZ_CONFIG:-config/qwen3_5_moe.yaml}");;
    board) args+=(--ddk "${CPRO_DDK_ROOT:?}" --runtime-config "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}"
                  --diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}");;
esac
if [[ ${CONDA_DEFAULT_ENV:-} == "$env_name" ]]; then exec python -u "$tool" "$action" "${args[@]}" "$@"; fi
exec conda run --no-capture-output -n "$env_name" python -u "$tool" "$action" "${args[@]}" "$@"
