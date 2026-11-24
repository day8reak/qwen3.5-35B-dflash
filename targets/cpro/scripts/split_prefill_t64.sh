#!/usr/bin/env bash
# 78 exports AIR and 310P references. 53 compiles, runs and compares.
set -euo pipefail
repo=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
cd "$repo"
usage() {
    echo "Usage: bash $0 {air|board|compare} --mode {linear|full|40} [--run DIR] [--capture DIR]"
    echo "       board [--recovery [REPLAY_DIR]] [--from-stage STAGE]; compare [--replay DIR]"
}
action=${1:-}
case "$action" in --help|-h) usage; exit 0;; air|board|compare) shift;; *) usage >&2; exit 2;; esac
mode= run= capture= replay= restart=
recovery=()
while (( $# )); do
    case "$1" in
        --mode) mode=${2:?missing mode}; shift 2;;
        --run) run=${2:?missing directory}; shift 2;;
        --capture) capture=${2:?missing capture}; shift 2;;
        --replay) replay=${2:?missing replay}; shift 2;;
        --from-stage) restart=${2:?missing stage}; shift 2;;
        --recovery)
            recovery=(--recovery); shift
            if (( $# )) && [[ $1 != --* ]]; then recovery+=("$1"); shift; fi;;
        *) usage >&2; exit 2;;
    esac
done
if [[ $action != board && ( ${#recovery[@]} != 0 || -n "$restart" ) ]] ||
   [[ -n "$restart" && ${#recovery[@]} == 0 ]] || [[ -n "$replay" && $action != compare ]] ||
   [[ -n "$capture" && $action != air ]]; then usage >&2; exit 2; fi
if [[ -n "$run" ]]; then run=$(realpath -m -- "$run"); fi
if [[ -n "$capture" ]]; then capture=$(realpath -e -- "$capture"); fi
tests="${AI_RUN_DIR:?}/target/tests"
export QWEN35_TRACE_COMMANDS=1
case "$mode" in
    linear)
        export L0_SPLIT_RUN="${run:-${L0_SPLIT_RUN:-$tests/l0-split-silupatch-t64-valid19-01}}"
        if [[ -n "$capture" ]]; then export L0_SPLIT_CAPTURE="$capture"; fi
        if [[ -n "$replay" ]]; then
            case "$replay" in /*) export L0_SPLIT_REPLAY="$replay";; *) export L0_SPLIT_REPLAY="$L0_SPLIT_RUN/$replay";; esac
        fi
        if [[ -n "$restart" ]]; then recovery+=(--from-stage "${restart#L00/}"); fi
        exec bash targets/cpro/scripts/l0_split_silu_t64.sh "$action" "${recovery[@]}";;
    full|40) run="${run:-$tests/prefill-split-$mode-t64-valid19-01}";;
    *) usage >&2; exit 2;;
esac
tool="$repo/targets/cpro/tools/split_prefill.py"
args=("$action" --mode "$mode" --run "$run")
in_env() {
    local name="$1"
    shift
    if [[ ${CONDA_DEFAULT_ENV:-} == "$name" ]]; then "$@"
    else conda run --no-capture-output -n "$name" "$@"; fi
}
case "$action" in
    air)
        mkdir -p "$run"
        if [[ -z "$capture" ]]; then
            candidates=("$run"/capture-*/310p)
            if [[ $mode == full ]]; then
                candidates+=("${L0_SPLIT_RUN:-$tests/l0-split-silupatch-t64-valid19-01}"/capture-*/310p)
            fi
            for dir in "${candidates[@]}"; do
                if [[ -s $dir/manifest.json ]] && python -c '
import json,sys
m=json.load(open(sys.argv[1]))
ok=("3" in m.get("attention_modules", {}) and "2" in m.get("boundaries", {})) if sys.argv[2]=="full" else "value_cache_39" in m.get("outputs", {})
sys.exit(not (m["status"]=="CAPTURED" and ok))' "$dir/manifest.json" "$mode"; then capture="$dir"; break; fi
            done
        fi
        if [[ -z "$capture" ]]; then
            capture=$(mktemp -d "$run/capture-XXXXXXXX")/310p
            src="${TARGET_SOURCE:?set the 310P source directory}"
            if [[ $mode == full ]]; then
                (cd "$src" && in_env qwen3_5_w4 python -u "$repo/targets/cpro/tools/check_segment_prefill.py" \
                    --source-root "$PWD" --config "${NZ_CONFIG:-config/qwen3_5_moe.yaml}" --attention-layers --capture-only --output "$capture")
            else
                in_env qwen3_5_w4 python -u "$tool" capture --mode 40 --source "$src" --run "$run" \
                    --config "${NZ_CONFIG:-config/qwen3_5_moe.yaml}" --capture "$capture"
            fi
        fi
        args+=(--capture "$capture" --baseline "${ORIGINAL4_RUN:-$tests/decoder4-original-t64-01}")
        if [[ -n ${CPRO_SOURCE:-} ]]; then args+=(--source "$CPRO_SOURCE"); fi
        if [[ -n ${TARGET_CONFIG:-} ]]; then args+=(--config "$TARGET_CONFIG"); fi
        in_env qwen3_5_cpro python -u "$tool" "${args[@]}";;
    board)
        args+=(--ddk "${CPRO_DDK_ROOT:?}" --diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}"
               --runtime-config "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}" "${recovery[@]}")
        if [[ -n "$restart" ]]; then args+=(--from-stage "$restart"); fi
        if [[ -n ${SPLIT_REMOTE_RUN:-} ]]; then args+=(--remote-run "$SPLIT_REMOTE_RUN"); fi
        in_env custom_pypto python -u "$tool" "${args[@]}";;
    compare)
        if [[ -n "$replay" ]]; then args+=(--replay "$replay"); fi
        in_env custom_pypto python -u "$tool" "${args[@]}";;
esac
