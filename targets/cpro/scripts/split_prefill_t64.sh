#!/usr/bin/env bash
# 78 prepares AIR, 310P and CPU references. 53 compiles, runs and compares.
set -euo pipefail
repo=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
cd "$repo"
usage() {
    echo "Usage: bash $0 {air|cpu|board|compare} --mode {linear|full|40|last4} [--run DIR] [--capture DIR]"
    echo "       bash $0 diagnose STAGE_DIR [STAGE_DIR ...]"
    echo "       bash $0 diagnose --saved DIAGNOSTICS_DIR"
    echo "       board [--recovery [REPLAY_DIR]] [--from-stage STAGE]; compare [--replay DIR]"
    echo "       board --mode {40|last4} [--per-layer-omc]; last4 tests CPU and 310P L35 exits on CPro L36-L39"
    echo "       board --mode linear [--embed-decoder-hidden]"
    echo "       board [--split-decoder-projections | --split-decoder-residual] (default: --split-decoder-residual)"
}
in_env() {
    local name="$1"
    shift
    if [[ ${CONDA_DEFAULT_ENV:-} == "$name" ]]; then "$@"
    else conda run --no-capture-output -n "$name" "$@"; fi
}
action=${1:-}
if [[ $action == diagnose ]]; then
    shift
    in_env custom_pypto python -u targets/cpro/tools/split_stage_log.py "$@"
    exit
fi
case "$action" in --help|-h) usage; exit 0;; air|cpu|board|compare) shift;; *) usage >&2; exit 2;; esac
mode= run= capture= replay= restart= embed_hidden= split_variant= per_layer=
recovery=()
while (( $# )); do
    case "$1" in
        --mode) mode=${2:?missing mode}; shift 2;;
        --run) run=${2:?missing directory}; shift 2;;
        --capture) capture=${2:?missing capture}; shift 2;;
        --replay) replay=${2:?missing replay}; shift 2;;
        --from-stage) restart=${2:?missing stage}; shift 2;;
        --per-layer-omc) per_layer=1; shift;;
        --embed-decoder-hidden) embed_hidden=1; shift;;
        --split-decoder-projections|--split-decoder-residual)
            if [[ -n "$split_variant" && "$split_variant" != "$1" ]]; then usage >&2; exit 2; fi
            split_variant=$1; shift;;
        --recovery)
            recovery=(--recovery); shift
            if (( $# )) && [[ $1 != --* ]]; then recovery+=("$1"); shift; fi;;
        *) usage >&2; exit 2;;
    esac
done
if [[ -n "$per_layer" && ( $action != board || ( $mode != 40 && $mode != last4 ) ) ]]; then usage >&2; exit 2; fi
if [[ -n "$embed_hidden" && ( $action != board || $mode != linear ) ]]; then usage >&2; exit 2; fi
if [[ -n "$split_variant" && ( $action != board || -n "$embed_hidden" ) ]]; then usage >&2; exit 2; fi
if [[ $action != board && ( ${#recovery[@]} != 0 || -n "$restart" ) ]] ||
   [[ -n "$restart" && ${#recovery[@]} == 0 ]] || [[ -n "$replay" && $action != compare ]] ||
   [[ -n "$capture" && $action != air && $action != cpu ]]; then usage >&2; exit 2; fi
if [[ -n "$per_layer" ]]; then recovery+=(--per-layer-omc); fi
if [[ -n "$split_variant" ]]; then recovery+=("$split_variant"); fi
if [[ -n "$run" ]]; then run=$(realpath -m -- "$run"); fi
if [[ -n "$capture" ]]; then capture=$(realpath -e -- "$capture"); fi
tests="${AI_RUN_DIR:?}/target/tests"
export QWEN35_TRACE_COMMANDS=1
case "$mode" in
    linear) run="${run:-${L0_SPLIT_RUN:-$tests/l0-split-silupatch-t64-valid19-02}}";;
    last4) run="${run:-$tests/prefill-split-40-t64-valid19-02}";;
    full|40) run="${run:-$tests/prefill-split-$mode-t64-valid19-02}";;
    *) usage >&2; exit 2;;
esac
cpu_reference() {
    local args=(--mode "$mode" --run "$run")
    if [[ -n "$capture" ]]; then args+=(--capture "$capture"); fi
    if [[ -n ${TARGET_SOURCE:-} ]]; then args+=(--source "$TARGET_SOURCE"); fi
    in_env qwen3_5_w4 env PYTHONPATH="$repo/model${PYTHONPATH:+:$PYTHONPATH}" python -u -m \
        qwen35_moe_dflash.target.prefill_t64.split_cpu_reference "${args[@]}"
}
if [[ $action == cpu ]]; then
    if [[ $mode == linear ]]; then capture="${capture:-${L0_SPLIT_CAPTURE:-}}"; fi
    cpu_reference
    exit
fi
if [[ $mode == linear ]]; then
    export L0_SPLIT_RUN="$run"
    if [[ -n "$capture" ]]; then export L0_SPLIT_CAPTURE="$capture"; fi
    if [[ -n "$replay" ]]; then
        case "$replay" in /*) export L0_SPLIT_REPLAY="$replay";; *) export L0_SPLIT_REPLAY="$L0_SPLIT_RUN/$replay";; esac
    fi
    if [[ -n "$restart" ]]; then recovery+=(--from-stage "${restart#L00/}"); fi
    if [[ -n "$embed_hidden" ]]; then recovery+=(--embed-decoder-hidden); fi
    exec bash targets/cpro/scripts/l0_split_silu_t64.sh "$action" "${recovery[@]}"
fi
tool="$repo/targets/cpro/tools/split_prefill.py"
args=("$action" --mode "$mode" --run "$run")
case "$action" in
    air)
        mkdir -p "$run"
        if [[ $mode == last4 && -z "$capture" && ( -s "$run/L36/air/manifest.json" || -s "$run/L35/cpu/manifest.json" ) ]]; then
            capture=$(python - "$run" <<'PY_CAPTURE'
import hashlib, json, pathlib, sys
run = pathlib.Path(sys.argv[1])
air, cpu = run/'L36/air/manifest.json', run/'L35/cpu/manifest.json'
needed = (json.loads(air.read_bytes())['reference'] if air.is_file() else json.loads(cpu.read_bytes()))['capture_sha256']
for path in run.parent.glob('*/capture-*/310p/manifest.json'):
    if hashlib.sha256(path.read_bytes()).hexdigest() == needed:
        print(path.parent)
        break
else:
    raise SystemExit(f'Pass --capture DIR for the existing 310P capture {needed}')
PY_CAPTURE
)
        fi
        if [[ -z "$capture" ]]; then
            candidates=("$run"/capture-*/310p "$tests/prefill-split-$mode-t64-valid19-01"/capture-*/310p)
            if [[ $mode == last4 ]]; then
                candidates+=("$tests"/prefill-split-40-*/capture-*/310p)
            fi
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
        in_env qwen3_5_cpro python -u "$tool" "${args[@]}"
        cpu_reference;;
    board)
        args+=(--ddk "${CPRO_DDK_ROOT:?}" --diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}"
               --runtime-config "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}" "${recovery[@]}")
        if [[ -n "$restart" ]]; then args+=(--from-stage "$restart"); fi
        if [[ -n ${SPLIT_REMOTE_RUN:-} ]]; then args+=(--remote-run "$SPLIT_REMOTE_RUN"); fi
        in_env custom_pypto python -u "$tool" "${args[@]}";;
    compare)
        if [[ -n "$replay" ]]; then args+=(--replay "$replay"); fi
        if [[ -n ${SPLIT_REMOTE_RUN:-} ]]; then args+=(--remote-run "$SPLIT_REMOTE_RUN"); fi
        in_env custom_pypto python -u "$tool" "${args[@]}";;
esac
