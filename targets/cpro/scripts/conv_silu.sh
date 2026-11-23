#!/usr/bin/env bash
# Select one attention step from the original T64 / valid19 AIR.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: conv_silu.sh prepare|board|extract|compile [--step NAME] [--layer 0|3] [run-directory]
       conv_silu.sh list [--layer 0|3]
For an FP32 CPUCL SiLU control, add --cpu-silu to board.
Default: L0 qkv_conv_silu. "list" shows which steps have automatic three-backend fixtures.
78: prepare data for a matrix, or extract just the selected AIR.
35: board runs the matrix, or compile builds just the selected OMC.
For incomplete input shapes, append --input EDGE DTYPE DIMS to extract.
EOF
}

action="${1:-}"
case "$action" in
    --help|-h) usage; exit 0 ;;
    prepare|board|extract|compile|list) shift ;;
    *) usage >&2; exit 2 ;;
esac
step=qkv_conv_silu
layer=
directory=
inputs=()
cpu_silu=()
while (( $# )); do
    case "$1" in
        --cpu-silu) cpu_silu=(--cpu-silu); shift ;;
        --step) step="${2:?missing step name}"; shift 2 ;;
        --layer) layer="${2:?missing layer}"; shift 2 ;;
        --input) inputs+=("$1" "${2:?missing edge}" "${3:?missing dtype}" "${4:?missing shape}"); shift 4 ;;
        --help|-h) usage; exit 0 ;;
        -*) usage >&2; exit 2 ;;
        *) if [[ -n "$directory" ]]; then usage >&2; exit 2; fi
           directory="$1"; shift ;;
    esac
done
case "$layer" in ''|0|3) ;; *) echo 'Select --layer 0 or 3' >&2; exit 2 ;; esac
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$script_dir/../../.." && pwd)"
export REPO_DIR
table="$REPO_DIR/specs/attention-step-presets.csv"
if (( ${#cpu_silu[@]} )) && [[ "$action" != board || "${layer:-0}" != 0 ||
    ( "$step" != qkv_conv_silu && "$step" != gate_silu && "$step" != silu_output ) ]]; then
    echo '--cpu-silu requires board and an L0 SiLU step' >&2
    exit 2
fi
if [[ "$action" == list ]]; then
    awk -F, -v layer="$layer" 'NR>1 && (layer=="" || $1==layer) {
        print "L" $1, $2, ($0 ~ /,prepare \/ board,/ ? "[prepare/board]" : "[extract/compile]"), $3
    }' "$table"
    exit 0
fi
layer="${layer:-0}"
# Accept the older matrix output labels as aliases for the L0 catalog entries.
if [[ "$layer" == 0 ]]; then
    case "$step" in rms_output) step=gated_rmsnorm ;; silu_output) step=gate_silu ;; out_proj_only) step=out_proj ;; esac
fi
mode="$(awk -F, -v layer="$layer" -v name="$step" '$1==layer && $2==name { print ($0 ~ /,prepare \/ board,/ ? "matrix" : "slice") }' "$table")"
if [[ -z "$mode" ]]; then echo "Unknown L$layer step '$step'. See: $0 list --layer $layer" >&2; exit 2; fi
if [[ "$mode" != matrix && ( "$action" == prepare || "$action" == board ) ]]; then
    echo "L$layer $step has no automatic three-backend fixture. Use $0 extract --layer $layer --step $step on 78 and compile on 35." >&2
    exit 2
fi
if (( ${#inputs[@]} )) && [[ "$action" != extract ]]; then
    echo '--input is used with extract' >&2
    exit 2
fi

tool="$REPO_DIR/targets/cpro/tools/attention_step.py"
tests="${AI_RUN_DIR:?}/target/tests"
tag="l$layer-${step//_/-}"
fixture="$tests/$tag-t64-valid19-01"
in_env() {
    local env_name="$1"
    shift
    if [[ "${CONDA_DEFAULT_ENV:-}" == "$env_name" ]]; then
        "$@"
    else
        conda run --no-capture-output -n "$env_name" "$@"
    fi
}

if [[ "$action" == prepare || "$action" == extract ]]; then
    fixture="${directory:-$fixture}"
    source_air="${ATTENTION_AIR:-$tests/attention4-t64-valid19-v1024-qlate-01/air}"
    source_case=l00_c00_gdn_layer_hidden_o0
    if [[ "$layer" == 3 ]]; then source_case=l03_c00_full_attention_layer_hidden_o0; fi
    if [[ -z "${ATTENTION_AIR:-}" && ( "$action" == extract || ! -d "$fixture/air" ) &&
          ( ! -f "$source_air/manifest.json" || ! -f "$source_air/models/$source_case/export.air" ) ]]; then
        original="$tests/attention4-source-01"
        in_env qwen3_5_w4 bash "$script_dir/check_attention4_tmatrix.sh" capture "$original"
        in_env qwen3_5_cpro bash "$script_dir/check_attention4_tmatrix.sh" air "$original"
        source_air="$original/air"
    fi
    if [[ "$action" == extract ]]; then
        in_env qwen3_5_cpro python "$tool" extract --layer "$layer" --step "$step" --run "$fixture" \
            --source-air "$source_air" "${inputs[@]}"
    else
        in_env qwen3_5_w4 python "$tool" prepare --step "$step" --reuse --run "$fixture" \
            --source-air "$source_air" --source-root "${TARGET_SOURCE:?}" --config "${NZ_CONFIG:-config/qwen3_5_moe.yaml}"
    fi
else
    ddk="${CPRO_DDK_ROOT:?}"
    if [[ -n "$directory" ]]; then
        run="$(cd -- "$directory" && pwd)"
    else
        mkdir -p "$tests"
        run="$(mktemp -d "$tests/$tag-$action-XXXXXXXX")"
    fi
    echo "L$layer $step $action run: $run"
    common=(--step "$step" --run "$run" --ddk-root "$ddk" --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}")
    if [[ "$action" == board ]]; then
        in_env custom_pypto python "$tool" board "${common[@]}" "${cpu_silu[@]}" --remote-air "${AIR_ON_78:-$fixture/air}" \
            --runtime-config "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}"
    else
        in_env custom_pypto python "$tool" compile "${common[@]}" --layer "$layer" --remote-slice "${SLICE_ON_78:-$fixture/slice}"
    fi
fi
