#!/usr/bin/env bash
# Prepare the L0 conv_out fixture on 78; run only F.silu(conv_out) on 35.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || ( ${1:-} != prepare && ${1:-} != board ) ]]; then
    echo "Usage: bash conv_silu.sh prepare [fixture-directory] | board [existing-run-directory]" >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$script_dir/../../.." && pwd)"
export REPO_DIR
tool="$REPO_DIR/targets/cpro/tools/attention_step.py"
tests="${AI_RUN_DIR:?}/target/tests"
fixture="$tests/l0-qkv-conv-silu-t64-valid19-01"

in_env() {
    local env_name="$1"
    shift
    if [[ "${CONDA_DEFAULT_ENV:-}" == "$env_name" ]]; then
        "$@"
    else
        conda run --no-capture-output -n "$env_name" "$@"
    fi
}

if [[ $1 == prepare ]]; then
    fixture="${2:-$fixture}"
    source_air="${ATTENTION_AIR:-$tests/attention4-t64-valid19-v1024-qlate-01/air}"
    if [[ ! -e "$fixture/air" && -z "${ATTENTION_AIR:-}" &&
          ( ! -f "$source_air/manifest.json" || ! -f "$source_air/models/l00_c00_gdn_layer_hidden_o0/export.air" ) ]]; then
        original="$tests/attention4-source-01"
        in_env qwen3_5_w4 bash "$script_dir/check_attention4_tmatrix.sh" capture "$original"
        in_env qwen3_5_cpro bash "$script_dir/check_attention4_tmatrix.sh" air "$original"
        source_air="$original/air"
    fi
    in_env qwen3_5_w4 python "$tool" prepare-qkv --reuse --run "$fixture" \
        --source-air "$source_air" --source-root "${TARGET_SOURCE:?}" \
        --config "${NZ_CONFIG:-config/qwen3_5_moe.yaml}"
else
    ddk="${CPRO_DDK_ROOT:?}"
    if [[ $# == 2 ]]; then
        run="$(cd -- "$2" && pwd)"
    else
        mkdir -p "$tests"
        run="$(mktemp -d "$tests/l0-conv-silu-board-XXXXXXXX")"
    fi
    echo "SiLU board run: $run"
    in_env custom_pypto python "$tool" board --run "$run" \
        --remote-air "${AIR_ON_78:-$fixture/air}" --ddk-root "$ddk" \
        --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}" \
        --runtime-config "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}"
fi
