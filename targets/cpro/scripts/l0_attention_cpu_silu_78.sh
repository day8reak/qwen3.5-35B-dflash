#!/usr/bin/env bash
# Stage only the complete L0 linear attention, its inputs, weights and 310P out_proj golden.
set -euo pipefail
export REPO_DIR="$(cd -- "$(dirname -- "$0")/../../.." && pwd)"
export QWEN35_TRACE_COMMANDS=1
set -x
tests="${AI_RUN_DIR:?}/target/tests"
dest="${L0_SILU_AIR:-$tests/l0-attention-cpu-silu-t64-valid19-01/air}"
source_air="${ATTENTION_AIR:-$tests/attention4-t64-valid19-v1024-qlate-01/air}"
casefile="models/l00_c00_gdn_layer_hidden_o0/export.air"
in_env() {
    local env_name="$1"
    shift
    if [[ "${CONDA_DEFAULT_ENV:-}" == "$env_name" ]]; then "$@"
    else conda run --no-capture-output -n "$env_name" "$@"; fi
}
if (( $# > 1 )) || [[ $# == 1 && "$1" != --stage-matrices ]]; then
    echo "Usage: $0 [--stage-matrices]" >&2
    exit 2
fi
if [[ "${1:-}" == --stage-matrices && -z "${ATTENTION_AIR:-}" && -s "$dest/manifest.json" ]]; then
    source_air="$dest"
fi
if [[ -z "${ATTENTION_AIR:-}" && ! -s "$source_air/$casefile" ]]; then
    original="$tests/attention4-source-01"
    if [[ -s "$dest/manifest.json" ]]; then
        source_air="$dest"
    else
        source_air="$original/air"
        if [[ ! -s "$source_air/$casefile" ]]; then
            in_env qwen3_5_w4 bash -x "$REPO_DIR/targets/cpro/scripts/check_attention4_tmatrix.sh" capture "$original"
            in_env qwen3_5_cpro bash -x "$REPO_DIR/targets/cpro/scripts/check_attention4_tmatrix.sh" air "$original"
        fi
    fi
fi
in_env qwen3_5_cpro python -u "$REPO_DIR/targets/cpro/tools/l0_attention_cpu_silu.py" prepare "$source_air" "$dest"
if [[ "${1:-}" == --stage-matrices ]]; then
    for backend in 310p cpu; do
        in_env qwen3_5_w4 python -u "$REPO_DIR/targets/cpro/tools/l0_attention_cpu_silu.py" stage-reference \
            "$backend" "$dest" "${L0_SILU_REFS:-$tests/l0-silu-stage-refs-01}" "${TARGET_SOURCE:?}" \
            "${NZ_CONFIG:-config/qwen3_5_moe.yaml}"
    done
fi
