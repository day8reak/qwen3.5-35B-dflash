#!/usr/bin/env bash
# Compile and run both complete L0 attention models, using the board's HIAI V2 tool.
set -euo pipefail
export REPO_DIR="$(cd -- "$(dirname -- "$0")/../../.." && pwd)"
export QWEN35_TRACE_COMMANDS=1
set -x
tests="${AI_RUN_DIR:?}/target/tests"
ddk="${CPRO_DDK_ROOT:?}"
: "${SN:?}" "${BIP:?}"
mkdir -p "$tests"
run="${L0_SILU_RUN:-}"
if [[ -z "$run" && "${1:-}" == --stage-matrices ]]; then
    newest=""
    for receipt in "$tests"/l0-attention-cpu-silu-ab-*/split-silu-*/run-suffix-cpu_silu/result.json; do
        split="${receipt%/run-suffix-cpu_silu/result.json}"
        if [[ -f "$split/comparison.csv" && -f "$receipt" && "$receipt" -nt "$newest" ]]; then
            newest="$receipt"
            run="${split%/*}"
        fi
    done
fi
if [[ -z "$run" ]]; then run="$(mktemp -d "$tests/l0-attention-cpu-silu-ab-XXXXXXXX")"; fi
echo "L0 attention A/B: $run"
env_cmd=()
if [[ "${CONDA_DEFAULT_ENV:-}" != custom_pypto ]]; then env_cmd=(conda run --no-capture-output -n custom_pypto); fi
"${env_cmd[@]}" python -u "$REPO_DIR/targets/cpro/tools/l0_attention_cpu_silu.py" board \
    "$run" "${AIR_ON_78:-$tests/l0-attention-cpu-silu-t64-valid19-01/air}" "$ddk" \
    "${GDR_SOURCE:-$ddk/ddk/ascendc/ops/impl/custom/chunk_gated_delta_rule.cpp}" \
    "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}" "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}" "$@"
