#!/usr/bin/env bash
# On 78 capture the real 310P residuals. On 53 reuse the two previously failing decoder exports.
set -euo pipefail
export REPO_DIR="$(cd -- "$(dirname -- "$0")/../../.." && pwd)"
export QWEN35_TRACE_COMMANDS=1
cd "$REPO_DIR"
tests="${AI_RUN_DIR:?}/target/tests"
tool=targets/cpro/tools/decoder_pair.py
transfer=targets/cpro/scripts/pull_artifacts.py
selection="$tests/decoder-tail-t64-valid19-selected.txt"
in_env() {
    local env_name="$1"
    shift
    if [[ "${CONDA_DEFAULT_ENV:-}" == "$env_name" ]]; then "$@"
    else conda run --no-capture-output -n "$env_name" "$@"; fi
}

set -x
if [[ $# == 1 && $1 == capture ]]; then
    : "${TARGET_SOURCE:?}"
    mkdir -p "$tests"
    job="$(mktemp -d "$tests/decoder-tail-capture-XXXXXXXX")"
    in_env qwen3_5_w4 python -u "$tool" tail-capture --run "$job/reference" \
        --source-root "$TARGET_SOURCE" --config "${NZ_CONFIG:-config/qwen3_5_moe.yaml}"
    in_env qwen3_5_w4 python "$transfer" --prepare "$job/reference"
    selected="$(mktemp "$tests/.tail-selection-XXXXXXXX")"
    printf '%s\n' "$job/reference" > "$selected"
    in_env qwen3_5_w4 python "$transfer" --prepare "$selected"
    mv -f -- "$selected" "$selection"
elif [[ ( $# == 2 || $# == 3 ) && $1 == projections ]]; then
    : "${CPRO_DDK_ROOT:?}" "${SN:?}" "${BIP:?}"
    parent="$(realpath -e -- "$2")"
    layer="${3:-0}"
    [[ "$layer" == 0 || "$layer" == 3 ]] || { echo 'Select layer 0 or 3' >&2; exit 2; }
    test -f "$parent/L$layer/air/manifest.json"
    job="$(mktemp -d "$parent/L$layer-projections-XXXXXXXX")"
    echo "Projection results: $job/projections.json"
    in_env custom_pypto python -u "$tool" tail-projections --run "$job" --parent "$parent" --layer "$layer" \
        --ddk-root "$CPRO_DDK_ROOT" --gdr-source "${GDR_SOURCE:-$CPRO_DDK_ROOT/ddk/ascendc/ops/impl/custom/chunk_gated_delta_rule.cpp}" \
        --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}"
elif [[ $# == 2 && $1 == board ]]; then
    : "${CPRO_DDK_ROOT:?}" "${SN:?}" "${BIP:?}"
    parent="$(realpath -e -- "$2")"
    for layer in L0 L3; do test -f "$parent/$layer/air/manifest.json"; done
    job="$(mktemp -d "$parent/decoder-tail-XXXXXXXX")"
    echo "Decoder tail results: $job/comparison.json"
    in_env custom_pypto python "$transfer" --from air "$selection" --to "$job/selected.txt"
    IFS= read -r ref < "$job/selected.txt"
    in_env custom_pypto python "$transfer" --from air "$ref" --to "$job/reference"
    in_env custom_pypto python -u "$tool" tail-run --run "$job" --parent "$parent" --reference "$job/reference" \
        --ddk-root "$CPRO_DDK_ROOT" --gdr-source "${GDR_SOURCE:-$CPRO_DDK_ROOT/ddk/ascendc/ops/impl/custom/chunk_gated_delta_rule.cpp}" \
        --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}"
else
    echo "Usage on 78: bash $0 capture" >&2
    echo "Usage on 53: bash $0 board <decoder-pair-prefill-board-run>" >&2
    echo "Usage on 53: bash $0 projections <decoder-tail-run> [0|3]" >&2
    exit 2
fi
