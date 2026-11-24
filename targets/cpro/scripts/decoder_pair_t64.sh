#!/usr/bin/env bash
# Two independent complete native-SiLU prefill decoders on measured 310P inputs.
set -euo pipefail
if ! [[ ( $# == 1 && ( "$1" == air || "$1" == board ) ) || ( $# == 2 && ( "$1" == single-omc || "$1" == full-chain ) ) ]]; then
    echo "Usage: bash $0 {air|board}  OR  bash $0 {single-omc|full-chain} <decoder-pair-prefill-board-run>" >&2
    exit 2
fi
export REPO_DIR="$(cd -- "$(dirname -- "$0")/../../.." && pwd)"
export QWEN35_TRACE_COMMANDS=1
cd "$REPO_DIR"
tests="${AI_RUN_DIR:?}/target/tests"
fixture="$tests/decoder-pair-prefill-t64-valid19"
tool=targets/cpro/tools/decoder_pair.py
transfer=targets/cpro/scripts/pull_artifacts.py
selection="$fixture/selected-air.txt"
in_env() {
    local env_name="$1"
    shift
    if [[ "${CONDA_DEFAULT_ENV:-}" == "$env_name" ]]; then "$@"
    else conda run --no-capture-output -n "$env_name" "$@"; fi
}
set -x
if [[ "$1" == single-omc ]]; then
    : "${SN:?}" "${BIP:?}"
    parent="$(realpath -e -- "$2")"
    for layer in L0 L3; do test -s "$parent/$layer/omc/manifest.json"; done
    run="$(mktemp -d "$parent/single-omc-XXXXXXXX")"
    echo "Single OMC results: $run/comparison.json"
    in_env custom_pypto python -u "$tool" single-omc --parent "$parent" --run "$run" \
        --runtime-config "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}"
elif [[ "$1" == full-chain ]]; then
    : "${CPRO_DDK_ROOT:?}" "${SN:?}" "${BIP:?}"
    parent="$(realpath -e -- "$2")"
    for layer in L0 L3; do test -s "$parent/$layer/air/manifest.json"; done
    run="$(mktemp -d "$parent/complete-decoders-XXXXXXXX")"
    echo "Complete decoder results: $run/comparison.json"
    in_env custom_pypto python -u "$tool" full-chain --parent "$parent" --run "$run" \
        --runtime-config "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}" \
        --ddk-root "$CPRO_DDK_ROOT" --gdr-source "${GDR_SOURCE:-$CPRO_DDK_ROOT/ddk/ascendc/ops/impl/custom/chunk_gated_delta_rule.cpp}" \
        --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}"
elif [[ "$1" == air ]]; then
    mkdir -p "$fixture"
    air="${DECODER_AIR_ON_78:-}"
    if [[ -z "$air" ]]; then
        previous=""
        if [[ -s "$selection" ]]; then IFS= read -r previous < "$selection"; fi
        for candidate in "$previous" "$tests/decoder-l0-l3-t64-valid19-v1024-qlate-01/air" \
                "$tests/prefill-40l-t64-valid19-v1024-01/air"; do
            if [[ -n "$candidate" && -f "$candidate/manifest.json" ]] && in_env qwen3_5_cpro python "$tool" check-air "$candidate"; then
                air="$candidate"
                break
            fi
        done
        if [[ -z "$air" ]]; then
            : "${TARGET_SOURCE:?}"
            ref="${DECODER_REFERENCE:-}"
            if [[ -z "$ref" ]]; then
                for candidate in "$tests/prefill-40l-t64-valid19-v1024-01/reference" \
                        "$tests/prefill40-single-t64-valid19-v1024-qlate-01/reference"; do
                    if [[ -s "$candidate/manifest.json" ]] && in_env qwen3_5_cpro python -c \
                        'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("status") != "CAPTURED")' "$candidate/manifest.json"; then
                        ref="$candidate"
                        break
                    fi
                done
            fi
            job="$(mktemp -d "$fixture/export-XXXXXXXX")"
            opts=(--source-root "$TARGET_SOURCE" --config "${NZ_CONFIG:-config/qwen3_5_moe.yaml}" --run "$job")
            if [[ -z "$ref" ]]; then
                in_env qwen3_5_w4 python -u targets/cpro/tools/prefill_40_single.py capture "${opts[@]}"
                ref="$job/reference"
            fi
            in_env qwen3_5_cpro python -u "$tool" export "${opts[@]}" --reference "$ref"
            air="$job/air"
        fi
    fi
    air="$(realpath -- "$air")"
    in_env qwen3_5_cpro python "$tool" check-air "$air"
    tmp="$(mktemp "$fixture/.selected-XXXXXXXX")"
    printf '%s\n' "$air" > "$tmp"
    mv -f -- "$tmp" "$selection"
    in_env qwen3_5_cpro python "$transfer" --prepare "$selection"
else
    : "${CPRO_DDK_ROOT:?}" "${SN:?}" "${BIP:?}"
    mkdir -p "$tests"
    run="$(mktemp -d "$tests/decoder-pair-prefill-board-XXXXXXXX")"
    echo "Decoder results: $run/comparison.json"
    air="${DECODER_AIR_ON_78:-}"
    if [[ -z "$air" ]]; then
        in_env custom_pypto python "$transfer" --from air "$selection" --to "$run/selected-air.txt"
        IFS= read -r air < "$run/selected-air.txt"
    fi
    in_env custom_pypto python -u "$tool" run --run "$run" --remote-air "$air" \
        --ddk-root "$CPRO_DDK_ROOT" --gdr-source "${GDR_SOURCE:-$CPRO_DDK_ROOT/ddk/ascendc/ops/impl/custom/chunk_gated_delta_rule.cpp}" \
        --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}"
fi
