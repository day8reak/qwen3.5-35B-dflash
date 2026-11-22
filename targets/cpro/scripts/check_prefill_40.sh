#!/usr/bin/env bash
# Use the 310P, AIR and DDK conda environments for reference, air and board respectively.
set -euo pipefail
repo="${REPO_DIR:?}"
run="${PREFILL40_RUN:-${AI_RUN_DIR:?}/target/tests/prefill-40l-t64-valid19-v1024-01}"
tool="$repo/targets/cpro/tools/prefill_40_layers.py"

case "${1:-}" in
  reference)
    python "$tool" capture --source-root "${TARGET_SOURCE:?}" --config "${TARGET_CONFIG:?}" \
      --baseline "${PREFILL40_BASELINE:-$AI_RUN_DIR/target/tests/gdr-stages-t64-valid19-fresh01/reference}" \
      --output "$run/reference"
    ;;
  air)
    python "$tool" export --source-root "${CPRO_SOURCE:?}" --config "${TARGET_CONFIG:?}" \
      --reference "$run/reference" --output "$run/air"
    ;;
  compile|board)
    python "$tool" compile --run "$run" --ddk-root "${CPRO_DDK_ROOT:?}" --ddk-id "${CPRO_DDK_ID:-}" \
      --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}"
    if [[ "$1" == compile ]]; then exit 0; fi
    python "$tool" run --compiled "$run/omc" --ddk-root "${CPRO_DDK_ROOT:?}" \
      --config "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}" --output "$run/board"
    ;;
  *) echo 'usage: bash check_prefill_40.sh reference|air|compile|board' >&2; exit 2 ;;
esac
