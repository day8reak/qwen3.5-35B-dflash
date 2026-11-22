#!/usr/bin/env bash
# Run capture and air in the two export-host environments, compile and run on the DDK host.
set -euo pipefail
tools="${REPO_DIR:?}/targets/cpro/tools"
run="${2:-${AI_RUN_DIR:?}/target/tests/l0-gdr-io-t64-valid19-01}"

retrieved() {
    python - "$run/recovery/result.json" "$run/execution" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
report = json.loads(p.read_text()) if p.is_file() else {}
cases = {'l00_c00_gdn_layer_hidden_o0', 'l00_c00_gdr_prefill_stages_o0'}
sys.exit(0 if (report.get('device_execution') == 'EXECUTED_OUTPUTS_RETRIEVED'
               and report.get('coverage_complete') and set(report.get('validated_cases', [])) == cases
               and report.get('recovery_source') == str(Path(sys.argv[2]).resolve())) else 1)
PY
}

case "${1:-}" in
  capture)
    python "$tools/check_segment_prefill.py" --source-root "${TARGET_SOURCE:?}" \
      --config "${NZ_CONFIG:-config/qwen3_5_moe.yaml}" --gdr-stages --output "$run/reference"
    ;;
  air)
    python - "$run/reference/manifest.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as stream:
    capture = json.load(stream)
if capture['input_ids_i32_sha256'] != 'ec5119e4b49de3c9e0cd48b7cff7faf13b18a0a10dda89c2a5180b194f05eaf7':
    sys.exit('The padded token IDs differ from the original valid19 prompt. Check the 310P prompt/config.')
PY
    python "$tools/prepare_cpro_prefill_t64.py" --source-root "${CPRO_SOURCE:?}" \
      --config "${TARGET_CONFIG:-config/qwen3_5_moe.yaml}" --fixture "$run/reference" \
      --attention-layers --output "$run/export" --export
    python "$tools/gdr_prefill_stages.py" prepare --baseline "$run/export" \
      --reference "$run/reference" --l0-gdr-only --output "$run/air"
    ;;
  compile)
    ddk="${CPRO_DDK_ROOT:?}"
    op=chunk_gated_delta_rule.cpp
    src="$ddk/tools/tools_ascendc/custom_op/ascend310b/src/$op"
    impl="$ddk/ddk/ascendc/ops/impl/custom/$op"
    if [[ -e "$run/omc" ]]; then
        echo "OMC output already exists at $run/omc. Use run for a completed compilation." >&2
        exit 1
    fi
    cmp "$src" "$impl"
    sha256sum "$src" "$impl" | tee "$run/gdr-source.sha256"
    python "$tools/target_layer_probe.py" compile --batch "$run/air" --output "$run/omc" \
      --ddk-root "$ddk" --ddk-id "${CPRO_DDK_ID:-100.520.020.010}" --compiler-profile isolated-kirinx90 \
      --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}" --use-origin-format
    sha256sum -c "$run/gdr-source.sha256"
    sha256sum "$run/omc/compile.json" "$run/gdr-source.sha256" > "$run/compiled.sha256"
    ;;
  run)
    sha256sum -c "$run/compiled.sha256"
    if [[ ! -f "$run/execution/request.json" ]]; then
        python "$tools/validate_draft_vector.py" --config "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}" \
          --compiled "$run/omc" --ddk-root "${CPRO_DDK_ROOT:?}" --output "$run/execution" --apply
    fi
    if ! retrieved; then
        rc=0
        python "$tools/validate_draft_vector.py" --recover "$run/execution" --wait --output "$run/recovery" || rc=$?
        if ! retrieved; then
            echo "Incomplete device outputs (recover rc=$rc). See $run/recovery/result.json" >&2
            exit 1
        fi
    fi
    if [[ ! -f "$run/dump/manifest.json" ]]; then
        python "$tools/gdr_prefill_stages.py" compare --reference "$run/reference" --trace "$run" --output "$run/dump"
    fi
    ;;
  *) echo 'usage: bash check_l0_gdr_io.sh capture|air|compile|run [run_directory]' >&2; exit 2 ;;
esac
