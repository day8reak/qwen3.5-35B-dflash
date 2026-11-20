#!/usr/bin/env bash
# L0, physical64/valid19. Run on the DDK host with the retained 310P capture.
set -euo pipefail
tools_dir="${REPO_DIR:?}/targets/cpro/tools"
group="${AI_RUN_DIR:?}/target/tests/gdr-stages-t64-valid19-fresh01"
pair="$group/l0-silu-pair"
stages="$tools_dir/gdr_prefill_stages.py"
validate="$tools_dir/validate_draft_vector.py"

for manifest in "$group/reference/manifest.json" "$group/air/manifest.json" "$group/gamma-readout/recovery/result.json"; do
    if [[ ! -f "$manifest" ]]; then
        echo "Missing input for the L0 SiLU comparison: $manifest" >&2
        exit 1
    fi
done
if [[ ! -f "$pair/air/manifest.json" ]]; then
    python "$stages" prepare --baseline "$group" --reference "$group/reference" \
        --indexed-silu-layer 0 --paired-silu --output "$pair/air"
fi
if [[ ! -f "$pair/omc/compile.json" ]]; then
    python "$tools_dir/target_layer_probe.py" compile \
        --batch "$pair/air" --output "$pair/omc" \
        --ddk-root "${CPRO_DDK_ROOT:?}" --ddk-id "${CPRO_DDK_ID:?}" \
        --compiler-profile isolated-kirinx90 \
        --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}" --use-origin-format
fi
python "$stages" inspect-abi --compiled "$pair/omc" --ddk-root "$CPRO_DDK_ROOT" \
    --case l00_c00_norm_silu_pair_prefill_stages_o0
if [[ ! -e "$pair/execution/request.json" ]]; then
    python "$validate" --config "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}" \
        --compiled "$pair/omc" --ddk-root "$CPRO_DDK_ROOT" --output "$pair/execution" --apply
fi
retrieved() {
    python - "$pair/recovery/result.json" <<'PY'
import json
from pathlib import Path
import sys
p = Path(sys.argv[1])
sys.exit(0 if p.is_file() and json.loads(p.read_text()).get('device_execution') == 'EXECUTED_OUTPUTS_RETRIEVED' else 1)
PY
}
if ! retrieved; then
    # The original module hidden can fail against 310P even when readback succeeds.
    python "$validate" --recover "$pair/execution" --wait --output "$pair/recovery" || retrieved
fi
if [[ -f "$pair/dump/silu-result.json" ]]; then
    python - "$pair/dump/silu-result.json" <<'PY'
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
print(json.dumps(report, ensure_ascii=False, indent=2))
sys.exit(0 if report['status'] == 'PASSED' else 1)
PY
else
    python "$stages" compare --baseline "$group/gamma-readout" --reference "$group/reference" \
        --trace "$pair" --output "$pair/dump"
fi
