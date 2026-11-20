#!/usr/bin/env bash
# Run on the DDK host in custom_pypto. An optional argument selects a fresh run directory.
set -euo pipefail
group="${AI_RUN_DIR:?}/target/tests/gdr-stages-t64-valid19-fresh01"
run="${1:-$group/l0-s1-v1}"
tools="${REPO_DIR:?}/targets/cpro/tools"
stages="$tools/gdr_prefill_stages.py"
ddk="${CPRO_DDK_ROOT:?}"
: "${CPRO_DDK_ID:?}"

if [[ ! -f "$group/reference/manifest.json" ]]; then
    python "$REPO_DIR/targets/cpro/scripts/pull_artifacts.py" "$group/reference"
fi
# The old norm and SiLU probes were created on this host. They can also be
# regenerated here from the original whole-module AIR transferred from the AIR host.
if [[ ! -f "$run/air/manifest.json" && ! -f "$group/air/manifest.json" ]]; then
    for batch in gamma-readout norm-single-l0 l0-silu-pair; do
        if [[ ! -f "$group/$batch/air/manifest.json" ]]; then
            python "$REPO_DIR/targets/cpro/scripts/pull_artifacts.py" "$group/air"
            break
        fi
    done
fi
if [[ ! -f "$run/air/manifest.json" ]]; then
    python "$stages" prepare-s1 --group "$group" --run "$run"
fi
# Keep the installed GDR files identifiable even when the DDK version string stays the same.
python - "$ddk" "$CPRO_DDK_ID" "$run" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path
ddk, run = Path(sys.argv[1]).resolve(strict=True), Path(sys.argv[3])
files = {}
for directory, _, names in os.walk(ddk):
    for name in names:
        path = Path(directory, name)
        relative = str(path.relative_to(ddk))
        if path.suffix in ('.pyc', '.log', '.tmp') or '__pycache__' in path.parts:
            continue
        if 'chunkgateddeltarule' in relative.lower().replace('_', '').replace('-', '') and path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1 << 20), b''):
                    digest.update(block)
            files[relative] = digest.hexdigest()
record = dict(ddk_root=str(ddk), ddk_id=sys.argv[2], installed_gdr_files=files)
receipt = run / 'ddk-gdr.json'
if receipt.exists():
    if json.loads(receipt.read_text()) != record:
        sys.exit('The DDK or installed GDR files changed. Pass a new run directory to this script.')
else:
    receipt.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
print(f'DDK {ddk}, {len(files)} named GDR files recorded in {receipt}', flush=True)
PY
if [[ ! -f "$run/omc/compile.json" ]]; then
    python "$tools/target_layer_probe.py" compile \
        --batch "$run/air" --output "$run/omc" \
        --ddk-root "$ddk" --ddk-id "$CPRO_DDK_ID" \
        --compiler-profile isolated-kirinx90 \
        --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}" --use-origin-format
fi
python - "$run/omc/compile.json" "$CPRO_DDK_ID" <<'PY'
import json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text())
if r['ddk_id'] != sys.argv[2] or not r['cases'] or any(c['status'] != 'COMPILED' for c in r['cases'].values()):
    sys.exit(f"OMC batch is incomplete or DDK ID changed. Inspect {sys.argv[1]}.")
PY
if [[ ! -f "$run/execution/request.json" ]]; then
    python "$tools/validate_draft_vector.py" \
        --config "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}" \
        --compiled "$run/omc" --ddk-root "$ddk" --output "$run/execution" --apply
fi
retrieved() {
    python - "$run/recovery/result.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
r = json.loads(p.read_text()) if p.is_file() else {}
sys.exit(0 if r.get('device_execution') == 'EXECUTED_OUTPUTS_RETRIEVED' and r.get('coverage_complete') else 1)
PY
}
if ! retrieved; then
    python "$tools/validate_draft_vector.py" --recover "$run/execution" --wait --output "$run/recovery" || retrieved
fi
python "$stages" summarize-s1 --group "$group" --run "$run"
