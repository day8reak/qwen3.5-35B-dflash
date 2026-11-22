#!/usr/bin/env bash
# Four independent prefill modules. Run capture/air on the export host, board on the OMC host.
set -euo pipefail
tools="${REPO_DIR:?}/targets/cpro/tools"
tests="${AI_RUN_DIR:?}/target/tests"
run="${2:-$tests/attention4-t64-valid19-tmatrix-01}"
previous="${ATTENTION4_BASELINE:-$tests/attention4-t64-valid19-v1024-qlate-01}"
probe="$tools/attention4_qscale.py"

stage_available_air() {
    if [[ -e "$run/air" ]]; then
        python "$probe" stage-air --run "$run" || exit 1
        return 0
    fi
    for src in "$previous/air" "$run/export/air"; do
        if [[ -f "$src/manifest.json" ]] && python "$probe" stage-air --source-air "$src" --run "$run"; then
            return 0
        fi
    done
    return 1
}

case "${1:-}" in
    capture)
        if stage_available_air; then exit 0; fi
        if [[ ! -f "$run/reference/manifest.json" ]]; then
            python "$tools/check_segment_prefill.py" --source-root "${TARGET_SOURCE:?}" \
              --config "${NZ_CONFIG:-config/qwen3_5_moe.yaml}" --attention-layers --output "$run/reference"
        elif [[ ! -f "$run/reference/module-cpu/report.json" ]]; then
            python "$tools/check_segment_prefill.py" --attention-layers --compare-cpu "$run/reference" \
              --output "$run/reference/module-cpu"
        fi
        ;;
    air)
        if stage_available_air; then exit 0; fi
        PYTHONPATH="$REPO_DIR/model${PYTHONPATH:+:$PYTHONPATH}" python - "$run/reference" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from qwen35_moe_dflash.target.layer_probe.common import read_array
root = Path(sys.argv[1])
with (root / 'manifest.json').open() as f:
    capture = json.load(f)
ids = read_array(root, capture['inputs']['input_ids']).astype('<i4', copy=False)
if hashlib.sha256(ids.tobytes()).hexdigest() != 'ec5119e4b49de3c9e0cd48b7cff7faf13b18a0a10dda89c2a5180b194f05eaf7':
    sys.exit('Token IDs differ from the original T64/valid19 prefill.')
PY
        python "$tools/prepare_cpro_prefill_t64.py" --source-root "${CPRO_SOURCE:?}" \
          --config "${TARGET_CONFIG:-config/qwen3_5_moe.yaml}" --fixture "$run/reference" \
          --attention-layers --export --output "$run/export"
        python "$probe" stage-air --source-air "$run/export/air" --run "$run"
        ;;
    board)
        if ! stage_available_air; then
            python "$REPO_DIR/targets/cpro/scripts/pull_artifacts.py" --from air "$run/air"
        fi
        python "$probe" board --run "$run" --baseline-run "$previous" --label 'T-matrix' \
          --gdr-source "${GDR_SOURCE:?set GDR_SOURCE to the modified allops kernel source}" --csv "$run/tmatrix-accuracy.csv"
        ;;
    *) echo 'usage: bash check_attention4_tmatrix.sh capture|air|board [run_directory]' >&2; exit 2 ;;
esac
