#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'HELP'
decoder_compare.sh {source|prepare|board|plan} [--mode {linear|full|4|40|layerwise} --phase {prefill|decode}]
                   [--run-root DIR] [--remote-run DIR] [--patched-source DIR] [--recovery]
source      Generate or select the editable CPro source directory.
prepare     On the 310P/export host: capture reference, compute CPU golden, export static AIR.
board       On the DDK host: fetch, compile, run on device and compare CPU / 310P / CPro.
plan        Print the layers, sequence length and number of OMCs without running the model.
--recovery  Reuse verified completed OMC outputs in layerwise mode when resuming on the same board.
DECODER_ENV may point to a host-local shell configuration file.
HELP
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    source|prepare|board|plan) action=$1; shift ;;
    *) usage >&2; exit 2 ;;
esac

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo"
env_file=${DECODER_ENV:-${XDG_CONFIG_HOME:-$HOME/.config}/qwen35-cpro/decoder.env}
if [[ -f $env_file || -n ${DECODER_ENV:-} ]]; then
    set -a
    source "$env_file"
    set +a
fi
export REPO_DIR="$repo"
export PYTHONDONTWRITEBYTECODE=1
if [[ $action == prepare || $action == board ]]; then
    export QWEN35_TRACE_COMMANDS=${QWEN35_TRACE_COMMANDS:-1}
fi
tool="$repo/targets/cpro/tools/decoder_cases.py"

exec python3 "$tool" "$action" "$@"
