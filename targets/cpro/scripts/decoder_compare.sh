#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'HELP'
decoder_compare.sh {prepare|board|plan} --mode {linear|full|4|40|layerwise} --phase {prefill|decode}
                   [--run DIR] [--remote-run DIR] [--recovery]
prepare     On the 310P/export host: capture reference, compute CPU golden, export static AIR.
board       On the DDK host: fetch, compile, run on device and compare CPU / 310P / CPro.
plan        Print the layers, sequence length and number of OMCs without running the model.
--recovery  Reuse verified completed OMC outputs in layerwise mode when resuming on the same board.
DECODER_ENV may point to a host-local shell configuration file.
HELP
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    prepare|board|plan) action=$1; shift ;;
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
tool="$repo/targets/cpro/tools/decoder_cases.py"

in_env() {
    local name=$1
    shift
    if [[ $name == current || ${CONDA_DEFAULT_ENV:-} == "$name" ]]; then
        python -u "$tool" "$@"
    else
        conda run --no-capture-output -n "$name" python -u "$tool" "$@"
    fi
}
case "$action" in
    plan) python3 "$tool" plan "$@" ;;
    prepare)
        in_env "${DECODER_NZ_ENV:-qwen3_5_w4}" reference "$@"
        in_env "${DECODER_EXPORT_ENV:-qwen3_5_cpro}" air "$@"
        ;;
    board) in_env "${DECODER_BOARD_ENV:-custom_pypto}" board "$@" ;;
esac
