#!/usr/bin/env bash
set -euo pipefail
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == --help || ${1:-} == -h ]]; then exec bash "$scripts/decoder_compare.sh" --help; fi
baseline=
args=()
while (($#)); do
    case "$1" in
        --check-padding-from)
            if [[ -z ${2:-} || $2 == --* || -n $baseline || ${2##*/} != 4-prefill-t64 ]]; then
                echo 'Expected one --check-padding-from /path/to/4-prefill-t64' >&2
                exit 2
            fi
            baseline=$2
            shift 2 ;;
        *) args+=("$1"); shift ;;
    esac
done
if [[ -n $baseline ]]; then
    bash "$scripts/decoder_compare.sh" l0-dump "${args[@]}" --mode 4 --phase prefill --run "$baseline" --remote-run "$baseline"
fi
bash "$scripts/decoder_transfer.sh" air "${args[@]}"
bash "$scripts/decoder_omc.sh" "${args[@]}"
exec bash "$scripts/decoder_board.sh" "${args[@]}"
