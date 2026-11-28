#!/usr/bin/env bash
set -euo pipefail
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == --help || ${1:-} == -h ]]; then exec bash "$scripts/decoder_compare.sh" --help; fi
# A scale A/B can reuse the exact inputs and intermediate references from the first run.
source_case=
args=()
while (($#)); do
    case "$1" in
        --reuse-golden-from)
            if [[ -z ${2:-} || $2 == --* || -n $source_case ]]; then
                echo 'Expected one --reuse-golden-from CASE_DIR' >&2
                exit 2
            fi
            source_case=$2
            shift 2 ;;
        *) args+=("$1"); shift ;;
    esac
done
if [[ -n $source_case ]]; then
    bash "$scripts/decoder_transfer.sh" golden --local-source "$source_case" "${args[@]}"
else
    bash "$scripts/decoder_golden.sh" "${args[@]}"
fi
exec bash "$scripts/decoder_air.sh" "${args[@]}"
