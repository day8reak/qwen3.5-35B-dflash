#!/usr/bin/env bash
set -euo pipefail
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == --help || ${1:-} == -h ]]; then exec bash "$scripts/decoder_compare.sh" --help; fi
bash "$scripts/decoder_transfer.sh" air "$@"
bash "$scripts/decoder_omc.sh" "$@"
bash "$scripts/decoder_board.sh" "$@"
