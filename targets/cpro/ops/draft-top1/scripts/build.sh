#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
# Stage only. The user runs their existing allops build command to install.
exec python3 "$SCRIPT_DIR/prepare_allops.py" "$@"
