#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
# Prepare only. The user runs the selected SDK's generated build.sh to install.
exec python3 "$SCRIPT_DIR/prepare_project.py" "$@"
