#!/usr/bin/env python3
"""Review existing packed observations, then run bounded native-dtype outputs."""
import sys

from run_draft_trace import main


if __name__ == "__main__":
    raise SystemExit(main([*sys.argv[1:], "--direct"]))
