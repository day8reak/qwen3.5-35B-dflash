#!/usr/bin/env python3
"""Fail closed until selected X90 build and runtime adapters exist."""
import json
import sys
print(json.dumps({"artifact_kind": "qwen35-custom-top1-target", "status": "PENDING",
    "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN",
    "reason": "Source is supplied; a compiled OMC and bound device runner are still required",
    "next": "Prepare and compile StableArgMax with the selected SDK, then validate the standalone OMC"}))
sys.exit(2)
