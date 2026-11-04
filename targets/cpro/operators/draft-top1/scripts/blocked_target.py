#!/usr/bin/env python3
"""Fail closed until selected X90 build and runtime adapters exist."""
import json
import sys
print(json.dumps({"artifact_kind": "qwen35-custom-top1-target", "status": "PENDING",
    "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN",
    "reason": "Selected X90 registration/build adapter and device kernel are not implemented",
    "next": "Run scripts/inspect_ddk.py once and return ddk-context.tar.gz"}))
sys.exit(2)
