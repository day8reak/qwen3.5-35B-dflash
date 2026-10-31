"""Process-local backend selection for explicit CPU/CUDA reference commands.

Transformers 5.13 probes an installed torch_npu even for eager CPU attention.
Keep an incomplete AIR installation from breaking unrelated host references.
This changes optional backend discovery only, never model math or vendor files.
AIR commands must run in a fresh process without this policy.
"""
import os
import sys

_REFERENCE_DEVICE = None


def configure_reference_backend(device):
    global _REFERENCE_DEVICE
    kind = str(device).split(":", 1)[0]
    if kind not in ("cpu", "cuda"):
        return False
    if _REFERENCE_DEVICE is not None:
        return True
    if "torch_npu" in sys.modules:
        raise RuntimeError("CPU/CUDA reference isolation requires a fresh CLI process; "
                           "torch_npu has already been imported")
    # Must precede the first torch import in the CLI process.
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import torch  # Load only the selected host backend before Transformers.
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    if not callable(getattr(imports, "is_torch_npu_available", None)):
        raise RuntimeError("this Transformers version lacks the expected NPU detector; "
                           "use the project-pinned Transformers version")
    def npu_unavailable(check_device=False):
        return False
    # Both aliases are used by Transformers integrations. Set them before
    # importing any model; this is not an official Transformers env switch.
    imports.is_torch_npu_available = npu_unavailable
    utils.is_torch_npu_available = npu_unavailable
    _REFERENCE_DEVICE = str(device)
    return True


def require_unmodified_air_backend():
    if _REFERENCE_DEVICE is not None:
        raise RuntimeError("AIR export cannot run after CPU/CUDA reference backend isolation "
                           "in the same process; start a fresh export-air/export-tiny-air CLI")
