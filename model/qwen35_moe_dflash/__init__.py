"""Portable DFlash reference; model dependencies load only when requested."""

from importlib import import_module as _import_module

_EXPORT_MODULES = {
    "DFlashDraftModel": "model",
    "extract_context_feature": "model",
    "DFlashOps": "ops",
    "ModuleDFlashOps": "ops",
    "TorchDFlashOps": "ops",
    "DFlashBaseModelOutputWithPast": "target_features",
    "DFlashCausalLMOutputWithPast": "target_features",
    "DFlashFeatureCollector": "target_features",
    "DFlashTargetFeatureSpec": "target_features",
    "QWEN35_35B_DFLASH_TARGET_FEATURES": "target_features",
    "DFLASH_CONFIG": "config",
    "DFLASH_TARGET_FEATURE_SIZE": "config",
    "DFLASH_TARGET_HIDDEN_SIZE": "config",
    "DFLASH_TARGET_LAYER_IDS": "config",
    "DFLASH_TARGET_NUM_HIDDEN_LAYERS": "config",
    "Qwen35DFlashConfig": "config",
    "audit_official_35b_dflash_config": "config",
    "audit_dflash_checkpoint": "weights",
}

__all__ = [
    "DFlashDraftModel",
    "DFlashOps",
    "DFlashBaseModelOutputWithPast",
    "DFlashCausalLMOutputWithPast",
    "DFlashFeatureCollector",
    "DFlashTargetFeatureSpec",
    "DFLASH_CONFIG",
    "DFLASH_TARGET_FEATURE_SIZE",
    "DFLASH_TARGET_HIDDEN_SIZE",
    "DFLASH_TARGET_LAYER_IDS",
    "DFLASH_TARGET_NUM_HIDDEN_LAYERS",
    "ModuleDFlashOps",
    "QWEN35_35B_DFLASH_TARGET_FEATURES",
    "Qwen35DFlashConfig",
    "TorchDFlashOps",
    "audit_dflash_checkpoint",
    "audit_official_35b_dflash_config",
    "extract_context_feature",
]


def __getattr__(name):
    module = _EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
