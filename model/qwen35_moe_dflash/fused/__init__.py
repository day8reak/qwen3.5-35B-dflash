"""Opt-in fused/stateful target lane; never selected by the legacy entry points."""
from qwen35_moe_dflash.fused.model import FusedTargetModel, FusedTargetOutput, load_prepared_target
from qwen35_moe_dflash.fused.ops import FusedOps, ModuleFusedOps, SelectiveFusedOps
from qwen35_moe_dflash.fused.session import LayerState, TargetState, StateStore, FusedTargetSession, TargetVerification
from qwen35_moe_dflash.fused.quantization import OriginalQuantOps, OriginalQuantizedWeights, load_original_quantized_target
from qwen35_moe_dflash.fused.original_wrapper import OriginalCproTargetWrapper

__all__ = ["FusedTargetModel", "FusedTargetOutput", "FusedOps", "ModuleFusedOps", "SelectiveFusedOps",
           "LayerState", "TargetState", "StateStore", "load_prepared_target",
           "FusedTargetSession", "TargetVerification", "OriginalQuantOps",
           "OriginalQuantizedWeights", "load_original_quantized_target", "OriginalCproTargetWrapper"]
