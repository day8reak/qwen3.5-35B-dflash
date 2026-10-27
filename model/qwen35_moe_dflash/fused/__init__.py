"""Opt-in fused/stateful target lane; never selected by the legacy entry points."""
from .model import FusedTargetModel, FusedTargetOutput, load_prepared_target
from .ops import FusedOps, ModuleFusedOps, SelectiveFusedOps
from .session import LayerState, TargetState, StateStore, FusedTargetSession, TargetVerification

__all__ = ["FusedTargetModel", "FusedTargetOutput", "FusedOps", "ModuleFusedOps", "SelectiveFusedOps",
           "LayerState", "TargetState", "StateStore", "load_prepared_target",
           "FusedTargetSession", "TargetVerification"]
