"""Explicit operator boundary for the independent stateful target model.

This is the model's logical adapter ABI, not a claim about yellow-zone GE port
names. A vendor adapter must translate these calls using its actual registered
schemas, including physical cache layout and convolution-state conventions.
No operator implementation is loaded implicitly and no arithmetic fallback is
provided here. The original model and its operator path remain independent.
"""

from __future__ import annotations

import importlib
import math
from types import MappingProxyType, ModuleType
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


FUSED_OPS_ABI = 1


@runtime_checkable
class FusedOps(Protocol):
    """Logical ND operations consumed by the stateful target fork.

    The model owns position, state lifetime, feature taps and residual order.
    Implementations must preserve the declared tensor dtypes. In particular,
    recurrent-state precision is explicit; this interface never silently casts
    a vendor's FP32 state into an FP16 model policy.
    """

    abi_version: int
    backend_id: str
    is_reference_backend: bool

    def rms_norm(self, x: Tensor, gamma: Tensor, eps: float) -> Tensor:
        """Normalize using effective gamma; do not add another weight offset."""
        ...

    def conv_state_update(
        self, projected: Tensor, history: Tensor
    ) -> tuple[Tensor, Tensor]:
        """[B,S,C], [B,C,K-1] -> (new history [B,C,K-1], [B,C,S+K-1]).

        A vendor operation that returns K history elements must be translated
        explicitly by its adapter, after checking its actual slicing semantics.
        """
        ...

    def causal_conv1d(self, padded: Tensor, weight: Tensor) -> Tensor:
        """[B,C,S+K-1], [C,1,K] -> [B,S,C], without SiLU activation."""
        ...

    def chunk_gated_delta_rule(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        g: Tensor,
        beta: Tensor,
        initial_state: Tensor,
        effective_length: Tensor,
        *,
        chunk_size: int,
        use_qk_l2norm_in_kernel: bool,
    ) -> tuple[Tensor, Tensor]:
        """Return core [B,S,H,Dv] and final state [B,H,Dk,Dv].

        Q/K/V use sequence-major ND. Q/K have already been repeated to the
        value-head count. g/beta are [B,S,H]; effective_length is INT16 [B].
        State must always be returned to the caller. A true normalization flag
        assigns Q/K L2 to this operation, so the model must not apply it twice.
        """
        ...

    def cache_update(
        self,
        key: Tensor | None,
        value: Tensor | None,
        new_key: Tensor,
        new_value: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Append logical ND [B,Hkv,S,D] K/V, returning the complete cache.

        Both existing tensors may be None on an empty cache. Mapping this
        boundary to paged/NZ CacheUpdate belongs to the explicit vendor adapter.
        """
        ...

    def attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        query_start: int,
        scale: float,
    ) -> Tensor:
        """Causal GQA: [B,H,S,D], cached [B,Hkv,T,D] -> [B,H,S,D].

        Query row i may see key positions <= query_start+i. The adapter owns
        causal mask construction and any vendor physical-layout conversion.
        """
        ...

    def moe(
        self,
        hidden: Tensor,
        router_weight: Tensor,
        gate_up: Tensor,
        down: Tensor,
        *,
        top_k: int,
    ) -> Tensor:
        """Routed experts only; the model separately adds the shared expert.

        Weights use logical [E,H], [E,2I,H], [E,H,I]. Selection follows router
        softmax/top-k with selected weights renormalized. Physical quantization
        and packed weight formats require their own explicit adapter contract.
        """
        ...


class ModuleFusedOps:
    """Bind all operations to one explicitly selected Python adapter module.

    Modules must declare FUSED_OPS_ABI=1 and a nonempty BACKEND_ID. A host-only
    module marked IS_REFERENCE_BACKEND=True also requires allow_reference=True;
    it cannot be selected accidentally as a production custom-op backend.

    Functions are bound at construction. Exceptions and return values propagate
    unchanged: a failed or unavailable custom operation is never retried using
    the old unfolded graph, a CPU implementation, or a different dtype.
    """

    required_operations = (
        "rms_norm",
        "conv_state_update",
        "causal_conv1d",
        "chunk_gated_delta_rule",
        "cache_update",
        "attention",
        "moe",
    )

    def __init__(self, module: ModuleType, *, allow_reference: bool = False) -> None:
        if not isinstance(module, ModuleType):
            raise TypeError("fused operators require an explicit Python adapter module")
        if type(allow_reference) is not bool:
            raise TypeError("allow_reference must be a bool")
        version = getattr(module, "FUSED_OPS_ABI", None)
        if type(version) is not int or version != FUSED_OPS_ABI:
            raise ValueError(f"fused adapter must declare FUSED_OPS_ABI={FUSED_OPS_ABI}")
        backend_id = getattr(module, "BACKEND_ID", None)
        if not isinstance(backend_id, str) or not backend_id.strip():
            raise ValueError("fused adapter must declare a nonempty BACKEND_ID")
        reference = getattr(module, "IS_REFERENCE_BACKEND", False)
        if type(reference) is not bool:
            raise ValueError("IS_REFERENCE_BACKEND must be a bool when declared")
        if reference and not allow_reference:
            raise ValueError("host reference backend requires explicit allow_reference=True")
        missing = [
            name for name in self.required_operations
            if not callable(getattr(module, name, None))
        ]
        if missing:
            raise ValueError(f"fused adapter is missing callable operations: {', '.join(missing)}")
        self.abi_version = version
        self.backend_id = backend_id
        self.is_reference_backend = reference
        self.module = module
        self._operations = MappingProxyType({
            name: getattr(module, name) for name in self.required_operations
        })

    @classmethod
    def from_name(cls, module_name: str, *, allow_reference: bool = False) -> "ModuleFusedOps":
        if not isinstance(module_name, str) or not module_name.strip():
            raise ValueError("an explicit fused adapter module name is required")
        return cls(importlib.import_module(module_name), allow_reference=allow_reference)

    def rms_norm(self, x: Tensor, gamma: Tensor, eps: float) -> Tensor:
        return self._operations["rms_norm"](x, gamma, eps)

    def conv_state_update(
        self, projected: Tensor, history: Tensor
    ) -> tuple[Tensor, Tensor]:
        return self._operations["conv_state_update"](projected, history)

    def causal_conv1d(self, padded: Tensor, weight: Tensor) -> Tensor:
        return self._operations["causal_conv1d"](padded, weight)

    def chunk_gated_delta_rule(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        g: Tensor,
        beta: Tensor,
        initial_state: Tensor,
        effective_length: Tensor,
        *,
        chunk_size: int,
        use_qk_l2norm_in_kernel: bool,
    ) -> tuple[Tensor, Tensor]:
        return self._operations["chunk_gated_delta_rule"](
            query, key, value, g, beta, initial_state, effective_length,
            chunk_size=chunk_size, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    def cache_update(
        self,
        key: Tensor | None,
        value: Tensor | None,
        new_key: Tensor,
        new_value: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self._operations["cache_update"](key, value, new_key, new_value)

    def attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        query_start: int,
        scale: float,
    ) -> Tensor:
        return self._operations["attention"](
            query, key, value, query_start=query_start, scale=scale,
        )

    def moe(
        self,
        hidden: Tensor,
        router_weight: Tensor,
        gate_up: Tensor,
        down: Tensor,
        *,
        top_k: int,
    ) -> Tensor:
        return self._operations["moe"](
            hidden, router_weight, gate_up, down, top_k=top_k,
        )


class SelectiveFusedOps(ModuleFusedOps):
    """Override an explicit complete base backend with documented vendor calls.

    ``fused_ops`` selects ``none`` or a subset of ``gdn,conv_state,rms``. Other
    operations always use ``base_ops``; failure never changes that selection.
    There is no implicit arithmetic implementation or FP32 reference fallback.
    The caller is responsible for selecting an appropriate FP16 base backend.

    Vendor mode explicitly imports torch_npu, then requires registered torch.ops
    calls. The document describes Ascend310P3 interfaces, not proven support on
    another target. GDN is restricted to 32 heads of width 128; conv to 8192
    channels and kernel 4. Selecting none loads no vendor module.

    GDN native input state dtype must be explicitly chosen; its documented
    output is FP32. Casts between these ports and the model cache require
    ``gdn_state_casts="allow"``. They never change Q/K/V computation dtype.
    """

    _vendor_names = {
        "gdn": ("npu", "npu_chunk_gated_delta_rule"),
        "rms": ("npu", "adn_rms_norm"),
        "conv_state": ("my_ops", "linear_attention_convstate_update"),
    }

    def __init__(self, base_ops: ModuleFusedOps, *, fused_ops="none",
                 vendor_backend_id=None, gdn_initial_state_dtype=None,
                 gdn_state_casts="reject", allow_reference=False):
        if not isinstance(base_ops, ModuleFusedOps) or isinstance(base_ops, SelectiveFusedOps):
            raise TypeError("base_ops must be a complete ModuleFusedOps backend, not a selective wrapper")
        if type(allow_reference) is not bool:
            raise TypeError("allow_reference must be a bool")
        if base_ops.is_reference_backend and not allow_reference:
            raise ValueError("a reference base requires explicit allow_reference=True")
        if isinstance(fused_ops, str):
            roles = [] if fused_ops == "none" else fused_ops.split(",")
        else:
            try:
                roles = list(fused_ops)
            except TypeError:
                raise ValueError("fused_ops must select none or gdn,conv_state,rms") from None
        if (any(not isinstance(role, str) or role not in self._vendor_names for role in roles)
                or len(roles) != len(set(roles))):
            raise ValueError("fused_ops must contain unique gdn,conv_state,rms roles")
        if gdn_state_casts not in {"reject", "allow"}:
            raise ValueError("gdn_state_casts must be reject or allow")
        if "gdn" in roles and gdn_initial_state_dtype not in (torch.float16, torch.float32):
            raise ValueError("gdn requires an explicit float16 or float32 initial-state dtype")
        if roles and (not isinstance(vendor_backend_id, str) or not vendor_backend_id.strip()):
            raise ValueError("selected vendor roles require an explicit vendor_backend_id")
        self.base_ops = base_ops
        self.fused_ops = frozenset(roles)
        self.vendor_backend_id = vendor_backend_id if roles else None
        self.gdn_initial_state_dtype = gdn_initial_state_dtype
        self.gdn_state_casts = gdn_state_casts
        calls = {}
        if roles:
            # Explicit selection only. This import can register npu operations;
            # my_ops registration must also have been supplied by the caller.
            importlib.import_module("torch_npu")
            for role in sorted(roles):
                namespace, name = self._vendor_names[role]
                call = getattr(getattr(torch.ops, namespace), name, None)
                if not callable(call):
                    raise ValueError(f"selected vendor op is not registered: {namespace}::{name}")
                calls[role] = call
        self._vendor_calls = MappingProxyType(calls)
        module = ModuleType("qwen35_documented_selective_ops")
        module.FUSED_OPS_ABI = FUSED_OPS_ABI
        module.BACKEND_ID = f"selective[{base_ops.backend_id};roles={','.join(sorted(roles)) or 'none'};vendor={self.vendor_backend_id}]"
        module.IS_REFERENCE_BACKEND = base_ops.is_reference_backend
        for name in self.required_operations:
            setattr(module, name, getattr(base_ops, name))
        overrides = {"rms": ("rms_norm", self._documented_rms),
                     "conv_state": ("conv_state_update", self._documented_conv),
                     "gdn": ("chunk_gated_delta_rule", self._documented_gdn)}
        for role in roles:
            name, implementation = overrides[role]
            setattr(module, name, implementation)
        super().__init__(module, allow_reference=allow_reference)

    @staticmethod
    def _tensor(value, shape, dtype, device, label):
        if (not isinstance(value, Tensor) or tuple(value.shape) != tuple(shape)
                or value.dtype != dtype or value.device != device):
            raise ValueError(f"{label}: expected {dtype} {tuple(shape)} on {device}")
        return value

    @staticmethod
    def _pair(result, label):
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise ValueError(f"{label} must return exactly two tensors")
        return result

    def _state_policy(self, cache_dtype):
        if cache_dtype not in (torch.float16, torch.float32):
            raise ValueError("GDN cache must be float16 or float32")
        if (self.gdn_state_casts == "reject"
                and (cache_dtype != self.gdn_initial_state_dtype or cache_dtype != torch.float32)):
            raise ValueError("GDN state ports differ from cache dtype; explicitly select gdn_state_casts='allow'")

    def validate_model_config(self, config, recurrent_dtype):
        """Reject unsupported documented geometry before loading model tensors."""
        if not any(kind == "linear_attention" for kind in config["layer_types"]):
            return
        if "gdn" in self.fused_ops:
            if (config["linear_num_value_heads"], config["linear_key_head_dim"],
                    config["linear_value_head_dim"]) != (32, 128, 128):
                raise ValueError("documented GDN requires 32 value heads and 128-dimensional Q/K/V")
            self._state_policy(recurrent_dtype)
        if "conv_state" in self.fused_ops:
            channels = (2 * config["linear_num_key_heads"] * config["linear_key_head_dim"]
                        + config["linear_num_value_heads"] * config["linear_value_head_dim"])
            if channels != 8192 or config["linear_conv_kernel_dim"] != 4:
                raise ValueError("documented conv_state requires 8192 channels and kernel size 4")

    def _documented_rms(self, x, gamma, eps):
        if not isinstance(x, Tensor) or x.ndim < 1:
            raise ValueError("AdnRmsNorm input must be a tensor with a final hidden dimension")
        self._tensor(x, x.shape, torch.float16, x.device, "AdnRmsNorm input")
        self._tensor(gamma, (x.shape[-1],), torch.float16, x.device, "AdnRmsNorm gamma")
        if not isinstance(eps, (int, float)) or isinstance(eps, bool) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("AdnRmsNorm epsilon must be finite and positive")
        y, rstd = self._pair(self._vendor_calls["rms"](x, gamma, float(eps)), "AdnRmsNorm")
        self._tensor(y, x.shape, torch.float16, x.device, "AdnRmsNorm y")
        self._tensor(rstd, (*x.shape[:-1], 1), torch.float32, x.device, "AdnRmsNorm rstd")
        return y

    def _documented_conv(self, projected, history):
        if not isinstance(projected, Tensor) or projected.ndim != 3 or projected.shape[1] < 1:
            raise ValueError("conv_state projected input must be [1,S,8192] with S > 0")
        sequence = projected.shape[1]
        self._tensor(projected, (1, sequence, 8192), torch.float16, projected.device, "conv_state projected")
        self._tensor(history, (1, 8192, 3), torch.float16, projected.device, "conv_state history")
        control = torch.zeros(128, dtype=torch.int32, device=projected.device)
        raw_history, padded = self._pair(
            self._vendor_calls["conv_state"](projected, history, control), "LinearAttentionConvstateUpdate")
        self._tensor(raw_history, (1, 8192, 4), torch.float16, projected.device, "conv_state y1")
        self._tensor(padded, (1, 8192, sequence + 3), torch.float16, projected.device, "conv_state y2")
        # y2 is documented as history || current input. Its tail determines the
        # canonical next history without assuming the ordering inside raw y1.
        return padded[..., -3:].detach().clone().contiguous(), padded

    def _documented_gdn(self, query, key, value, g, beta, initial_state, effective_length,
                        *, chunk_size, use_qk_l2norm_in_kernel):
        if not isinstance(query, Tensor) or query.ndim != 4 or not 1 <= query.shape[1] <= 32767:
            raise ValueError("GDN query must be [1,S,32,128] with positive INT16 length")
        sequence, device = query.shape[1], query.device
        for label, tensor in (("query", query), ("key", key), ("value", value)):
            self._tensor(tensor, (1, sequence, 32, 128), torch.float16, device, "GDN " + label)
        for label, tensor in (("g", g), ("beta", beta)):
            self._tensor(tensor, (1, sequence, 32), torch.float16, device, "GDN " + label)
        self._tensor(effective_length, (1,), torch.int16, device, "GDN effective_length")
        if not isinstance(initial_state, Tensor):
            raise ValueError("GDN initial_state must be a tensor")
        cache_dtype = initial_state.dtype
        self._state_policy(cache_dtype)
        state_shape = (1, 32, 128, 128)
        self._tensor(initial_state, state_shape, cache_dtype, device, "GDN initial_state")
        if type(chunk_size) is not int or chunk_size != (1 if sequence == 1 else 64):
            raise ValueError("documented GDN uses chunk_size=1 for decode and 64 for prefill")
        if type(use_qk_l2norm_in_kernel) is not bool:
            raise ValueError("use_qk_l2norm_in_kernel must be a bool")
        native_state = initial_state.to(dtype=self.gdn_initial_state_dtype)
        # Signature order is from user document section 4.4. State must always
        # be returned; Q/K reach the kernel unchanged, with L2 controlled there.
        core, final_state = self._pair(self._vendor_calls["gdn"](
            query, key, value, g, beta, chunk_size, native_state, effective_length,
            1, int(use_qk_l2norm_in_kernel)), "ChunkGatedDeltaRule")
        self._tensor(core, value.shape, torch.float16, device, "GDN core_attn")
        self._tensor(final_state, state_shape, torch.float32, device, "GDN last_recurrent_state")
        return core, final_state.to(dtype=cache_dtype)
