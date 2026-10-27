"""Backend selection and argument-preservation checks for the new fused fork."""

import types
import unittest
from unittest.mock import patch

import torch
import pytest

from qwen35_moe_dflash.fused.ops import FusedOps, ModuleFusedOps, SelectiveFusedOps


def adapter_module():
    module = types.ModuleType("test_fused_adapter")
    module.FUSED_OPS_ABI = 1
    module.BACKEND_ID = "test-adapter-v1"
    module.calls = []
    module.results = {}

    def operation(name):
        def call(*args, **kwargs):
            module.calls.append((name, args, kwargs))
            return module.results[name]
        return call

    for name in ModuleFusedOps.required_operations:
        setattr(module, name, operation(name))
        module.results[name] = object()
    return module


class FusedOpsContractTests(unittest.TestCase):
    def test_requires_explicit_module_and_exact_abi_version(self):
        with self.assertRaises(TypeError):
            ModuleFusedOps(None)
        for version in (None, True, 1.0, "1", 0, 2):
            with self.subTest(version=version):
                module = adapter_module()
                module.FUSED_OPS_ABI = version
                with self.assertRaisesRegex(ValueError, "FUSED_OPS_ABI=1"):
                    ModuleFusedOps(module)

    def test_requires_backend_identity(self):
        for identity in (None, "", "  ", 1):
            with self.subTest(identity=identity):
                module = adapter_module()
                module.BACKEND_ID = identity
                with self.assertRaisesRegex(ValueError, "BACKEND_ID"):
                    ModuleFusedOps(module)

    def test_every_operation_is_required_and_callable(self):
        for name in ModuleFusedOps.required_operations:
            with self.subTest(name=name):
                module = adapter_module()
                setattr(module, name, None)
                with self.assertRaisesRegex(ValueError, name):
                    ModuleFusedOps(module)
                self.assertEqual(module.calls, [])

    def test_reference_backend_requires_explicit_opt_in(self):
        module = adapter_module()
        module.IS_REFERENCE_BACKEND = True
        with self.assertRaisesRegex(ValueError, "allow_reference=True"):
            ModuleFusedOps(module)
        ops = ModuleFusedOps(module, allow_reference=True)
        self.assertTrue(ops.is_reference_backend)
        self.assertEqual(ops.abi_version, 1)
        self.assertEqual(ops.backend_id, module.BACKEND_ID)
        self.assertIsInstance(ops, FusedOps)
        with self.assertRaises(TypeError):
            ModuleFusedOps(module, allow_reference="True")
        module.IS_REFERENCE_BACKEND = "False"
        with self.assertRaisesRegex(ValueError, "IS_REFERENCE_BACKEND"):
            ModuleFusedOps(module, allow_reference=True)

    def test_all_callbacks_preserve_tensor_identity_and_keywords(self):
        module = adapter_module()
        ops = ModuleFusedOps(module)
        x = torch.ones(1, dtype=torch.float16)
        state = torch.ones(1, dtype=torch.float32)
        length = torch.tensor([1], dtype=torch.int16)
        invocations = [
            ("rms_norm", (x, x, 1e-6), {}),
            ("conv_state_update", (x, x), {}),
            ("causal_conv1d", (x, x), {}),
            ("chunk_gated_delta_rule", (x, x, x, x, x, state, length),
             {"chunk_size": 64, "use_qk_l2norm_in_kernel": True}),
            ("cache_update", (None, None, x, x), {}),
            ("attention", (x, x, x), {"query_start": 7, "scale": 0.125}),
            ("moe", (x, x, x, x), {"top_k": 8}),
        ]
        for name, args, kwargs in invocations:
            with self.subTest(name=name):
                result = getattr(ops, name)(*args, **kwargs)
                self.assertIs(result, module.results[name])
                actual_name, actual_args, actual_kwargs = module.calls[-1]
                self.assertEqual(actual_name, name)
                self.assertEqual(len(actual_args), len(args))
                for actual, original in zip(actual_args, args):
                    self.assertIs(actual, original)
                self.assertEqual(actual_kwargs, kwargs)
        self.assertEqual(len(module.calls), len(invocations))
        self.assertFalse(ops.is_reference_backend)

    def test_adapter_failure_propagates_without_fallback(self):
        module = adapter_module()
        failure = RuntimeError("registered custom kernel unavailable")

        def failing(*args):
            raise failure

        module.rms_norm = failing
        ops = ModuleFusedOps(module)
        with self.assertRaises(RuntimeError) as caught:
            ops.rms_norm(torch.ones(1), torch.ones(1), 1e-6)
        self.assertIs(caught.exception, failure)
        self.assertEqual(module.calls, [])

    def test_from_name_imports_only_explicit_adapter_and_preserves_errors(self):
        module = adapter_module()
        module.IS_REFERENCE_BACKEND = True
        with patch("qwen35_moe_dflash.fused.ops.importlib.import_module", return_value=module) as load:
            ops = ModuleFusedOps.from_name("test_fused_adapter", allow_reference=True)
        load.assert_called_once_with("test_fused_adapter")
        self.assertTrue(ops.is_reference_backend)
        for name in (None, "", "   "):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    ModuleFusedOps.from_name(name)
        failure = ModuleNotFoundError("adapter not installed")
        with patch("qwen35_moe_dflash.fused.ops.importlib.import_module", side_effect=failure):
            with self.assertRaises(ModuleNotFoundError) as caught:
                ModuleFusedOps.from_name("absent_fused_adapter")
        self.assertIs(caught.exception, failure)


if __name__ == "__main__":
    unittest.main()


@pytest.fixture
def documented_vendor(monkeypatch):
    calls = []

    def rms(x, gamma, epsilon):
        calls.append(("rms", (x, gamma, epsilon)))
        return x.clone(), torch.ones((*x.shape[:-1], 1), dtype=torch.float32, device=x.device)

    def conv(projected, history, control):
        calls.append(("conv_state", (projected, history, control)))
        padded = torch.cat((history, projected.transpose(1, 2)), dim=-1)
        # Deliberately different from y2's tail: no ordering inside y1 is assumed.
        return torch.full((1, 8192, 4), -7, dtype=projected.dtype), padded

    def gdn(*args):
        calls.append(("gdn", args))
        return args[2].clone(), args[6].float().clone()

    namespace = types.SimpleNamespace(
        npu=types.SimpleNamespace(adn_rms_norm=rms, npu_chunk_gated_delta_rule=gdn),
        my_ops=types.SimpleNamespace(linear_attention_convstate_update=conv))
    monkeypatch.setattr("qwen35_moe_dflash.fused.ops.torch.ops", namespace)
    with patch("qwen35_moe_dflash.fused.ops.importlib.import_module") as load:
        yield namespace, calls, load


def selective(base=None, **options):
    return SelectiveFusedOps(base or ModuleFusedOps(adapter_module()),
                             vendor_backend_id="test-only-vendor", **options)


def documented_gdn_inputs(dtype=torch.float16, sequence=2):
    query = torch.full((1, sequence, 32, 128), 0.125, dtype=torch.float16)
    return (query, query.clone(), query.clone(),
            torch.zeros((1, sequence, 32), dtype=torch.float16),
            torch.ones((1, sequence, 32), dtype=torch.float16),
            torch.zeros((1, 32, 128, 128), dtype=dtype),
            torch.tensor([sequence], dtype=torch.int16))


def test_select_none_never_imports_vendor_and_binds_complete_base():
    module = adapter_module()
    base = ModuleFusedOps(module)
    with patch("qwen35_moe_dflash.fused.ops.importlib.import_module") as load:
        ops = SelectiveFusedOps(base)
        assert ops.rms_norm(1, 2, 3) is module.results["rms_norm"]
        assert ops.moe(1, 2, 3, 4, top_k=2) is module.results["moe"]
    load.assert_not_called()
    assert isinstance(ops, FusedOps)
    assert ops.fused_ops == frozenset()
    assert base.backend_id in ops.backend_id


@pytest.mark.parametrize("roles", ["", "none,rms", "rms,rms", "attention", ["rms", 4]])
def test_selection_is_strict_and_does_not_load_vendor(roles):
    with patch("qwen35_moe_dflash.fused.ops.importlib.import_module") as load:
        with pytest.raises(ValueError, match="fused_ops"):
            selective(fused_ops=roles)
    load.assert_not_called()


def test_reference_base_requires_second_explicit_opt_in():
    module = adapter_module()
    module.IS_REFERENCE_BACKEND = True
    base = ModuleFusedOps(module, allow_reference=True)
    with pytest.raises(ValueError, match="reference base"):
        SelectiveFusedOps(base)
    ops = SelectiveFusedOps(base, allow_reference=True)
    assert ops.is_reference_backend
    assert "test-adapter-v1" in ops.backend_id


def test_vendor_identity_and_state_dtype_are_explicit():
    base = ModuleFusedOps(adapter_module())
    with pytest.raises(ValueError, match="vendor_backend_id"):
        SelectiveFusedOps(base, fused_ops="rms")
    with pytest.raises(ValueError, match="initial-state dtype"):
        selective(fused_ops="gdn")
    with pytest.raises(ValueError, match="state_casts"):
        selective(gdn_state_casts="implicit")


def test_only_selected_ops_need_registration_and_vendor_failure_is_not_retried(documented_vendor):
    namespace, calls, load = documented_vendor
    del namespace.npu.npu_chunk_gated_delta_rule
    module = adapter_module()
    ops = selective(ModuleFusedOps(module), fused_ops="rms")
    load.assert_called_once_with("torch_npu")
    assert ops.attention(1, 2, 3, query_start=0, scale=.5) is module.results["attention"]
    with pytest.raises(ValueError, match="not registered"):
        selective(fused_ops="gdn", gdn_initial_state_dtype=torch.float32)
    failure = RuntimeError("vendor kernel failed")
    def fail(*args):
        raise failure
    namespace.npu.adn_rms_norm = fail
    ops = selective(ModuleFusedOps(module), fused_ops="rms")
    with pytest.raises(RuntimeError) as error:
        ops.rms_norm(torch.ones(1, 4, dtype=torch.float16), torch.ones(4, dtype=torch.float16), 1e-6)
    assert error.value is failure
    assert [name for name, _, _ in module.calls] == ["attention"]


def test_documented_rms_preserves_gamma_and_checks_both_outputs(documented_vendor):
    namespace, calls, _ = documented_vendor
    x = torch.ones(1, 2, 4, dtype=torch.float16)
    gamma = torch.full((4,), 1.5, dtype=torch.float16)
    ops = selective(fused_ops="rms")
    torch.testing.assert_close(ops.rms_norm(x, gamma, 1e-6), x)
    assert calls[0][1][1] is gamma
    assert calls[0][1][2] == 1e-6
    namespace.npu.adn_rms_norm = lambda *args: (x, torch.ones(1, 2, 1, dtype=torch.float16))
    with pytest.raises(ValueError, match="rstd"):
        selective(fused_ops="rms").rms_norm(x, gamma, 1e-6)


def test_documented_conv_uses_padded_tail_and_device_int32_control(documented_vendor):
    _, calls, _ = documented_vendor
    projected = torch.arange(2, dtype=torch.float16)[None, :, None].expand(1, 2, 8192).contiguous()
    history = torch.full((1, 8192, 3), 3, dtype=torch.float16)
    new_history, padded = selective(fused_ops="conv_state").conv_state_update(projected, history)
    control = calls[0][1][2]
    assert control.dtype == torch.int32 and control.shape == (128,) and control.device == projected.device
    assert not control.any()
    torch.testing.assert_close(new_history, padded[..., -3:], rtol=0, atol=0)
    padded.zero_()
    assert new_history[0, 0].tolist() == [3, 0, 1]
    with pytest.raises(ValueError, match="projected"):
        selective(fused_ops="conv_state").conv_state_update(projected[..., :48], history[:, :48])


@pytest.mark.parametrize("sequence,chunk", [(1, 1), (2, 64)])
@pytest.mark.parametrize("normalize", [False, True])
def test_documented_gdn_signature_l2_and_explicit_output_cast(documented_vendor, sequence, chunk, normalize):
    _, calls, _ = documented_vendor
    inputs = documented_gdn_inputs(sequence=sequence)
    ops = selective(fused_ops="gdn", gdn_initial_state_dtype=torch.float16, gdn_state_casts="allow")
    core, state = ops.chunk_gated_delta_rule(*inputs, chunk_size=chunk, use_qk_l2norm_in_kernel=normalize)
    args = calls[0][1]
    assert args[0] is inputs[0] and args[1] is inputs[1]  # No external Q/K L2.
    assert args[5] == chunk and args[6] is inputs[5] and args[7] is inputs[6]
    assert args[8:] == (1, int(normalize))
    assert core.dtype == state.dtype == torch.float16


def test_gdn_cast_policy_rejects_before_kernel_and_allows_only_declared_boundaries(documented_vendor):
    _, calls, _ = documented_vendor
    half = documented_gdn_inputs()
    rejecting = selective(fused_ops="gdn", gdn_initial_state_dtype=torch.float16)
    with pytest.raises(ValueError, match="state ports"):
        rejecting.chunk_gated_delta_rule(*half, chunk_size=64, use_qk_l2norm_in_kernel=True)
    assert calls == []
    converting = selective(fused_ops="gdn", gdn_initial_state_dtype=torch.float32, gdn_state_casts="allow")
    _, result = converting.chunk_gated_delta_rule(*half, chunk_size=64, use_qk_l2norm_in_kernel=True)
    assert calls[-1][1][6].dtype == torch.float32 and result.dtype == torch.float16
    full = documented_gdn_inputs(dtype=torch.float32)
    exact = selective(fused_ops="gdn", gdn_initial_state_dtype=torch.float32)
    _, result = exact.chunk_gated_delta_rule(*full, chunk_size=64, use_qk_l2norm_in_kernel=True)
    assert calls[-1][1][6] is full[5] and result.dtype == torch.float32


def test_gdn_native_output_dtype_is_checked_before_optional_cast(documented_vendor):
    namespace, _, _ = documented_vendor
    namespace.npu.npu_chunk_gated_delta_rule = lambda *args: (args[2], args[6].half())
    ops = selective(fused_ops="gdn", gdn_initial_state_dtype=torch.float16, gdn_state_casts="allow")
    with pytest.raises(ValueError, match="last_recurrent_state"):
        ops.chunk_gated_delta_rule(*documented_gdn_inputs(), chunk_size=64, use_qk_l2norm_in_kernel=True)
