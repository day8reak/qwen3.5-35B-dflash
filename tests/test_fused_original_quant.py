"""Small CPU reference tests; no vendor QPARAM or device validation is implied."""
import json
import struct
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from qwen35_moe_dflash.fused import FusedTargetModel, ModuleFusedOps, OriginalQuantOps, load_original_quantized_target
from qwen35_moe_dflash.fused.model import _configuration, required_weight_shapes
from qwen35_moe_dflash.ge_quant_plan import target_tensor_quantization
from qwen35_moe_dflash.ge_weights import checkpoint_key, tiny_weights
from qwen35_moe_dflash.original_quant import canonicalkeys


def write_tensor_file(path, items):
    header, chunks, offset = {}, [], 0
    for name, value in items.items():
        raw = value.tobytes()
        dtype = {np.dtype("int8"): "I8", np.dtype("float32"): "F32", np.dtype("float16"): "F16"}[value.dtype]
        header[name] = {"dtype": dtype, "shape": list(value.shape), "data_offsets": [offset, offset + len(raw)]}
        chunks.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))


def unpack_one(carrier):
    """Independent I4ZN inverse for a single [K,N/8] matrix, never a bank."""
    k, n = carrier.shape[0], carrier.shape[1] * 8
    packed = carrier.contiguous().view(torch.uint8).reshape(k // 64, n // 16, 16, 32)
    lo, hi = packed & 15, packed >> 4
    signed = torch.stack((lo, hi), -1).reshape(k // 64, n, 64).permute(1, 0, 2).reshape(n, k).short()
    return torch.where(signed >= 8, signed - 16, signed).float()


class ReferenceQuant:
    """Test backend, intentionally not a claimed implementation of vendor QPARAM."""
    def __init__(self):
        self.calls = []

    def npu_dynamic_quant(self, x):
        scale = x.float().abs().amax(-1).clamp_min(1e-8) / 127
        return (x.float() / scale[:, None]).round().clamp(-127, 127).to(torch.int8), scale

    def npu_trans_quant_param(self, scale, unused, mode):
        assert unused is None and mode == 1
        # Reversible mock encoding ONLY. Production uses its explicit vendor op.
        return scale.contiguous().view(torch.int32).to(torch.int64)

    def npu_quant_matmul(self, x, weight, scale, *, pertoken_scale, output_dtype):
        assert weight.dtype == torch.int32 and scale.dtype == torch.float32
        dequant = unpack_one(weight) * scale.T.repeat_interleave(256, -1)
        self.calls.append("QuantMatmul")
        return ((x.float() * pertoken_scale) @ dequant.T).to(output_dtype)

    def npu_moe_gating_top_k_softmax(self, logits, finished, *, k):
        assert finished is None
        self.calls.append("MoeGatingTopKSoftmax")
        prob, ids = logits.float().softmax(-1).topk(k, -1)
        return prob, ids, None

    def npu_moe_init_routing_v2(self, x, **kwargs):
        ids = kwargs["expert_idx"]
        count = kwargs["expert_num"]
        assert kwargs["quant_mode"] == -1 and kwargs["expert_tokens_num_type"] == 1
        order = ids.flatten().argsort(stable=True)
        rows = torch.arange(x.shape[0]).repeat_interleave(ids.shape[1])
        counts = torch.bincount(ids.flatten().long(), minlength=count)
        self.calls.append("MoeInitRoutingV2")
        return x[rows[order]], order, counts, None

    def npu_grouped_matmul(self, xs, weights, **kwargs):
        assert "bias" not in kwargs
        assert kwargs["group_list_type"] == 1 and kwargs["split_item"] == 3
        weight, qparam = weights[0], kwargs["scale"][0]
        assert weight.dtype == torch.int32 and qparam.dtype == torch.int64
        scale = qparam.to(torch.int32).contiguous().view(torch.float32)
        counts = kwargs["group_list"]
        x = xs[0].float() * kwargs["per_token_scale"][0][:, None]
        outputs, start = [], 0
        # Dequantize only each selected expert, not a complete float bank.
        for expert, size in enumerate(counts.tolist()):
            if size:
                w = unpack_one(weight[expert]) * scale[expert].T.repeat_interleave(256, -1)
                outputs.append(x[start:start + size] @ w.T)
                start += size
        self.calls.append("GroupedMatmul")
        return [torch.cat(outputs).to(kwargs["output_dtype"])]

    def npu_moe_finalize_routing(self, x, **kwargs):
        assert kwargs["drop_pad_mode"] == 2
        order, scales = kwargs["expanded_src_to_dst_row"], kwargs["scales"]
        restored = torch.empty_like(x)
        restored[order] = x
        self.calls.append("MoeFinalizeRouting")
        return (restored.reshape(*scales.shape, x.shape[-1]) * scales[..., None]).sum(1)


@pytest.fixture
def model_data(tmp_path):
    from reference import fused_ops
    torch.set_num_threads(1)
    c = dict(hidden_size=256, num_hidden_layers=2, vocab_size=32,
        num_attention_heads=16, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=8, linear_num_value_heads=16,
        linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, num_experts=16, num_experts_per_tok=2,
        moe_intermediate_size=256, shared_expert_intermediate_size=256,
        max_position_embeddings=64, layer_types=["linear_attention", "full_attention"],
        rms_norm_eps=1e-6, rope_parameters={"partial_rotary_factor": .5})
    shapes = required_weight_shapes(c)
    rng, quant, base, logical = np.random.default_rng(11), {}, {}, {}
    for key, shape in shapes.items():
        kind = target_tensor_quantization(key, shape)["kind"]
        if kind.endswith("int4"):
            weight = rng.integers(-3, 4, shape, dtype=np.int8)
            scale_shape = (*shape[:-1], shape[-1] // 256) + (() if len(shape) == 3 else (1,))
            scale = rng.uniform(.002, .008, scale_shape).astype(np.float32)
            wk, sk = canonicalkeys(key)
            quant[wk], quant[sk], logical[key] = weight, scale, (weight, scale)
        elif kind == "float16":
            base[checkpoint_key(key)[1]] = tiny_weights({key: shape})[key]
    base["model.language_model.layers.0.input_layernorm.weight"][:] = .25
    base["model.language_model.layers.0.linear_attn.norm.weight"][:] = 1.5
    hf, qdir = tmp_path / "hf", tmp_path / "quant"
    hf.mkdir(); qdir.mkdir()
    write_tensor_file(hf / "model.safetensors", base)
    write_tensor_file(qdir / "data0.safetensors", quant)
    embedding = rng.integers(-100, 101, (32, 256), dtype=np.int8)
    scales = rng.uniform(.001, .01, (32, 1)).astype(np.float32)
    embedding.tofile(tmp_path / "embedding.bin")
    scales.tofile(tmp_path / "scale.bin")
    backend, chunks = ReferenceQuant(), []
    module = ModuleType("recording_reference")
    module.__dict__.update({name: getattr(fused_ops, name) for name in dir(fused_ops) if not name.startswith("__")})
    def gdn(*args, **kwargs):
        chunks.append(kwargs["chunk_size"])
        return fused_ops.chunk_gated_delta_rule(*args, **kwargs)
    module.chunk_gated_delta_rule = gdn
    model = load_original_quantized_target(c, model_file=hf, quanted_pth=qdir,
        embedding_weight_path=tmp_path / "embedding.bin", embedding_scale_path=tmp_path / "scale.bin",
        feature_layers=(0, 1), ops=ModuleFusedOps(module, allow_reference=True),
        quant_ops=OriginalQuantOps(backend, backend_id="CPU_TEST_ONLY", is_reference=True, allow_reference=True))
    return model, logical, backend, embedding, scales, chunks


def test_original_quant_loader_prefill_decode_and_full_logits(model_data):
    model, _, backend, embedding, scales, chunks = model_data
    ids = torch.tensor([[3, 5]], dtype=torch.int32)
    actual_embedding = model.quantization.embedding(ids)
    expected = torch.from_numpy(embedding[[[3, 5]]].astype(np.float32) * scales[[[3, 5]]])
    torch.testing.assert_close(actual_embedding, expected, rtol=0, atol=0)
    assert all("experts" not in key and "embed_tokens" not in key for key in model._weight_names)
    before = model.initial_state()
    out = model(ids, before, return_logits=True)
    assert out.logits.shape == (1, 2, 32) and out.logits.dtype == torch.float32
    torch.testing.assert_close(out.top1, out.logits.argmax(-1).int(), rtol=0, atol=0)
    assert out.features.shape == (1, 2, 512) and out.state.position == 2
    assert before.position == 0 and torch.count_nonzero(before.layers[0].recurrent) == 0
    decoded = model(torch.tensor([[7]], dtype=torch.int32), out.state)
    assert decoded.logits is None and decoded.state.position == 3 and chunks == [64, 1]
    assert {"QuantMatmul", "GroupedMatmul", "MoeGatingTopKSoftmax", "MoeInitRoutingV2", "MoeFinalizeRouting"} <= set(backend.calls)
    assert model.original_quant_contract["accuracy"] == "UNVERIFIED"


def test_source_external_embedding_path_matches_token_path_and_selects_only_head_rows(model_data, monkeypatch):
    model, _, _, embedding, scales, _ = model_data
    ids = torch.tensor([[3, 5, 7]], dtype=torch.int32)
    expected = model(ids, return_logits=True)
    head_dtypes = []
    original_linear = model._linear

    def linear(x, prefix):
        if prefix == "target.lm_head":
            head_dtypes.append(x.dtype)
        return original_linear(x, prefix)

    monkeypatch.setattr(model, "_linear", linear)
    output = model.forward_embeddings(torch.from_numpy(embedding[ids.numpy()]),
        embed_scale=torch.from_numpy(scales[ids.numpy()]), output_pos=torch.tensor([-1, 0]))
    torch.testing.assert_close(output.logits, expected.logits[:, [2, 0]], rtol=0, atol=0)
    torch.testing.assert_close(output.top1, expected.top1[:, [2, 0]], rtol=0, atol=0)
    torch.testing.assert_close(output.features, expected.features, rtol=0, atol=0)
    assert output.features.dtype == torch.float16 and output.state.position == 3
    assert head_dtypes == [torch.float32, torch.float32]
    for actual, reference in zip(output.state.layers, expected.state.layers):
        for name in ("conv", "recurrent", "key", "value"):
            left, right = getattr(actual, name), getattr(reference, name)
            if left is not None:
                torch.testing.assert_close(left, right, rtol=0, atol=0)
    decoded = model(torch.tensor([[4]], dtype=torch.int32), output.state, output_pos=-1, return_logits=True)
    assert decoded.logits.shape == (1, 1, 32) and decoded.state.position == 4


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_external_float_embeddings_are_already_dequantized(model_data, dtype):
    model = model_data[0]
    x = torch.linspace(-.3, .3, 512).reshape(1, 2, 256).to(dtype)
    result = model.forward_embeddings(x, output_pos=-1)
    reference = model.forward_embeddings(x.float(), output_pos=-1)
    torch.testing.assert_close(result.logits, reference.logits, rtol=0, atol=0)
    assert result.logits.shape == (1, 1, 32)


@pytest.mark.parametrize("output_pos", [2, -3, True, [], [0, .5], torch.tensor([0.]), torch.tensor(0)])
def test_invalid_head_positions_fail_before_any_layer_compute(model_data, monkeypatch, output_pos):
    model = model_data[0]
    monkeypatch.setattr(model, "_linear", lambda *a, **kw: pytest.fail("must validate positions before compute"))
    with pytest.raises(ValueError, match="output_pos"):
        model(torch.tensor([[3, 5]], dtype=torch.int32), output_pos=output_pos)
    with pytest.raises(ValueError, match="output_pos"):
        model.forward_embeddings(torch.zeros(1, 2, 256), output_pos=output_pos)


@pytest.mark.parametrize("failure", ["missing-scale", "half-scale", "scale-shape", "negative-scale",
                                     "nonfinite-scale", "double-scale", "nonfinite-input", "overflow"])
def test_external_embeddings_reject_invalid_scale_or_dtype_before_compute(model_data, monkeypatch, failure):
    model = model_data[0]
    x, scale = torch.ones(1, 2, 256, dtype=torch.int8), torch.ones(1, 2, 1)
    if failure == "missing-scale":
        scale = None
    elif failure == "half-scale":
        scale = scale.half()
    elif failure == "scale-shape":
        scale = torch.ones(1, 2)
    elif failure == "negative-scale":
        scale.fill_(-1)
    elif failure == "nonfinite-scale":
        scale.fill_(float("nan"))
    elif failure == "double-scale":
        x = x.float()
    elif failure == "nonfinite-input":
        x, scale = x.float().fill_(float("inf")), None
    else:
        x.fill_(127)
        scale.fill_(torch.finfo(torch.float32).max)
    monkeypatch.setattr(model, "_linear", lambda *a, **kw: pytest.fail("must validate inputs before compute"))
    with pytest.raises(ValueError):
        model.forward_embeddings(x, embed_scale=scale)


def test_original_gated_norm_rounds_after_fp32_silu_product(model_data):
    model = model_data[0]
    generator = torch.Generator().manual_seed(9)
    core = torch.randn(1, 3, 16, 16, generator=generator).half()
    z = torch.randn(core.shape, generator=generator).half()
    prefix = "target.layers.0.linear_attn.norm"
    gamma = model.weight(prefix + ".weight").float()
    normalized = core.float() * torch.rsqrt(core.float().square().mean(-1, keepdim=True) + 1e-6)
    expected = (normalized * gamma * F.silu(z.float())).half()
    actual = model._gated_norm(core, z, prefix)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    premature_rounding = (normalized * gamma).half() * F.silu(z)
    assert not torch.equal(actual, premature_rounding)


def test_original_rope_keeps_positions_and_trigonometry_fp32(model_data):
    model = model_data[0]
    x = torch.linspace(-1, 1, 32).reshape(1, 1, 2, 16).half()
    width, start = model.config["rotary_dim"], 4097
    frequency = torch.arange(start, start + 2, dtype=torch.float32)[:, None] * model.inv_freq[None, :]
    angle = torch.cat((frequency, frequency), -1)[None, None]
    first, second = x[..., :width].chunk(2, -1)
    rotated = x[..., :width] * angle.cos().half() + torch.cat((-second, first), -1) * angle.sin().half()
    expected = torch.cat((rotated, x[..., width:]), -1)
    assert model.inv_freq.dtype == torch.float32
    torch.testing.assert_close(model._rotary(x, start), expected, rtol=0, atol=0)


def test_original_decay_uses_fp32_arithmetic_before_declared_fp16_gdn_port(model_data, monkeypatch):
    model = model_data[0]
    prefix = "target.layers.0.linear_attn"
    seen = {}
    original_linear = model._linear
    original_ops = model.ops
    module = ModuleType("capture_gate")
    module.FUSED_OPS_ABI = 1
    module.BACKEND_ID = "CPU_TEST_ONLY"
    module.IS_REFERENCE_BACKEND = True
    for name in ModuleFusedOps.required_operations:
        setattr(module, name, getattr(original_ops, name))

    def linear(x, name):
        output = original_linear(x, name)
        if name == prefix + ".in_proj_a":
            seen["a"] = output.clone()
        return output

    def gdn(q, k, v, g, beta, *args, **kwargs):
        seen["g"] = g.clone()
        return original_ops.chunk_gated_delta_rule(q, k, v, g, beta, *args, **kwargs)

    module.chunk_gated_delta_rule = gdn
    monkeypatch.setattr(model, "_linear", linear)
    model.ops = ModuleFusedOps(module, allow_reference=True)
    model._gdn(torch.linspace(-.5, .5, 512).reshape(1, 2, 256), prefix, model.initial_state().layers[0])
    expected = (-model.weight(prefix + ".A_log").float().exp()
                * F.softplus(seen["a"].float() + model.weight(prefix + ".dt_bias").float())).half()
    assert seen["g"].dtype == torch.float16
    torch.testing.assert_close(seen["g"], expected, rtol=0, atol=0)


def test_original_linear_matches_independent_group_dequantization(model_data):
    model, logical, backend, *_ = model_data
    key = "target.layers.0.linear_attn.in_proj_qkv.weight"
    x = torch.linspace(-.25, .35, 512).reshape(1, 2, 256).half()
    actual = model.quantization.linear(x, key)
    weight, scale = logical[key]
    dequant = torch.from_numpy(weight.astype(np.float32) * np.repeat(scale.squeeze(-1), 256, -1))
    xq, xs = backend.npu_dynamic_quant(x.reshape(-1, 256))
    expected = ((xq.float() * xs[:, None]) @ dequant.T).half().reshape(actual.shape)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_raw_norm_weights_add_one_only_in_main_norm(model_data):
    model = model_data[0]
    main = model._norm(torch.ones(1, 1, 256).half(), "target.layers.0.input_layernorm")
    gated = model._norm(torch.ones(1, 1, 16).half(), "target.layers.0.linear_attn.norm", zero_centered=False)
    torch.testing.assert_close(main, torch.full_like(main, 1.25), atol=.001, rtol=0)
    torch.testing.assert_close(gated, torch.full_like(gated, 1.5), atol=.001, rtol=0)


def test_grouped_routing_matches_tokenwise_selected_expert_reference(model_data):
    model, logical, backend, *_ = model_data
    prefix = "target.layers.0.mlp"
    x = torch.linspace(-.8, .9, 3 * 256).reshape(1, 3, 256).half()

    def evaluate(rows, key, expert=None):
        raw, scale = logical[key]
        if expert is not None:
            raw, scale = raw[expert], scale[expert]
        else:
            scale = scale.squeeze(-1)
        weight = torch.from_numpy(raw.astype(np.float32) * np.repeat(scale, 256, -1))
        q, s = backend.npu_dynamic_quant(rows)
        return ((q.float() * s[:, None]) @ weight.T).half()

    logits = evaluate(x[0], prefix + ".gate.weight")
    probability, selected = logits.float().softmax(-1).topk(2, -1)
    probability = (probability / probability.sum(-1, keepdim=True)).half()
    expected = []
    for row in range(3):
        contributions = []
        for rank, expert in enumerate(selected[row].tolist()):
            hidden = evaluate(x[0, row:row + 1], prefix + ".experts.gate_up_proj", expert)
            gate, up = hidden.chunk(2, -1)
            hidden = evaluate(F.silu(gate) * up, prefix + ".experts.down_proj", expert)
            contributions.append(hidden * probability[row, rank])
        expected.append(torch.stack(contributions).sum(0))
    actual = model.quantization.moe(x, prefix, top_k=2)
    torch.testing.assert_close(actual, torch.cat(expected).unsqueeze(0), rtol=0, atol=2e-6)


def test_quant_backend_requires_capabilities_and_explicit_reference_opt_in():
    with pytest.raises(RuntimeError, match="npu_grouped_matmul"):
        OriginalQuantOps(SimpleNamespace(), backend_id="missing", is_reference=False)
    with pytest.raises(ValueError, match="allow_reference"):
        OriginalQuantOps(ReferenceQuant(), backend_id="ref", is_reference=True)


def test_real_npu_binding_rejects_default_cpu_before_reading_weights(model_data):
    model = model_data[0]
    # Reference method bodies are never called: this only checks the device
    # binding preflight of a declared non-reference backend, without an NPU.
    backend = OriginalQuantOps(ReferenceQuant(), backend_id="vendor-contract-only", is_reference=False)
    with pytest.raises(ValueError, match="requires device='npu:0'"):
        load_original_quantized_target(model.config, model_file="missing", quanted_pth="missing",
            embedding_weight_path="missing", embedding_scale_path="missing",
            feature_layers=(0, 1), ops=model.ops, quant_ops=backend)


def test_target_uses_smaller_kv_capacity_without_mutating_prior_state(model_data):
    source = model_data[0]
    c = dict(source.config, kv_cache_max_len=2)
    weights = {key: source.weight(key) for key in source._weight_names}
    model = FusedTargetModel(c, weights, feature_layers=(0, 1), ops=source.ops,
                            quantization=source.quantization)
    assert model.cache_capacity == 2
    with pytest.raises(ValueError, match="context capacity"):
        model(torch.tensor([[1, 2, 3]], dtype=torch.int32))
    complete = model(torch.tensor([[1, 2]], dtype=torch.int32))
    with pytest.raises(ValueError, match="context capacity"):
        model(torch.tensor([[3]], dtype=torch.int32), complete.state)
    assert complete.state.position == 2
    model.config["kv_cache_max_len"] = 9999
    assert model.cache_capacity == 64


@pytest.mark.parametrize("capacity", [True, 0, -1, "2048", None])
def test_target_rejects_invalid_optional_kv_capacity(model_data, capacity):
    with pytest.raises(ValueError, match="kv_cache_max_len"):
        _configuration(dict(model_data[0].config, kv_cache_max_len=capacity), (0, 1))
