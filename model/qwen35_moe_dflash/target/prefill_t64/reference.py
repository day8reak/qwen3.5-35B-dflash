"""Compute the independent CPU and audited 310P references for the same four-layer T64 Prefill."""
import argparse
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import sys


POLICY = "FOUR_LAYER_NZ_PREFILL_WITH_L3_INDEPENDENT_ATTENTION_MATH"


def repo_root(explicit, entry):
    marker = Path("model/qwen35_moe_dflash/target/prefill_fixture.py")
    candidates = [Path(explicit)] if explicit else entry.resolve().parents
    for path in candidates:
        if (path / marker).is_file():
            return path.resolve()
    raise ValueError("Supply the model repository with --repo or REPO_DIR")


def stored_value(item, fixture, read_array, device):
    import torch

    if isinstance(item, dict):
        return torch.from_numpy(read_array(fixture, item)).to(device)
    if isinstance(item, list):
        return [stored_value(child, fixture, read_array, device) for child in item]
    return item


def restore_inputs(records, fixture, read_array, device):
    import torch

    args = {name: stored_value(item, fixture, read_array, "cpu" if name == "output_pos" else device)
            for name, item in records.items()}
    # JSON erases tuple types. The NZ wrapper only returns a layer's caches for a tuple pair.
    pairs = args["past_key_values"]
    if (not isinstance(pairs, list) or len(pairs) != 4
            or any(not isinstance(pair, list) or len(pair) != 2
                   or not all(isinstance(t, torch.Tensor) for t in pair) for pair in pairs)):
        raise ValueError("Expected four serialized cache tensor pairs")
    args["past_key_values"] = [tuple(pair) for pair in pairs]
    return args


@contextmanager
def bind_core(npu, fixture, baseline, read_array):
    """Supply independent attention only if the live call matches the original audit."""
    import numpy as np
    import torch

    rows = baseline["attention_audit"]["calls"]
    if len(rows) != 1:
        raise ValueError("Expected one captured L3 attention call")
    row = rows[0]
    contract = row["reference"]
    expected = {r["name"]: read_array(fixture, r) for r in row["inputs"]}
    logical = read_array(fixture, row["independent_reference"])
    heads, dim = contract["num_heads"], contract["head_dim"]
    if (contract["status"] != "COMPUTED" or contract["kv_length"] != 64
            or logical.shape != (1, heads, 64, dim) or logical.dtype != np.float16
            or dim % 16 or not np.isfinite(logical).all()):
        raise ValueError("Expected the finite independent T64 attention output")
    query_shape = (1, heads * (dim // 16), 64, 16)
    if expected["query"].shape != query_shape:
        raise ValueError("Expected a packed T64 query")
    packed = logical.reshape(1, heads, 64, dim // 16, 16).transpose(0, 1, 3, 2, 4)
    packed = np.ascontiguousarray(packed.reshape(query_shape))
    scalars = {
        "actual_seq_lengths_q": [64],
        "actual_seq_lengths_kv": [expected["mask"].shape[-1]],
        "num_heads": heads,
        "num_key_value_heads": contract["num_key_value_heads"],
        "input_layout": "BNSD",
        "inner_precise": 2,
        "block_size": expected["key"].shape[2],
        "scale_value": contract["scale_value"],
    }
    keys = set(scalars) | {"query", "key", "value", "atten_mask", "block_table", "all_seq_lengths_q"}
    calls = 0

    def from_audit(*args, **kwargs):
        nonlocal calls
        calls += 1
        if args or calls != 1 or set(kwargs) != keys:
            raise ValueError("Unexpected attention call signature or count")
        lengths = kwargs["all_seq_lengths_q"]
        if torch.is_tensor(lengths):
            lengths = lengths.tolist()
        if list(lengths) != [64] or any(kwargs[name] != value for name, value in scalars.items()):
            raise ValueError("Live L3 attention scalar arguments differ from the T64 audit")
        if len(kwargs["key"]) != 1 or len(kwargs["value"]) != 1:
            raise ValueError("Expected one key cache and one value cache")
        live = {
            "query": kwargs["query"], "key": kwargs["key"][0], "value": kwargs["value"][0],
            "mask": kwargs["atten_mask"], "table": kwargs["block_table"],
        }
        for name, tensor in live.items():
            actual = tensor.detach().cpu().contiguous().numpy()
            ref = expected[name]
            if actual.shape != ref.shape or actual.dtype != ref.dtype or actual.tobytes() != ref.tobytes():
                raise ValueError(f"Live L3 attention operand differs from the T64 audit: {name}")
        return torch.from_numpy(packed).to(kwargs["query"].device, copy=True)

    original = npu.adn_fused_infer_attention
    npu.adn_fused_infer_attention = from_audit
    try:
        yield
        if calls != 1:
            raise ValueError(f"Expected one L3 attention call, observed {calls}")
    finally:
        npu.adn_fused_infer_attention = original


def source_config(source_root, config_path, baseline):
    import yaml
    from qwen35_moe_dflash.target.layer_probe.common import sha

    cfg = yaml.safe_load(config_path.read_text())
    for name in ("model_file", "quanted_pth", "embedding_weight_path", "embedding_scale_path"):
        cfg[name] = str((source_root / cfg[name]).resolve(strict=True))
        if Path(cfg[name]) != Path(baseline["resources"][name]).resolve():
            raise ValueError(f"Model resource differs from the T64 fixture: {name}")
    sources = [
        source_root / "inference.py", source_root / "utils.py", config_path,
        source_root / "models/export_model_wrapper_qwen3_5_moe.py",
        source_root / "models/modeling_qwen3_5_moe_hiai_nz.py",
    ]
    if [sha(path) for path in sources] != [row["sha256"] for row in baseline["sources"]]:
        raise ValueError("Original 310P source or configuration differs from the T64 fixture")
    return cfg


def reference(args):
    import numpy as np
    import torch
    from qwen35_moe_dflash.target import prefill_fixture as pf
    from qwen35_moe_dflash.target.original_quant import load_embedding
    from qwen35_moe_dflash.target.layer_probe.common import metrics, new_output, read_array, save_json, sha
    from qwen35_moe_dflash.target.prefill_t64.cpu_reference import CpuPrefill, verify_floating_weights
    from qwen35_moe_dflash.target.prefill_t64.qualification import compare_outputs, output_records, select_golden

    fixture = args.fixture.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    config_path = (source_root / args.config).resolve(strict=True)
    manifest = fixture / "manifest.json"
    baseline = json.loads(manifest.read_text())
    if (baseline.get("status") != "CAPTURED" or baseline["sequence_length"] != 64
            or baseline["layers"] != [0, 1, 2, 3] or baseline["input_contract"]["output_pos"] != 63):
        raise ValueError("Expected the original four-layer T64 fixture")
    out = args.output.resolve()
    for original in (source_root, fixture):
        if out == original or out in original.parents or original in out.parents:
            raise ValueError("Choose a distinct output outside the original source and fixture")
    cfg = source_config(source_root, config_path, baseline)

    specs = baseline["outputs"]
    if [r["semantic_name"] for r in specs] != ["logits", *pf.STATE_NAMES]:
        raise ValueError("T64 fixture output names differ from the original capture contract")
    out = new_output(out)
    report = {"status": "STARTING", "original_fixture_sha256": sha(manifest),
              "scope": "FOUR_LAYER_FRESH_T64_LAST_POSITION", "comparison": {},
              "cpu_input_basis": "TOKEN_LOOKUP_IN_ORIGINAL_EXTERNAL_EMBEDDING_TABLES",
              "corrected_reference_basis": POLICY, "selected_golden": None,
              "historical_core_audit": baseline["attention_audit"]["status"]}
    try:
        source = pf.import_source(source_root)
        import torch_npu

        report["runtime"] = {"device": args.device, "torch": str(torch.__version__),
                             "torch_npu": getattr(torch_npu, "__version__", None)}

        with pf.hf_config_prefix(cfg["model_file"], 4, report), pf.working_directory(source_root):
            wrapper = pf.build_four_layers(source, cfg["model_file"], report)
        cpu_args = restore_inputs(baseline["source_arguments"], fixture, read_array, "cpu")
        tokens = torch.from_numpy(read_array(fixture, baseline["tokens"])).long()
        if not torch.equal(cpu_args["input_ids"], tokens):
            raise ValueError("Captured wrapper input IDs differ from the T64 fixture")
        size = wrapper.model.config
        embedding, scale = load_embedding(cfg["embedding_weight_path"], cfg["embedding_scale_path"],
                                          (size.vocab_size, size.hidden_size))
        if (tokens < 0).any() or (tokens >= size.vocab_size).any():
            raise ValueError("T64 token outside embedding vocabulary")
        for name, tensor in (("inputs_embeds", embedding), ("embed_scale", scale)):
            lookup = torch.from_numpy(tensor[tokens.numpy()])
            captured = cpu_args[name]
            if lookup.dtype != captured.dtype or not torch.equal(lookup.view(torch.uint8), captured.view(torch.uint8)):
                raise ValueError(f"CPU token lookup differs from captured input: {name}")
            cpu_args[name] = lookup
        cache_before = dict(zip(pf.STATE_NAMES[-2:], (v.numpy() for v in cpu_args["past_key_values"][3])))
        table = wrapper.model.model.layers[3].self_attn.block_table.cpu().numpy().copy()
        core_call, = baseline["attention_audit"]["calls"]
        table_spec, = (s for s in core_call["inputs"] if s["name"] == "table")
        if not np.array_equal(table, read_array(fixture, table_spec)):
            raise ValueError("Original page table differs from the fixture")
        report["stage"] = "CPU_REFERENCE"
        report["floating_checkpoint_parameters"] = verify_floating_weights(wrapper.model, cfg["model_file"])
        cpu_outputs, cpu_layers = CpuPrefill(wrapper, cfg["quanted_pth"]).run(cpu_args)
        if any(not torch.isfinite(v).all() for v in (*cpu_outputs, *cpu_layers)):
            raise ValueError("Nonfinite CPU reference")
        cpu_records = output_records(out, "cpu/outputs", cpu_outputs, specs)
        report["cpu_outputs"] = cpu_records
        report["comparison"]["historical_310p"] = compare_outputs(out, cpu_records, fixture, specs, cache_before, table)
        print("[target-prefill] CPU outputs saved:", out / "cpu/outputs", flush=True)
        report["stage"] = "NATIVE_310P"
        save_json(out / "report.json", report)
        device_name = getattr(getattr(torch_npu, "npu", None), "get_device_name", None)
        if callable(device_name):
            report["runtime"]["device_name"] = device_name(torch.device(args.device).index)
        source.quant_model_inplace(
            wrapper, cfg["quanted_pth"], wrapper.model.config.num_experts,
            start_layer_idx=0, device="cpu", pseudo_quant=False, quant_bits=4,
        )
        wrapper.to(args.device).eval()

        def run_device(folder, core_context):
            work = out / folder
            work.mkdir(exist_ok=True)
            operands = restore_inputs(baseline["source_arguments"], fixture, read_array, args.device)
            boundary = []

            def observe(module, arguments, result):
                boundary.append(result[0].detach())

            handles = [layer.register_forward_hook(observe) for layer in wrapper.model.model.layers]
            try:
                with torch.no_grad(), pf.working_directory(work), core_context:
                    result = wrapper(**operands)
            finally:
                for handle in handles:
                    handle.remove()
            if len(boundary) != len(cpu_layers):
                raise ValueError("Missing decoder-layer output in 310P forward")
            records = output_records(out, folder + "/outputs", result, specs)
            comparison = compare_outputs(out, cpu_records, out, records, cache_before, table)
            comparison["decoder_hidden"] = [metrics(x.cpu().contiguous().numpy(), y.numpy())
                                            for x, y in zip(boundary, cpu_layers)]
            comparison["outputs_status"] = comparison["status"]
            failed_layers = [i for i, row in enumerate(comparison["decoder_hidden"]) if not row["passed"]]
            comparison["first_failed_decoder_layer"] = failed_layers[0] if failed_layers else None
            if failed_layers:
                comparison["status"] = "FAILED"
            return records, comparison

        native_dir = out / "native-310p"
        audit = {}
        observer = pf.PrefillCapture(native_dir, tokens, size, size.kv_cache_max_len, audit)
        native, comparison = run_device("native-310p", pf.attention_audit(torch_npu, observer))
        save_json(native_dir / "audit.json", audit)
        report["native_core_audit"] = audit["attention_audit"]["status"]
        report["native_outputs"] = native
        report["comparison"]["native_310p"] = comparison
        report["native_repeat"] = {"status": "NOT_RUN"}
        print("[target-prefill] native 310P vs CPU:", comparison["status"],
              "L3 core:", report["native_core_audit"], flush=True)
        if comparison["status"] == "PASSED" and report["native_core_audit"] == "PASSED":
            repeat, repeat_cmp = run_device("native-repeat", nullcontext())
            changed = [x["name"] for x, y in zip(native, repeat) if x["sha256"] != y["sha256"]]
            report["comparison"]["native_repeat"] = repeat_cmp
            report["native_repeat"] = {"status": "FAILED" if changed else "PASSED", "changed_outputs": changed}
        report["selected_golden"] = select_golden(report, out)
        report["stage"] = "L3_MATH_DIAGNOSTIC"
        save_json(out / "report.json", report)
        try:
            records, corrected = run_device("l3-math-310p", bind_core(torch_npu, native_dir, audit, read_array))
            changed = [x["name"] for x, y in zip(records[1:], native[1:]) if x["sha256"] != y["sha256"]]
            corrected["changed_states_vs_native"] = changed
            corrected["attention_operands_bitwise_equal_to_native"] = True
            if changed:
                corrected["status"] = "NOT_ISOLATED"
            report["corrected_outputs"] = records
        except (ValueError, KeyError, RuntimeError) as exc:
            corrected = {"status": "UNAVAILABLE", "failed_outputs": [], "error": f"{type(exc).__name__}: {exc}"}
        report["comparison"]["l3_math_310p"] = corrected
        report.update(stage="COMPLETE", status="QUALIFIED" if report["selected_golden"] else "NUMERICAL_FAILED")
    except Exception as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_json(out / "report.json", report)
    fields = ("status", "failed_outputs", "first_failed_decoder_layer", "error")
    comparisons = {name: {key: row[key] for key in fields if key in row}
                   for name, row in report["comparison"].items()}
    golden = report["selected_golden"]
    return {"status": report["status"], "native_core_audit": report["native_core_audit"],
            "comparisons": comparisons, "golden": golden["basis"] if golden else None,
            "report": str(out / "report.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "config", "fixture", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=os.environ.get("REPO_DIR"))
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    entry = Path(__file__).resolve()
    try:
        repo = repo_root(args.repo, entry)
    except ValueError as exc:
        parser.error(str(exc))
    # A standalone job copy can sit beside CPro source files with the same module names as NZ.
    sys.path[:] = [item for item in sys.path if Path(item).resolve() != entry.parent]
    sys.path.insert(0, str(repo / "model"))
    result = reference(args)
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] != "QUALIFIED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
