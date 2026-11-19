"""Replay four original 310P prefill layers with audited T64 attention through the LM head."""
import argparse
from contextlib import contextmanager
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
        "actual_seq_lengths_kv": [2048],
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


def reference(args):
    import numpy as np
    import torch
    import yaml
    from qwen35_moe_dflash.target import prefill_fixture as pf
    from qwen35_moe_dflash.target.layer_probe.common import checked, identity, new_output, read_array, save_json, sha

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
    cfg = yaml.safe_load(config_path.read_text())
    for name in ("model_file", "quanted_pth"):
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

    out = new_output(out)
    report = {"reference_basis": POLICY, "status": "STARTING", "original_fixture_sha256": sha(manifest), "outputs": []}
    try:
        source = pf.import_source(source_root)
        import torch_npu

        with pf.hf_config_prefix(cfg["model_file"], 4, report), pf.working_directory(source_root):
            wrapper = pf.build_four_layers(source, cfg["model_file"], report)
        source.quant_model_inplace(
            wrapper, cfg["quanted_pth"], wrapper.model.config.num_experts,
            start_layer_idx=0, device="cpu", pseudo_quant=False, quant_bits=4,
        )
        wrapper.to(args.device).eval()
        operands = {
            name: stored_value(item, fixture, read_array, "cpu" if name == "output_pos" else args.device)
            for name, item in baseline["source_arguments"].items()
        }
        tokens = torch.from_numpy(read_array(fixture, baseline["tokens"])).long()
        if not torch.equal(operands["input_ids"].cpu(), tokens):
            raise ValueError("Captured wrapper input IDs differ from the T64 fixture")
        with torch.no_grad(), pf.working_directory(out), bind_core(torch_npu, fixture, baseline, read_array):
            result = wrapper(**operands)
        names = ("logits", *pf.STATE_NAMES)
        if len(result) != len(names) or [row["semantic_name"] for row in baseline["outputs"]] != list(names):
            raise ValueError("Unexpected original wrapper output order")
        outputs_dir = out / "outputs"
        outputs_dir.mkdir()
        for name, tensor, spec in zip(names, result, baseline["outputs"]):
            arr = tensor.detach().cpu().contiguous().numpy()
            if list(arr.shape) != spec["shape"] or arr.dtype.name != spec["dtype"] or not np.isfinite(arr).all():
                raise ValueError(f"Invalid recomputed wrapper output: {name}")
            path = outputs_dir / (name + ".bin")
            arr.tofile(path)
            record = dict(identity(path, out), name=name, shape=list(arr.shape), dtype=arr.dtype.name)
            report["outputs"].append(record)
            if name != "logits":
                checked(fixture, spec)
                if record["sha256"] != spec["sha256"]:
                    raise ValueError(f"{name} changed during reference recomputation")
        report.update(status="COMPUTED", attention_operand_match="BITWISE_ALL_FIVE",
                      state_outputs_match=len(pf.STATE_NAMES), logits_top1=int(result[0].argmax().item()),
                      whole_model_accuracy="UNVERIFIED")
    except Exception as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_json(out / "report.json", report)
    return {"status": report["status"], "reference_basis": POLICY, "report": str(out / "report.json")}


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
    print(json.dumps(reference(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
