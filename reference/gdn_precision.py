"""Host-only GDN arithmetic diagnostics; no accuracy gate or device claim.

NPZ keys are q/k/v [1,S,H,K/K/V] and g/beta [1,S,H]. They are the
pre-L2-normalization GDN inputs, with nonpositive decay g and sigmoid beta.
Both arithmetic paths consume the same FP16-rounded inputs. The FP32 path
uses the pinned Transformers chunk64 implementation and recomputes each
prefix from zero state to expose the reference state at each chunk boundary.
This bounded optional diagnostic accepts S<=512, H<=8, K/V<=128 and
H*ceil(S/64)<=16. Slice a short token prefix and a few heads for larger samples.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import math
from pathlib import Path
import sys
import zipfile

if __name__ == "__main__":
    from qwen35_moe_dflash.backends import configure_reference_backend
    configure_reference_backend("cpu")

import numpy as np
import torch

from qwen35_moe_dflash.artifacts import output_path, save_json
from qwen35_moe_dflash.ge_ir import Graph
from qwen35_moe_dflash.ge_target import _Builder, _chunk_gdn
from qwen35_moe_dflash.hashing import sha256_file
from reference.ge_torch import evaluate_torch


NAMES = ("q", "k", "v", "g", "beta")
MAX_NPZ_BYTES = 8 << 20
LIMITS = {"sequence": 512, "heads": 8, "key_dim": 128, "value_dim": 128, "head_chunks": 16}


def _validate_shapes(metadata):
    if set(metadata) != set(NAMES):
        raise ValueError("GDN NPZ must contain exactly q, k, v, g, beta")
    for name, (shape, dtype) in metadata.items():
        if dtype not in (np.dtype("float16"), np.dtype("float32"), np.dtype("float64")):
            raise ValueError(f"{name}: expected a float16/float32/float64 array")
        if len(shape) != (4 if name in ("q", "k", "v") else 3) or any(dim <= 0 for dim in shape):
            raise ValueError(f"{name}: expected nonempty GDN dimensions")
    q, k, v, g, beta = (metadata[name][0] for name in NAMES)
    if q[0] != 1 or k != q or v[:3] != q[:3] or g != q[:3] or beta != q[:3]:
        raise ValueError("GDN needs matching batch=1, sequence/head axes and q/k dimensions")
    if (q[1] > LIMITS["sequence"] or q[2] > LIMITS["heads"] or q[3] > LIMITS["key_dim"]
            or v[3] > LIMITS["value_dim"] or q[2] * ((q[1] + 63) // 64) > LIMITS["head_chunks"]):
        raise ValueError("GDN diagnostic size limit: S<=512, H<=8, K/V<=128, H*ceil(S/64)<=16; "
                         "slice a short token prefix and a few heads from the sample")


def metrics(reference, actual):
    """Report errors on finite pairs, with explicit nonfinite/zero-norm status."""
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if reference.shape != actual.shape:
        raise ValueError("metric shapes differ")
    valid = np.isfinite(reference) & np.isfinite(actual)
    ref, got = reference[valid], actual[valid]
    with np.errstate(over="ignore", invalid="ignore"):
        error = got - ref
    result = {"shape": list(reference.shape), "elements": int(reference.size),
              "compared_finite_elements": int(valid.sum()),
              "reference_nonfinite": int((~np.isfinite(reference)).sum()),
              "actual_nonfinite": int((~np.isfinite(actual)).sum()),
              "status": "FINITE" if valid.all() else "NONFINITE",
              "max_abs": None, "relative_l2": None, "relative_l2_status": "NO_FINITE_PAIRS"}
    if not valid.any():
        return result
    maximum = float(np.max(np.abs(error)))
    if not math.isfinite(maximum):
        result.update(status="METRIC_OVERFLOW", relative_l2_status="METRIC_OVERFLOW")
        return result
    result["max_abs"] = maximum
    scale = float(np.max(np.abs(ref)))
    if scale == 0:
        result.update(relative_l2=0.0 if maximum == 0 else None,
                      relative_l2_status="DEFINED" if maximum == 0 else "ZERO_REFERENCE_NORM")
        return result
    # Scale before norm evaluation so squared FP32 magnitudes cannot overflow.
    with np.errstate(over="ignore", invalid="ignore"):
        relative = 0.0 if maximum == 0 else ((maximum / scale) *
            float(np.linalg.norm(error / maximum) / np.linalg.norm(ref / scale)))
    result.update(relative_l2=relative if math.isfinite(relative) else None,
                  relative_l2_status="DEFINED" if math.isfinite(relative) else "METRIC_OVERFLOW")
    return result


def validate_inputs(values):
    if set(values) != set(NAMES):
        raise ValueError("GDN NPZ must contain exactly q, k, v, g, beta")
    arrays = {name: np.asarray(values[name]) for name in NAMES}
    _validate_shapes({name: (array.shape, array.dtype) for name, array in arrays.items()})
    q, k, v, g, beta = (arrays[name] for name in NAMES)
    if np.any(g[np.isfinite(g)] > 0):
        raise ValueError("GDN decay g must be nonpositive")
    finite_beta = beta[np.isfinite(beta)]
    if np.any((finite_beta < 0) | (finite_beta > 1)):
        raise ValueError("GDN beta must be sigmoid values in [0,1]")
    return arrays


def synthetic_inputs(seed=17):
    rng = np.random.default_rng(seed)
    return {**{name: rng.normal(0, .25, (1, 128, 2, 8)).astype(np.float32) for name in ("q", "k", "v")},
            "g": -rng.uniform(.01, .5, (1, 128, 2)).astype(np.float32),
            "beta": rng.uniform(.05, .95, (1, 128, 2)).astype(np.float32)}


def load_inputs(path):
    path = Path(path).expanduser().resolve(strict=True)
    if path.suffix.lower() != ".npz":
        raise ValueError("--inputs must be an NPZ file")
    if path.stat().st_size > MAX_NPZ_BYTES:
        raise ValueError("GDN diagnostic NPZ exceeds 8 MiB; slice a smaller sample")
    # Check uncompressed sizes and NPY shapes before NumPy allocates arrays.
    with zipfile.ZipFile(path) as archive:
        items = archive.infolist()
        if len(items) != len(NAMES) or {item.filename for item in items} != {name + ".npy" for name in NAMES}:
            raise ValueError("GDN NPZ must contain exactly q, k, v, g, beta without duplicates")
        if sum(item.file_size for item in items) > MAX_NPZ_BYTES:
            raise ValueError("GDN diagnostic expanded NPZ exceeds 8 MiB; slice a smaller sample")
        metadata = {}
        for item in items:
            with archive.open(item) as stream:
                version = np.lib.format.read_magic(stream)
                if version not in ((1, 0), (2, 0)):
                    raise ValueError("GDN NPZ requires standard NPY v1/v2 arrays")
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dtype = reader(stream)
                if dtype.hasobject or math.prod(shape) * dtype.itemsize != item.file_size - stream.tell():
                    raise ValueError("GDN NPZ array byte size does not match its header")
                metadata[item.filename[:-4]] = (shape, dtype)
        _validate_shapes(metadata)
    digest = sha256_file(path)
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("duplicate GDN NPZ keys")
        values = validate_inputs({name: archive[name].copy() for name in archive.files})
    if sha256_file(path) != digest:
        raise ValueError("GDN NPZ changed while reading")
    return values, {"kind": "NPZ_INPUTS", "path": str(path), "sha256": digest,
                    "scope": "one supplied GDN input sample; not a full-model quality evaluation"}


def _reference_chunk(*inputs):
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_chunk_gated_delta_rule
    return torch_chunk_gated_delta_rule(*inputs, chunk_size=64, initial_state=None,
                                      output_final_state=True, use_qk_l2norm_in_kernel=True)


def diagnose(values, *, source):
    arrays = validate_inputs(values)
    with np.errstate(over="ignore", invalid="ignore"):
        fixed = {name: np.array(value, dtype=np.float16, copy=True) for name, value in arrays.items()}
    input_report = {name: {"source_dtype": str(arrays[name].dtype), "shape": list(arrays[name].shape),
                          "source_sha256": hashlib.sha256(arrays[name].tobytes(order="C")).hexdigest(),
                          "fixed_fp16_sha256": hashlib.sha256(fixed[name].tobytes(order="C")).hexdigest(),
                          "rounding_error": metrics(arrays[name], fixed[name])} for name in NAMES}
    report = {"schema_version": 1, "status": "DIAGNOSTIC_COMPLETE", "source": source,
              "device_execution": "NOT_RUN", "reference_backend": "cpu", "accuracy_gate": "NOT_EVALUATED", "chunk_size": 64,
              "diagnostic_limits": {**LIMITS, "npz_bytes": MAX_NPZ_BYTES},
              "comparison": "current direct GE FP16 IR via host Torch oracle versus upstream FP32 chunk GDN",
              "input_policy": "round all inputs to FP16 once; feed identical values to FP16 IR and FP32 reference",
              "input_rounding": input_report, "chunks": [],
              "limitations": ["Synthetic data does not establish model accuracy or identify the user's precision failure.",
                              "FP16 versus FP32 differences include every GDN arithmetic stage, not only recurrent state accumulation.",
                              "FP32 reference is a numerical comparator, not a proof of exact mathematical results.",
                              "Host Torch evaluation does not execute GE scan lowering, OMG kernels or the NPU.",
                              "No tolerance, deployment dtype, weights, RMSNorm initialization or production output ABI is changed."]}
    if any(not np.isfinite(array).all() for array in (*arrays.values(), *fixed.values())):
        report["status"] = "DIAGNOSTIC_NONFINITE_INPUT"
        return report
    from transformers import __version__ as transformers_version
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_chunk_gated_delta_rule
    report["reference"] = {"transformers_version": transformers_version, "torch_version": str(torch.__version__),
                           "function": "torch_chunk_gated_delta_rule", "dtype": "float32",
                           "use_qk_l2norm_in_kernel": True, "initial_state": "zeros",
                           "source_sha256": hashlib.sha256(inspect.getsource(torch_chunk_gated_delta_rule).encode()).hexdigest()}
    report["sources"] = [{"path": str(path), "sha256": sha256_file(path)} for path in
                         (Path(__file__).resolve(), Path(inspect.getfile(_chunk_gdn)).resolve(),
                          Path(inspect.getfile(evaluate_torch)).resolve())]
    graph = Graph("gdn_precision_diagnostic")
    inputs = [graph.input(name, fixed[name].shape) for name in NAMES]
    states = []
    output = _chunk_gdn(_Builder(graph), *inputs, state_outputs=states)
    graph.outputs = [output, *states]
    actual_output, *actual_states = evaluate_torch(graph, fixed)
    reference_inputs = [torch.from_numpy(fixed[name]).float() for name in NAMES]
    sequence = fixed["q"].shape[1]
    with torch.inference_mode():
        for index, start in enumerate(range(0, sequence, 64)):
            end = min(start + 64, sequence)
            reference_output, reference_state = _reference_chunk(*(value[:, :end] for value in reference_inputs))
            row = {"chunk": index, "token_start": start, "token_end": end,
                   "output": metrics(reference_output[:, start:end].numpy(), actual_output[:, start:end].numpy()),
                   "state": metrics(reference_state.numpy(), actual_states[index].numpy())}
            report["chunks"].append(row)
    if any(row[kind]["status"] != "FINITE" for row in report["chunks"] for kind in ("output", "state")):
        report["status"] = "DIAGNOSTIC_NONFINITE_RESULT"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", help="NPZ with q, k, v, g, beta; omit for the small synthetic sample")
    parser.add_argument("--seed", type=int, default=17, help="seed for the default S128/H2/K8/V8 synthetic sample")
    parser.add_argument("--output", required=True, help="new JSON file below AI_RUN_DIR")
    args = parser.parse_args(argv)
    try:
        destination = output_path(args.output)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"diagnostic output already exists: {destination}")
        if destination.suffix.lower() != ".json":
            raise ValueError("--output must be a JSON filename")
        if args.inputs:
            values, source = load_inputs(args.inputs)
        else:
            values, source = synthetic_inputs(args.seed), {"kind": "SYNTHETIC", "seed": args.seed,
                "shape": {"S": 128, "H": 2, "K": 8, "V": 8}, "scope": "synthetic diagnostic, not model quality"}
        torch.set_num_threads(1)
        report = diagnose(values, source=source)
        save_json(destination, report)
        print(f"GDN precision: {report['status']}; {destination}")
        return 0 if report["status"] == "DIAGNOSTIC_COMPLETE" else 1
    except (OSError, ValueError, RuntimeError, TypeError, zipfile.BadZipFile) as error:
        print(f"diagnose-gdn-precision: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
