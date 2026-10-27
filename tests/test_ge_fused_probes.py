"""Bounded contracts for the five documented single-op GE cases."""
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from qwen35_moe_dflash.ge_fused.bindings import document_bindings
from qwen35_moe_dflash.ge_fused.probes import build_fused_probes, compare_fused_probes, inspect_fused_probes
from qwen35_moe_dflash.ge_fused.recipe import read_recipe
from qwen35_moe_dflash.hashing import sha256_file


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    binding = tmp_path / "binding.json"
    binding.write_text(json.dumps(document_bindings(target="kirinx90", ddk_id="test-ddk", gdn_initial_state_dtype="float32")))
    ge = tmp_path / "ge"
    (ge / "include/graph").mkdir(parents=True)
    (ge / "lib64").mkdir()
    for name in ("graph.h", "operator.h"):
        (ge / "include/graph" / name).write_text("MOCK HEADER")
    (ge / "lib64/libgraph.so").write_bytes(b"MOCK LIBGRAPH")
    ddk = tmp_path / "ddk"
    binary = ddk / "tools/tools_omg/master/omg"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"MOCK OMG")
    binary.chmod(0o700)
    return binding, ge, ddk


def _mock_tools(monkeypatch, fail=None):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        folder = Path(kwargs["cwd"])
        if "--target=omc" in command:
            (folder / "dump_graph").mkdir()
            (folder / "kernel_meta").mkdir()
            (folder / "kernel_meta/worker.log").write_text("MOCK child compilation traceback\n")
            if folder.parent.name == fail:
                (folder / "dump_graph/02_after_infershape_1_fail.irpb").write_bytes(b"MOCK FAILED IR")
                raise subprocess.CalledProcessError(1, command)
            (folder / "dump_graph/after_infershape.irpb").write_bytes(b"MOCK SUCCESS IR")
            (folder / "probe.omc").write_bytes(b"MOCK OMC " + folder.parent.name.encode())
        else:
            path = command[command.index("-o") + 1] if "-o" in command else command[1]
            Path(path).write_bytes(b"MOCK HOST OUTPUT")
        kwargs["stdout"].write("Mock tools; no GE/OMG/device execution.\n")
    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_five_source_graphs_are_exact_single_custom_ops(setup, tmp_path, monkeypatch):
    binding, _, _ = setup
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("source construction must not run a tool"))
    result = build_fused_probes(tmp_path / "source", binding)
    assert result["status"] == "SOURCE_READY" and len(result["cases"]) == 5
    root = tmp_path / "source"
    for case in result["cases"]:
        assert case["construction"] == "PASSED" and case["export"] == case["compile"] == "NOT_RUN"
        graph = json.loads((root / "cases" / case["name"] / "graph.json").read_text())
        assert len(graph["nodes"]) == len(graph["custom_ops"]) == 1
        assert graph["weights"] == {} and graph["constants"] == {}
        recipe = read_recipe(root / "cases" / case["name"] / "recipe/graph.recipe")
        assert [record["type"] for record in recipe["records"]] == ["Data"] * len(case["inputs"]) + [case["operator_type"]]
        abi = json.loads((root / case["operator_abi"]["path"]).read_text())
        assert [item["name"] for item in case["inputs"]] == [item["name"] for item in abi["inputs"]]
        assert len(case["outputs"]) == 2 and case["bundle_manifest"] is None
        if case["role"] == "gdn":
            assert case["inputs"][0]["shape"] == [1, case["sequence_length"], 32, 128]
            assert case["inputs"][5]["dtype"] == "float32" and case["inputs"][6]["dtype"] == "int16"
            assert all(value["type"] == "int" for value in abi["attrs"].values())
            assert case["input_policy"]["ranges"]["initial_state"]["nonzero"] == 32 * 128 * 128
        elif case["role"] == "conv_state":
            assert case["outputs"][0]["reference"] is None
            assert case["outputs"][0]["shape"] == [1, 8192, 4]
        else:
            assert case["inputs"][0]["name"] == "self" and case["inputs"][0]["shape"] == [1, 8, 2048]
            assert case["input_policy"]["ranges"]["gamma"]["min"] == .75
    assert not list(root.rglob("*.air")) and not list(root.rglob("*.omc"))


def test_all_cases_add_cache_and_dynamic_attention_without_production_bindings(setup, tmp_path):
    from qwen35_moe_dflash.ge_fused.ir import ATTENTION_INPUT_SCHEMA, ATTENTION_CONNECTED_PORTS
    binding, _, _ = setup
    original = binding.read_bytes()
    root = tmp_path / "all-operators"
    manifest = build_fused_probes(root, binding, cases="all")
    assert manifest["status"] == "SOURCE_READY" and len(manifest["cases"]) == 9
    assert binding.read_bytes() == original
    assert set(manifest["binding_contract"]["operators"]) == {"gdn", "conv_state", "rms"}
    for case in manifest["cases"][5:]:
        assert case["construction"] == "PASSED"
        folder = root / "cases" / case["name"]
        abi = json.loads((folder / "operator-abi.json").read_text())
        recipe = read_recipe(folder / "recipe/graph.recipe")
        assert len(recipe["records"]) == len(case["inputs"]) + 1
        assert recipe["records"][-1]["type"] == case["operator_type"]
        assert all(output["reference"] is None for output in case["outputs"])
        assert abi["probe_contract"]["source"]["path"].endswith("operators-and-quantization-analysis.md")
        sequence = case["sequence_length"]
        if case["role"] == "cache":
            assert [port["name"] for port in case["inputs"]] == ["x", "updates", "targetBlock", "offsetInBlock"]
            assert case["inputs"][-1]["shape"] == [] and case["inputs"][-1]["bytes"] == 4
            assert case["inputs"][1]["shape"] == [sequence, 32, 16]
            assert case["outputs"][0]["shape"] == [64, 32, 32, 16]
        else:
            assert tuple(port["name"] for port in case["inputs"]) == ATTENTION_CONNECTED_PORTS
            assert [entry["name"] for entry in abi["probe_contract"]["ge_input_schema"]] == [p for p, _ in ATTENTION_INPUT_SCHEMA]
            assert [entry["index"] for entry in abi["probe_contract"]["ge_input_schema"]] == list(range(18))
            assert case["inputs"][0]["shape"] == case["outputs"][0]["shape"] == [1, 256, sequence, 16]
            assert case["inputs"][4]["dtype"] == case["inputs"][5]["dtype"] == "int64"
            cpp = (folder / "probe.cpp").read_text()
            assert 'DynamicInputRegister("key", 1);' in cpp and 'DynamicInputRegister("value", 1);' in cpp
            assert 'OptionalInputRegister("pse_shift");' in cpp
            assert 'OptionalInputRegister("actual_seq_lengths_q_back");' in cpp
            assert cpp.index('DynamicInputRegister("value", 1)') < cpp.index('OptionalInputRegister("pse_shift")') < cpp.index('OptionalInputRegister("atten_mask")')
            assert 'InputRegister("key0")' not in cpp and 'InputRegister("value0")' not in cpp


def test_new_operator_bundle_compare_and_scalar_wire(setup, tmp_path, monkeypatch):
    binding, ge, ddk = setup
    _mock_tools(monkeypatch)
    root = tmp_path / "new-compiled"
    manifest = build_fused_probes(root, binding, cases="cache_decode,attention_decode",
                                  ge_root=ge, ddk_root=ddk, ddk_id="test-ddk")
    assert manifest["status"] == "COMPILED"
    wire = (root / "bundle/cache_decode/step.tsv").read_text()
    assert "input\t3\toffsetInBlock\tint32\t\t" in wire
    board = tmp_path / "new-board"
    _fake_results(root, board, manifest)
    result = compare_fused_probes(root, board, tmp_path / "new-comparison", allow_host_fake=True)
    assert all(case["process"] == "PASSED" for case in result["cases"])
    assert all(case["numerical"] == "DIAGNOSTIC_COMPLETE" for case in result["cases"])


def test_attention_recipe_cpp_parser_reads_dynamic_elements_and_absent_slots(setup, tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    binding, _, _ = setup
    root = tmp_path / "attention-parser"
    result = build_fused_probes(root, binding, cases="attention_decode")
    assert result["status"] == "SOURCE_READY"
    folder = root / "cases/attention_decode"
    executable = tmp_path / "attention-recipe-reader"
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-DQ35_RECIPE_PARSE_ONLY",
                    str(folder / "probe.cpp"), "-o", str(executable)], check=True, capture_output=True, text=True)
    parsed = subprocess.run([str(executable), str(folder / "recipe/graph.recipe")],
                            check=True, capture_output=True, text=True, timeout=15)
    assert parsed.stdout.startswith("RECIPE_VALID ")
    assert "AIR_EXPORTED" not in parsed.stdout  # host parser check, no GE/compiler/kernel claim


def test_attention_compute_graph_verifier_preserves_optional_slots(tmp_path):
    """Execute the generated Verify body against an explicit host graph fixture.

    This tests our verifier, not GE registration, serialization or DDK support.
    The fixture has 18 registered inputs and eight edges at noncompact indices.
    """
    from qwen35_moe_dflash.ge_fused.recipe import _CPP
    from qwen35_moe_dflash.ge_recipe import _CPP as legacy_cpp
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    structs = _CPP[_CPP.index("struct Desc {"):_CPP.index("class Reader {")]
    fixture = r'''
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
void Require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
@STRUCTS@
namespace ge {
constexpr int GRAPH_SUCCESS = 0;
struct AscendString { std::string value; AscendString() = default;
  explicit AscendString(const char* s) : value(s) {} const char* GetString() const { return value.c_str(); } };
using TensorDesc = Desc;
struct Tensor { TensorDesc GetTensorDesc() const { return {}; } size_t GetSize() const { return 0; }
  const uint8_t* GetData() const { return nullptr; } };
struct GNode {
  std::string name, type; std::vector<std::string> inputs, outputs;
  int reported_input_count = -1;
  std::vector<Desc> in_desc, out_desc;
  std::vector<std::pair<std::shared_ptr<GNode>, int32_t>> edges;
  int GetName(AscendString& v) const { v.value = name; return 0; }
  int GetType(AscendString& v) const { v.value = type; return 0; }
  size_t GetInputsSize() const { return reported_input_count >= 0 ? static_cast<size_t>(reported_input_count) : inputs.size(); }
  size_t GetOutputsSize() const { return outputs.size(); }
  static int Index(const std::vector<std::string>& ports, const AscendString& n, int32_t& i) {
    auto it = std::find(ports.begin(), ports.end(), n.value); if (it == ports.end()) return 1;
    i = static_cast<int32_t>(std::distance(ports.begin(), it)); return 0;
  }
  int GetInputIndexByName(const AscendString& n, int32_t& i) const { return Index(inputs, n, i); }
  int GetOutputIndexByName(const AscendString& n, int32_t& i) const { return Index(outputs, n, i); }
  int GetInputDesc(int32_t i, TensorDesc& d) const { d = in_desc.at(i); return 0; }
  int GetOutputDesc(int32_t i, TensorDesc& d) const { d = out_desc.at(i); return 0; }
  auto GetInDataNodesAndPortIndexs(int32_t i) const { return edges.at(i); }
  int GetAttr(const AscendString&, Tensor&) const { return 1; }
};
struct Graph { std::vector<GNode> nodes; bool IsValid() const { return true; }
  const auto& GetAllNodes() const { return nodes; } };
}
void Check(int status, const char* why) { Require(status == 0, why); }
void VerifyDesc(const ge::TensorDesc& got, const Desc& want, const std::string& name) {
  Require(got.dtype == want.dtype && got.shape == want.shape, "descriptor mismatch " + name);
}
void VerifyAttribute(const ge::GNode&, const Attr&) { throw std::runtime_error("fixture has no attributes"); }
uint64_t Hash(const uint8_t*, size_t) { throw std::runtime_error("fixture has no constants"); }
@VERIFIER@
int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "valid";
  try {
    const std::vector<std::string> names = {"query", "key0", "value0", "pse_shift", "atten_mask",
      "actual_seq_lengths_q", "actual_seq_lengths_kv", "dequant_scale1", "quant_scale1", "dequant_scale2",
      "quant_scale2", "quant_offset2", "antiquant_scale", "antiquant_offset", "block_table",
      "kv_padding_size", "all_seq_lengths_q", "actual_seq_lengths_q_back"};
    const std::vector<int> connected = {0, 1, 2, 4, 5, 6, 14, 16};
    Recipe recipe; ge::Graph graph; Record attention; attention.name = "attention";
    attention.type = "AdnFusedInferAttention";
    ge::GNode node; node.name = attention.name; node.type = attention.type;
    node.inputs = names; node.in_desc.resize(18); node.edges.resize(18);
    node.reported_input_count = 8; // Observed GE counts connected inputs, with noncompact name indices.
    if (mode == "registered_count") node.reported_input_count = -1;
    const Desc output{1, {1, 256, 8, 16}, 65536};
    attention.outputs.push_back({"attention_out", output, 0, ""});
    node.outputs = {"attention_out"}; node.out_desc = {output};
    for (size_t i = 0; i < connected.size(); ++i) {
      const int slot = connected[i]; const Desc d{1, {static_cast<int64_t>(i + 1)}, 2 * (i + 1)};
      Record r; r.name = "source_" + std::to_string(i); r.type = "FixtureInput";
      r.outputs.push_back({"y", d, 0, ""}); recipe.records.push_back(r);
      ge::GNode source; source.name = r.name; source.type = r.type; source.outputs = {"y"}; source.out_desc = {d};
      graph.nodes.push_back(source); node.in_desc[slot] = d; node.edges[slot] = {std::make_shared<ge::GNode>(source), 0};
      attention.inputs.push_back({names[slot], d, static_cast<uint32_t>(i), "y"});
    }
    if (mode == "compact" || mode == "generic_valid" || mode == "generic_swapped") {
      ge::GNode compact = node; compact.inputs.clear(); compact.in_desc.clear(); compact.edges.clear();
      for (int slot : connected) { compact.inputs.push_back(node.inputs[slot]);
        compact.in_desc.push_back(node.in_desc[slot]); compact.edges.push_back(node.edges[slot]); }
      node = compact;
    }
    if (mode.find("generic_") == 0) { node.type = attention.type = "OrdinaryOp"; node.reported_input_count = -1; }
    if (mode == "swapped" || mode == "generic_swapped") std::swap(node.inputs[0], node.inputs[1]);
    if (mode == "wrong_edge") node.edges[4] = node.edges[0];
    if (mode == "extra_optional") node.edges[3] = node.edges[0];
    if (mode == "wrong_descriptor") node.in_desc[14].shape = {999};
    if (mode == "disconnected") node.edges[5].first.reset();
    recipe.records.push_back(attention); graph.nodes.push_back(node);
    Verify(graph, recipe); std::cout << "HOST_VERIFIER_PASSED\n"; return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
'''.replace("@STRUCTS@", structs)
    for label, source in (("current", _CPP), ("before", legacy_cpp)):
        begin = source.index("const std::vector<std::string>& AttentionInputNames()") if label == "current" else source.index("void Verify(ge::Graph& graph, const Recipe& recipe)")
        verifier = source[begin:source.index("void Absent(", begin)]
        cpp, binary = tmp_path / (label + ".cpp"), tmp_path / label
        cpp.write_text(fixture.replace("@VERIFIER@", verifier))
        subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(binary)],
                       check=True, capture_output=True, text=True, timeout=30)
        result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
        if label == "before":
            assert result.returncode == 1 and "input port order mismatch attention" in result.stderr
            continue
        assert result.returncode == 0, result.stderr
        for mode, message in {
            "compact": "schema port order mismatch", "swapped": "schema port order mismatch",
            "wrong_edge": "input edge source mismatch", "extra_optional": "unexpected optional attention edge",
            "wrong_descriptor": "descriptor mismatch", "disconnected": "missing input edge",
            "generic_swapped": "input port order mismatch", "generic_expanded": "port count mismatch",
        }.items():
            bad = subprocess.run([str(binary), mode], capture_output=True, text=True, timeout=10)
            assert bad.returncode == 1 and message in bad.stderr, (mode, bad.stdout, bad.stderr)
        generic = subprocess.run([str(binary), "generic_valid"], capture_output=True, text=True, timeout=10)
        assert generic.returncode == 0, generic.stderr
        expanded = subprocess.run([str(binary), "registered_count"], capture_output=True, text=True, timeout=10)
        assert expanded.returncode == 0, expanded.stderr


@pytest.mark.parametrize("initial_dtype,final_dtype", [("float32", "float32"), ("float16", "float32"), ("float16", "float16")])
def test_selected_gdn_cases_compile_bundle_and_compare_in_canonical_order(setup, tmp_path, monkeypatch, capsys,
                                                                        initial_dtype, final_dtype):
    binding, ge, ddk = setup
    contract = json.loads(binding.read_text())
    contract["operators"]["gdn"]["inputs"]["initial_state"]["dtype"] = initial_dtype
    contract["operators"]["gdn"]["outputs"]["final_state"]["dtype"] = final_dtype
    binding.write_text(json.dumps(contract))
    calls = _mock_tools(monkeypatch)
    root = tmp_path / "compiled"
    manifest = build_fused_probes(root, binding, ge_root=ge, ddk_root=ddk, ddk_id="test-ddk",
                                  cases=["gdn_decode", "gdn_prefill"])
    assert manifest["selected_cases"] == ["gdn_prefill", "gdn_decode"]
    assert len(calls) == 6 and len(manifest["cases"]) == 2
    assert sorted(path.name for path in (root / "cases").iterdir()) == ["gdn_decode", "gdn_prefill"]
    script = (root / "bundle/run-board.sh").read_text()
    assert "gdn_prefill/step.tsv" in script and "gdn_decode/step.tsv" in script
    assert "conv_prefill" not in script and "rms/step.tsv" not in script
    progress = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["completed"] for row in progress] == [1, 2]
    assert all(row["compile"] == "PASSED" and row["total"] == 2 for row in progress)
    board = tmp_path / "board"
    _fake_results(root, board, manifest)
    comparison = compare_fused_probes(root, board, tmp_path / "compared", allow_host_fake=True)
    assert len(comparison["cases"]) == 2 and all(row["process"] == "PASSED" for row in comparison["cases"])
    for case, compared in zip(manifest["cases"], comparison["cases"]):
        assert case["inputs"][5]["dtype"] == initial_dtype and case["outputs"][1]["dtype"] == final_dtype
        assert compared["diagnostics"][1]["dtype"] == final_dtype
        assert compared["diagnostics"][1]["max_abs_error"] == 0
    inspection = inspect_fused_probes(root, ddk, tmp_path / "inspected")
    assert [row["name"] for row in inspection["cases"]] == manifest["selected_cases"]


@pytest.mark.parametrize("selected", [[], "", ["gdn_prefill", "gdn_prefill"], ["unknown"], [None]])
def test_probe_selection_rejects_empty_duplicate_unknown_before_writing(setup, tmp_path, selected):
    binding, _, _ = setup
    with pytest.raises(ValueError, match="nonempty unique subset"):
        build_fused_probes(tmp_path / "invalid", binding, cases=selected)
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown", "undeclared_subset", "wrong_contract"])
def test_manifest_case_selection_rejects_incomplete_or_forged_rows(setup, tmp_path, mutation):
    binding, _, ddk = setup
    root = tmp_path / "source"
    manifest = build_fused_probes(root, binding, cases="gdn_prefill,gdn_decode")
    if mutation == "missing":
        manifest["cases"].pop()
    elif mutation == "duplicate":
        manifest["selected_cases"] = ["gdn_prefill", "gdn_prefill"]
        manifest["cases"][1] = manifest["cases"][0]
    elif mutation == "unknown":
        manifest["selected_cases"] = ["unknown"]
        manifest["cases"] = [{"name": "unknown"}]
    elif mutation == "undeclared_subset":
        del manifest["selected_cases"]
    else:
        manifest["cases"][0]["operator_type"] = "DifferentOperator"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        compare_fused_probes(root, tmp_path / "board", tmp_path / "compared")
    with pytest.raises(ValueError):
        inspect_fused_probes(root, ddk, tmp_path / "inspected")
    assert not (tmp_path / "compared").exists() and not (tmp_path / "inspected").exists()


def test_legacy_complete_manifest_without_selection_remains_readable(setup, tmp_path):
    binding, _, ddk = setup
    root = tmp_path / "source"
    manifest = build_fused_probes(root, binding)
    del manifest["selected_cases"]
    (root / "manifest.json").write_text(json.dumps(manifest))
    assert len(inspect_fused_probes(root, ddk, tmp_path / "inspected")["cases"]) == 5
    assert len(compare_fused_probes(root, tmp_path / "board", tmp_path / "compared")["cases"]) == 5


def test_tbe_diagnosis_requires_compilation_and_distinguishes_audit_from_outcome(setup, tmp_path, monkeypatch):
    from qwen35_moe_dflash import compile as compiler
    binding, ge, ddk = setup
    diagnosis = tmp_path / "diagnosis"
    with pytest.raises(ValueError, match="actual DDK compilation"):
        build_fused_probes(tmp_path / "invalid", binding, tbe_diagnosis=diagnosis)
    assert not (tmp_path / "invalid").exists()
    calls = _mock_tools(monkeypatch, fail="gdn_prefill")
    tool_run = subprocess.run
    prepared = []
    def prepare(ddk_root, diagnosis_run, output_dir, *, inherited):
        assert ddk_root == ddk and diagnosis_run == diagnosis
        output_dir.mkdir()
        audit = output_dir / "worker-audit.jsonl"
        prepared.append(output_dir.parent.name)
        return dict(inherited, MOCK_TBE_AUDIT=str(audit)), {"audit_log": str(audit)}
    def verify(metadata):
        audit = Path(metadata["audit_log"])
        return {"status": "APPLIED" if audit.is_file() else "NOT_OBSERVED", "audit_log": str(audit),
                "events": [{"event": "APPLIED", "pid": 123,
                            "module_origins": {"tbe.module": "/mock/long-module-path"}}] if audit.is_file() else []}
    def run(command, **kwargs):
        if "--target=omc" in command:
            assert "--platform=kirinx90" in command
            audit = Path(kwargs["env"]["MOCK_TBE_AUDIT"])
            if Path(kwargs["cwd"]).parent.name == "gdn_prefill":
                audit.write_text('{"status":"ADAPTED"}\n')
        return tool_run(command, **kwargs)
    monkeypatch.setattr(compiler, "prepare_tbe_compatibility", prepare, raising=False)
    monkeypatch.setattr(compiler, "verify_tbe_compatibility", verify, raising=False)
    monkeypatch.setattr(subprocess, "run", run)
    root = tmp_path / "compiled"
    result = build_fused_probes(root, binding, ge_root=ge, ddk_root=ddk, ddk_id="test-ddk",
                                cases="gdn_prefill,gdn_decode", tbe_diagnosis=diagnosis)
    assert prepared == ["gdn_prefill", "gdn_decode"] and len(calls) == 6
    failed, passed = result["cases"]
    assert failed["compile"] == "FAILED" and failed["tbe_compatibility"]["audit"]["status"] == "APPLIED"
    assert failed["tbe_compatibility"]["compile"] == "FAILED"
    assert passed["compile"] == "PASSED" and passed["tbe_compatibility"]["audit"]["status"] == "NOT_OBSERVED"
    assert passed["tbe_compatibility"]["compiler_invocation"] == "ATTEMPTED"
    report = (root / "report.txt").read_text()
    assert "APPLIED" in report and "NOT_OBSERVED" in report
    assert '"applied_count": 1' in report and '"pids": [123]' in report and '"audit_log":' in report
    assert "module_origins" not in report and "long-module-path" not in report
    assert failed["tbe_compatibility"]["audit"]["events"][0]["module_origins"]
    from qwen35_moe_dflash.ge_fused.probes import _report
    rejected = dict(failed, tbe_compatibility=dict(failed["tbe_compatibility"], audit={
        "status": "REJECTED", "events": [{"event": "REJECTED", "pid": 456,
                                           "error": "module provenance mismatch", "module_origins": {"tbe": "wrong"}}]}))
    _report(tmp_path / "rejected.txt", [rejected])
    rejected_report = (tmp_path / "rejected.txt").read_text()
    assert '"rejected_count": 1' in rejected_report and "module provenance mismatch" in rejected_report
    assert "module_origins" not in rejected_report


def test_export_compile_boundaries_and_continue_after_failure(setup, tmp_path, monkeypatch):
    binding, ge, ddk = setup
    calls = _mock_tools(monkeypatch, fail="gdn_prefill")
    result = build_fused_probes(tmp_path / "compiled", binding, ge_root=ge, ddk_root=ddk, ddk_id="test-ddk")
    assert result["status"] == "PARTIAL" and len(calls) == 15
    assert result["cases"][0]["export"] == "PASSED" and result["cases"][0]["compile"] == "FAILED"
    assert all(case["status"] == "COMPILED" for case in result["cases"][1:])
    failed = tmp_path / "compiled/cases/gdn_prefill/compile-work"
    assert (failed / "dump_graph/02_after_infershape_1_fail.irpb").read_bytes() == b"MOCK FAILED IR"
    assert (failed / "kernel_meta/worker.log").read_text() == "MOCK child compilation traceback\n"
    assert (failed / "input.air").is_file() and not (failed / "input.air").is_symlink()
    assert "Retained compiler work:" in result["cases"][0]["error"]
    inventory = {item["path"]: item for item in result["cases"][0]["files"]}
    worker_record = inventory["cases/gdn_prefill/compile-work/kernel_meta/worker.log"]
    assert worker_record["sha256"] == sha256_file(failed / "kernel_meta/worker.log")
    assert not list((tmp_path / "compiled").rglob(".ipc-*"))
    script = (tmp_path / "compiled/bundle/run-board.sh").read_text()
    assert "gdn_prefill/step.tsv" not in script and "rms/step.tsv" in script
    for case in result["cases"][1:]:
        work = tmp_path / "compiled" / case["compile_work"]
        assert (work / "dump_graph/after_infershape.irpb").read_bytes() == b"MOCK SUCCESS IR"
        assert (work / "kernel_meta/worker.log").is_file()
        root = tmp_path / "compiled/bundle" / case["name"]
        assert not list(root.rglob("worker.log"))  # compiler diagnostics are not uploaded as model payloads
        lines = (root / "step.tsv").read_text().splitlines()
        assert lines[0] == "QWEN35_FUSED_OP_STEP\t1" and lines[3] == "operator\t" + case["operator_type"]
        assert lines[2].startswith("operator_abi\toperator-abi.json\t")
        assert not any("target_top1" in line for line in lines)


def _fake_results(probe, result_dir, manifest):
    result_dir.mkdir()
    for case in manifest["cases"]:
        step_root = probe / "bundle" / case["name"]
        step = json.loads((step_root / "manifest.json").read_text())
        output = result_dir / case["name"]
        output.mkdir()
        records = []
        for desc in case["outputs"]:
            dtype = "<f2" if desc["dtype"] == "float16" else "<f4"
            data = (np.frombuffer((probe / desc["reference"]["path"]).read_bytes(), dtype=dtype).reshape(desc["shape"]).copy()
                    if desc["reference"] else np.zeros(desc["shape"], dtype=dtype))
            if case["name"] == "gdn_prefill" and desc["logical"] == "core_attn":
                data[:, desc["valid_prefix_length"]:] = np.nan  # padding is explicitly outside comparison
            path = output / f"output_{desc['index']:04d}.bin"
            path.write_bytes(data.tobytes())
            records.append({**step["outputs"][desc["index"]], "path": path.name,
                            "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        result = {"schema_version": 1, "status": "EXECUTED", "scope": "fused-operator-probe",
                  "operator_type": step["operator_type"], "operator_calls": 1, "target_calls": 0, "draft_calls": 0,
                  "operator_abi_sha256": step["operator_abi"]["sha256"], "step_plan_sha256": step["step_plan"]["sha256"],
                  "model_sha256": step["model"]["sha256"], "ddk_id": "test-ddk", "runtime_version": "FAKE TEST ONLY",
                  "evidence_kind": "host-fake-hiai", "inputs": step["inputs"], "outputs": records}
        (output / "result.json").write_text(json.dumps(result))


def test_comparison_requires_bound_results_and_excludes_undefined_padding(setup, tmp_path, monkeypatch):
    binding, ge, ddk = setup
    _mock_tools(monkeypatch)
    root = tmp_path / "compiled"
    manifest = build_fused_probes(root, binding, ge_root=ge, ddk_root=ddk, ddk_id="test-ddk")
    board = tmp_path / "board"
    _fake_results(root, board, manifest)
    rejected = compare_fused_probes(root, board, tmp_path / "rejected")
    assert all(case["result_validation"] == "FAILED" for case in rejected["cases"])
    result = compare_fused_probes(root, board, tmp_path / "compared", allow_host_fake=True)
    assert all(case["process"] == "PASSED" and case["numerical"] == "DIAGNOSTIC_COMPLETE" for case in result["cases"])
    diagnostic = result["cases"][0]["diagnostics"][0]
    assert diagnostic["compared_shape"] == [1, 6, 32, 128] and diagnostic["max_abs_error"] == 0
    assert diagnostic["raw_nonfinite"] == 2 * 32 * 128 and diagnostic["actual_nonfinite"] == 0
    assert result["cases"][2]["diagnostics"][0]["reference"] == "SHAPE_ONLY_UNSPECIFIED_STATE_ORDER"
    path = board / "rms/result.json"
    bad = json.loads(path.read_text())
    bad["model_sha256"] = "0" * 64
    path.write_text(json.dumps(bad))
    changed = compare_fused_probes(root, board, tmp_path / "changed", allow_host_fake=True)
    assert changed["cases"][-1]["result_validation"] == "FAILED"
    assert all(case["process"] == "PASSED" for case in changed["cases"][:-1])


def test_compile_requires_matching_declared_platform_and_ddk(setup, tmp_path):
    binding, ge, ddk = setup
    with pytest.raises(ValueError, match="binding DDK identity"):
        build_fused_probes(tmp_path / "bad", binding, ge_root=ge, ddk_root=ddk, ddk_id="different")
    data = json.loads(binding.read_text())
    data["target"] = "Ascend310P3"
    binding.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="platform kirinx90"):
        build_fused_probes(tmp_path / "bad", binding, ge_root=ge, ddk_root=ddk, ddk_id="test-ddk")
    assert not (tmp_path / "bad").exists()


def test_no_compiled_cases_script_fails_without_invocation(setup, tmp_path):
    binding, _, _ = setup
    root = tmp_path / "source"
    build_fused_probes(root, binding)
    result = subprocess.run([str(root / "bundle/run-board.sh"), "/runner-that-does-not-exist",
                             str(tmp_path / "board"), "test-runtime", "test-ddk", "test-firmware"],
                            capture_output=True, text=True)
    assert result.returncode == 2 and "No COMPILED" in result.stderr
    assert "no operator invocation attempted" in (tmp_path / "board/run-error.txt").read_text()
    assert (root / "report.txt").is_file()
    assert not list((tmp_path / "board").glob("*.exitcode"))


def test_failure_inspection_reads_existing_middle_errors_and_never_runs_ddk(setup, tmp_path, monkeypatch):
    binding, _, ddk = setup
    root = tmp_path / "source"
    build_fused_probes(root, binding)
    log = root / "cases/gdn_prefill/compile.log"
    log.write_text("No such file or directory: optional.so\n" * 20 +
                   "Traceback (most recent call last):\n  wrapper.py:24\nModuleNotFoundError: missing_dependency\n" +
                   "compiler process died\n" * 20)
    source = ddk / "ddk/ascendc/ops/impl/custom/chunk_gated_delta_rule.py"
    source.parent.mkdir(parents=True)
    source.write_text("raise RuntimeError('DO NOT IMPORT THIS INSTALLED MODULE')\n"
                      "def chunk_gated_delta_rule(initial_state, output_final_state):\n    pass\n")
    before = {p: p.read_bytes() for folder in (root, ddk) for p in folder.rglob('*') if p.is_file()}
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("inspection must not execute any tool"))
    output = tmp_path / "inspection"
    result = inspect_fused_probes(root, ddk, output)
    assert result['new_compilation'] == result['device_execution'] == 'NOT_RUN'
    assert result['cases'][0]['compile_work_retained'] is False
    assert 'ModuleNotFoundError: missing_dependency' in (output / 'report.txt').read_text()
    assert len(result['installed_sources']) == 1
    assert (output / result['installed_sources'][0]['path']).read_bytes() == source.read_bytes()
    assert all(path.read_bytes() == payload for path, payload in before.items())
    assert result['cases'][1]['logs'][0]['status'] == 'UNREADABLE'
    for forbidden in (root / 'inspection', ddk / 'inspection'):
        with pytest.raises(ValueError, match='separate'):
            inspect_fused_probes(root, ddk, forbidden)
        assert not forbidden.exists()


def test_inspection_reads_export_errors_and_shared_registration_configs(setup, tmp_path, monkeypatch):
    binding, _, ddk = setup
    root = tmp_path / "source"
    build_fused_probes(root, binding, cases="conv_prefill,rms,attention_prefill")
    (root / "cases/conv_prefill/compile.log").write_text(
        "trace\n" * 500 + "Get format configs failed, node : conv_prefill_fused, substore: !\n" + "trace\n" * 500)
    (root / "cases/attention_prefill/export.log").write_text(
        "GE recipe failed: input port order mismatch attention_prefill_fused\n")
    config = ddk / "ddk/ascendc/ops/config/kirinx90/aic-supported-ops.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"LinearAttentionConvstateUpdate": {"format": "ND"},
                                 "RmsNorm": {"impl": "rms_norm"}}, indent=2))
    outside = tmp_path / "outside.json"
    outside.write_text('{"AdnRmsNorm": "OUTSIDE_MUST_NOT_READ"}')
    config.with_name('escape.json').symlink_to(outside)
    before = config.read_bytes()
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("inspection must not execute tools"))
    output = tmp_path / "inspection"
    result = inspect_fused_probes(root, ddk, output)
    report = (output / "report.txt").read_text()
    assert "Get format configs failed" in report and "input port order mismatch" in report
    registration = result["installed_registration"]
    assert registration["files_read"] == 1 and len(registration["records"]) == 1
    record = registration["records"][0]
    assert record["sha256"] == sha256_file(config) and (output / record["path"]).read_bytes() == before
    assert "LinearAttentionConvstateUpdate" in report and "RmsNorm" in report
    assert "OUTSIDE_MUST_NOT_READ" not in report and config.read_bytes() == before
    assert result["new_compilation"] == result["device_execution"] == "NOT_RUN"


def test_hash_bound_runtime_failures_preserve_stage_without_numerical_claim(setup, tmp_path, monkeypatch):
    binding, ge, ddk = setup
    _mock_tools(monkeypatch)
    root = tmp_path / "compiled"
    manifest = build_fused_probes(root, binding, ge_root=ge, ddk_root=ddk, ddk_id="test-ddk")
    board = tmp_path / "board"
    _fake_results(root, board, manifest)
    stages = ["LOAD_OR_SETUP", "INPUT_BINDING", "PROCESS", "PROCESS"]
    for case, stage in zip(manifest["cases"], stages):
        path = board / case["name"] / "result.json"
        result = json.loads(path.read_text())
        for key in ("runtime_version", "operator_calls", "inputs"):
            result.pop(key)
        result.update(status="RUNTIME_FAILED", failure_stage=stage, reason="injected runtime failure",
                      expected_runtime_version="FAKE TEST ONLY", outputs=[])
        if case["name"] == "conv_decode":
            result["model_sha256"] = "0" * 64
        path.write_text(json.dumps(result))
    rejected = compare_fused_probes(root, board, tmp_path / "rejected")
    assert all(case["result_validation"] == "FAILED" for case in rejected["cases"])
    output = tmp_path / "compared"
    result = compare_fused_probes(root, board, output, allow_host_fake=True)
    for case, stage, load, process in zip(result["cases"], stages,
                                        ("UNVERIFIED", "PASSED", "PASSED"),
                                        ("NOT_RUN", "NOT_RUN", "FAILED")):
        assert case["failure_stage"] == stage and case["runtime"] == "FAILED"
        assert case["load"] == load and case["process"] == process
        assert case["result_validation"] == "PASSED" and case["numerical"] == "NOT_EVALUATED"
        assert "diagnostics" not in case and case["result_sha256"]
    assert result["cases"][3]["result_validation"] == "FAILED"
    assert result["cases"][3]["load"] == result["cases"][3]["process"] == "UNVERIFIED"
    assert result["cases"][4]["process"] == "PASSED"
    report = (output / "report.txt").read_text()
    assert all("Native " + stage + " failed" in report for stage in stages[:3])
    assert (output / "report.json").is_file()


def test_inspection_preserves_traceback_and_reads_exact_platform_module_without_execution(setup, tmp_path, monkeypatch):
    binding, _, ddk = setup
    root = tmp_path / "source"
    build_fused_probes(root, binding)
    platform = tmp_path / "external/tbe/common/platform/platform_info.py"
    platform.parent.mkdir(parents=True)
    platform.write_text("import ctypes\nraise AssertionError('MUST NOT EXECUTE')\n"
                        "def te_update_version(soc):\n    return native_update(soc)\n"
                        "def te_set_version(soc):\n    return native_init(soc)\n"
                        "def unrelated():\n    return 'DO_NOT_DISPLAY'\n")
    case = root / "cases/gdn_prefill"
    log = ("Traceback (most recent call last):\n" + "  caller frame\n" * 30 +
           "  options = {'soc_version': 'kirinx90'}\n" +
           f'  File "{platform}", line 4, in te_update_version\n' +
           "RuntimeError: te_update_version() return error.\n")
    (case / "compile.log").write_text(log)
    (case / "omg-environment.json").write_text(json.dumps({
        "environment": {"SOC_VERSION": "kirinx90", "PYTHONPATH": "/selected/tbe", "SECRET_TOKEN": "do-not-copy"},
        "unrelated_secret": "do-not-copy"}))
    cpp = ddk / "ddk/ascendc/ops/impl/custom/chunk_gated_delta_rule.cpp"
    cpp.parent.mkdir(parents=True)
    cpp.write_text("".join(f"GlobalTensor<half> unrelated_{i};\n" for i in range(16)) +
                   "GlobalTensor<half> state;\nstate.SetGlobalBuffer(\n    (__gm__ half *)initial_state);\n")
    config = ddk / "ddk/ascendcpp/conf"
    config.mkdir(parents=True)
    (config / "soc.json").write_text('{}')
    before = platform.read_bytes()
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("read-only inspection must not run tools"))
    output = tmp_path / "inspection"
    result = inspect_fused_probes(root, ddk, output)
    report = (output / "report.txt").read_text()
    excerpts = result["cases"][0]["logs"][0]["excerpts"]
    assert [row["text"] for row in excerpts] == log.splitlines()
    assert "native_update(soc)" in report and "native_init(soc)" in report
    assert "DO_NOT_DISPLAY" not in report and "MUST NOT EXECUTE" not in report
    assert platform.read_bytes() == before and not list(platform.parent.rglob("*.pyc"))
    record = result["platform_sources"][0]
    assert record["sha256"] == sha256_file(platform) and (output / record["path"]).read_bytes() == before
    env = result["cases"][0]["compiler_environment"]
    assert env["environment"] == {"SOC_VERSION": "kirinx90", "PYTHONPATH": "/selected/tbe"}
    assert "do-not-copy" not in report and "do-not-copy" not in (output / env["path"]).read_text()
    assert "GlobalTensor<half> state" in report and "state.SetGlobalBuffer" in report
    assert next(row for row in result["ddk_directories"] if row["path"] == str(config))["entry_sample"] == ["soc.json"]
    # A path with the same basename outside the exact TBE suffix is never collected.
    other = tmp_path / "external/platform_info.py"
    other.write_text("raise RuntimeError('unrelated')\n")
    (case / "compile.log").write_text(f'Traceback (most recent call last):\n  File "{other}", line 1\n'
                                     + "  frame\n" * 100 + "RuntimeError: long traceback\n")
    limited = inspect_fused_probes(root, ddk, tmp_path / "bounded")
    assert limited["platform_sources"] == []
    indices = {row["line"] for row in limited["cases"][0]["logs"][0]["excerpts"]}
    assert 60 in indices and 61 not in indices  # bounded traceback plus separate terminal-error context
