"""Original target tool execution against an in-memory board, not real HIAI."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex

import pytest

from qwen35_moe_dflash import original_target_board as board
from qwen35_moe_dflash.existing_air_profile import get_existing_air_profile
from qwen35_moe_dflash.hashing import sha256_file


def contract():
    profile = get_existing_air_profile("cpro-40block-64-1")
    inputs = []
    for index, entry in enumerate(profile["input_shape"].split(";")):
        name, shape = entry.split(":")
        inputs.append({"name": name, "type": "RefData" if name.startswith("RefData_") else "Data",
            "shape": [int(d) for d in shape.split(",")], "schema_dtype_names": ["DT_FLOAT16"],
            "serialized_data_index": index})
    outputs = [{"name": "logits", "netoutput": "output", "output_index": 0,
                "shape": [1, 1, 248320], "schema_dtype_names": ["DT_FLOAT"]}]
    by_name = {r["name"]: r for r in inputs}
    outputs.extend({"name": f"RefData_{i}", "netoutput": "output", "output_index": i + 1,
                    "shape": by_name[f"RefData_{i}"]["shape"], "schema_dtype_names": ["DT_FLOAT16"]} for i in range(80))
    return {"artifact_kind": "qwen35-original-target-contract", "status": "OBSERVED",
            "boundaries": {"declared_profile_check": {"status": "MATCHED_VISIBLE_BOUNDARIES"},
                           "inputs_in_serialized_op_order": inputs, "outputs": outputs}}, profile


def test_plan_uses_compiled_names_with_user_input_semantics_and_independent_state_slots():
    observed, profile = contract()
    plan = board.tool_plan(observed, profile, 64, "/board/original")
    assert len(plan["inputs"]) == 89 and len(plan["outputs"]) == 81
    assert plan["inputs"][1]["name"] == "arg113_1" and plan["inputs"][1]["semantic"] == "attention_mask"
    assert plan["inputs"][2]["name"] == "arg107_1" and plan["inputs"][2]["semantic"] == "position_ids"
    assert "arg112_1:" not in plan["input_shape"] and "arg106_1:" not in plan["input_shape"]
    assert plan["inputs"][84]["name"] == "arg1238_1" and plan["inputs"][84]["semantic"] == "output_pos"
    assert [Path(plan["inputs"][i]["source"]).name for i in (3, 5, 7)] == ["past_conv_state.bin"] * 3
    assert len({r["file"] for r in plan["inputs"]}) == 89
    assert plan["inputs"][9]["source"].endswith("past_k.bin")
    assert plan["target_features"] == "NOT_EXPOSED_BY_USER_81_OUTPUT_CONTRACT"


def test_one_gear_resolves_only_dynamic_inputs_and_keeps_full_state_sizes():
    observed, profile = contract()
    prefill = board.tool_plan(observed, profile, 64, "/board/original")
    decode = board.tool_plan(observed, profile, 1, "/board/original")
    assert decode["inputs"][0]["shape"] == [1, 1, 2048]
    assert decode["inputs"][1]["shape"] == [1, 1, 1, 2048]
    assert decode["inputs"][9]["shape"] == prefill["inputs"][9]["shape"] == [64, 32, 32, 16]
    assert decode["state_continuity"] == "NOT_VALIDATED_SINGLE_TOOL_INVOCATION"


def test_separate_refdata_indices_are_not_assumed_to_be_hiai_runtime_slots():
    observed, profile = contract()
    for row in observed["boundaries"]["inputs_in_serialized_op_order"]:
        if row["name"].startswith("RefData_"):
            row["serialized_data_index"] = int(row["name"].split("_")[1])
    plan = board.tool_plan(observed, profile, 64, "/board/inputs")
    assert plan["serialized_index_check"] == "SEPARATE_OR_INCOMPLETE_INDICES_NOT_RUNTIME_SLOTS"
    assert plan["runtime_input_order"] == "USER_DECLARED_UNVERIFIED_BY_DEVICE_REFERENCE"
    assert plan["inputs"][3]["semantic"] == "RefData_0" and plan["inputs"][3]["serialized_data_index"] == 0


@pytest.mark.parametrize("change", ["wrong_index", "unknown_dtype", "output_count", "output_state_dtype", "profile_difference"])
def test_wrong_abi_or_unresolved_bytes_prevent_board_plan(change):
    observed, profile = contract()
    boundary = observed["boundaries"]
    if change == "wrong_index":
        boundary["inputs_in_serialized_op_order"][1]["serialized_data_index"] = 2
        boundary["inputs_in_serialized_op_order"][2]["serialized_data_index"] = 1
    elif change == "unknown_dtype":
        boundary["inputs_in_serialized_op_order"][0]["schema_dtype_names"] = ["UNKNOWN"]
    elif change == "output_count":
        boundary["outputs"].pop()
    elif change == "output_state_dtype":
        boundary["outputs"][1]["schema_dtype_names"] = ["DT_FLOAT"]
    else:
        boundary["declared_profile_check"]["status"] = "DIFFERENT_VISIBLE_BOUNDARIES"
    with pytest.raises(ValueError):
        board.tool_plan(observed, profile, 64, "/board/inputs")


@pytest.fixture
def simulated(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("SN", "FAKE-DEVICE")
    monkeypatch.setenv("BIP", "fake-server:8790")
    root = tmp_path / "compilation"
    files = root / "compile-work/generated/model"
    files.mkdir(parents=True)
    records = []
    for name, data in (("model.omc", b"OPAQUE MOCK OMC"), ("SubGraph_0.weight", b"MOCK WEIGHT")):
        path = files / name
        path.write_bytes(data)
        records.append({"path": str(path.relative_to(root)), "bytes": len(data), "sha256": sha256_file(path)})
    report = {"artifact_kind": "qwen35-existing-air-compilation", "platform": "kirinx90", "status": "COMPILED",
              "model_compilation": "PASSED", "requested_options": {"save_weights_as_external_data": "true"},
              "omc": records[0], "generated_artifacts": records}
    (root / "report.json").write_text(json.dumps(report))
    observed, profile = contract()
    observed.update(compilation_report_sha256=sha256_file(root / "report.json"), omc=records[0])
    source = tmp_path / "original-inputs"
    source.mkdir()
    plan = board.tool_plan(observed, profile, 64, "/board/inputs")
    for row in plan["inputs"]:
        path = source / PurePosixPath(row["source"]).name
        if not path.exists():
            path.write_bytes(bytes(row["bytes"]))
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(observed))
    state = {"remote": {}, "executions": 0, "corrupt": False, "status": 0, "commands": []}

    class FakeHdc:
        def __init__(self, *args):
            self.count = 0
        def send(self, local, remote):
            self.count += 1
            state["remote"][str(remote)] = Path(local).read_bytes()
        def receive(self, remote, local):
            self.count += 1
            raw = state["remote"][str(remote)]
            if state["corrupt"] and str(remote).endswith("vendor-output.bin"):
                raw += b"CORRUPTED IN TRANSIT"
            Path(local).write_bytes(raw)
        def shell(self, command, label, **kwargs):
            self.count += 1
            state["commands"].append(command)
            remote = state["remote"]
            def digest(path):
                return hashlib.sha256(remote[path]).hexdigest() + "  " + path
            if label == "device-preflight":
                return 0, "sha256sum\nfind\n" + "1" * 64 + "  /board/tool\nFAKE-FIRMWARE"
            if label == "inspect-model-cache":
                manifest = next(v.decode() for k, v in remote.items() if k.endswith("model-files.tsv"))
                return 0, "\n".join("SEND " + line.split(" ", 1)[1] for line in manifest.splitlines())
            if label == "verify-upload" or label.startswith("hash-tool-"):
                return 0, digest(shlex.split(command)[1])
            if label == "prepare-inputs":
                script = remote[shlex.split(command)[1]].decode()
                output = []
                for line in script.splitlines():
                    if line.startswith("cp "):
                        _, src, dst = shlex.split(line)
                        remote[dst] = remote[src]
                    if line.startswith("sha256sum "):
                        output.append(digest(shlex.split(line)[1]))
                return 0, "\r\n".join(output)
            if label == "execute-original-target":
                state["executions"] += 1
                attempt = str(PurePosixPath(next(k for k in remote if k.endswith("prepare-inputs.sh"))).parent)
                state["attempt"] = attempt
                remote[attempt + "/tool.stdout"] = b"mock Load/Run finished; no numerical evidence"
                remote[attempt + "/tool.stderr"] = b""
                remote[attempt + "/tool.exitcode"] = str(state["status"]).encode() + b"\n"
                remote[attempt + "/outputs/vendor-output.bin"] = b"MOCK OUTPUT"
                return state["status"], ""
            if label == "output-inventory":
                return 0, digest(state["attempt"] + "/outputs/vendor-output.bin")
            return 0, ""
    monkeypatch.setattr(board, "Hdc", FakeHdc)
    return dict(root=root, contract=contract_path, profile=profile, inputs=source, state=state, out=tmp_path / "board")


def run(simulated):
    case = simulated
    return board.execute_target_tool(case["root"], case["contract"], case["out"], profile=case["profile"],
                                      host_input_dir=case["inputs"], tool="/board/tool", board_model_root="/board/models")


def test_tool_executes_once_with_original_host_inputs_and_retains_unvalidated_status(simulated):
    result = run(simulated)
    assert result["status"] == "EXECUTED_UNVALIDATED" and simulated["state"]["executions"] == 1
    assert len(result["inputs"]) == 89
    assert len({row["file"] for row in result["inputs"]}) == 89
    assert result["numerical_validation"] == "NOT_RUN_NO_REFERENCE"
    assert result["state_continuity"] == "NOT_VALIDATED_SINGLE_TOOL_INVOCATION"
    assert result["retrieved"]["outputs/vendor-output.bin"]["sha256"] == hashlib.sha256(b"MOCK OUTPUT").hexdigest()
    cmd = json.loads((simulated["out"] / "command.json").read_text())["argv"]
    assert "--modelmanager_itf=V2" in cmd and "--times=1" in cmd
    assert next(a for a in cmd if a.startswith("--weight_dir=")).endswith("/payload/model")
    assert all("mount -o remount" not in c and "rm -rf" not in c for c in simulated["state"]["commands"])


@pytest.mark.parametrize("failure", ["native", "retrieval", "input_size"])
def test_device_or_input_failure_is_not_retried_or_relabelled_as_accuracy(simulated, failure):
    if failure == "native":
        simulated["state"]["status"] = 7
    elif failure == "retrieval":
        simulated["state"]["corrupt"] = True
    else:
        (simulated["inputs"] / "inputs_embeds.bin").write_bytes(b"BAD SIZE")
    result = run(simulated)
    assert result["status"] == "FAILED" and result["numerical_validation"] == "NOT_RUN_NO_REFERENCE"
    assert simulated["state"]["executions"] == (0 if failure == "input_size" else 1)
    if failure != "input_size":
        assert result["device_execution"] == "FAILED"
        assert (simulated["out"] / "tool.exitcode").is_file()
