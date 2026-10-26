"""Check runnable commands, resource failures and evidence identity gates."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from qwen35_moe_dflash.cli import execute, parser


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    root=tmp_path/"omc"; root.mkdir()
    graphs=[]
    for name in ["target", "draft"]:
        payload=name.encode(); (root/(name+".omc")).write_bytes(payload)
        graphs.append({"name":name,"omc":name+".omc","bytes":len(payload),"sha256":hashlib.sha256(payload).hexdigest()})
    manifest={"status":"COMPILED","artifact_kind":"qwen35-cpro-omc-pair","source":{"kind":"CLI-TEST-DOUBLE"},
              "sequence_length":16,"feature_width":32,"vocab_size":64,"draft_width":3,"graphs":graphs,
              "pad_token_id":0,"eos_token_ids":[10],"ddk_id":"fake","abi_version":2,
              "feature_dtype":"torch.float16","precision":"torch.float16","quantization":None}
    (root/"manifest.json").write_text(json.dumps(manifest))
    (tmp_path/"prompt.json").write_text(json.dumps({"prompt_token_ids":[1,2]}))
    return root, manifest


def test_board_command_uses_locked_paths_and_generate_default(bundle,tmp_path):
    root,m=bundle
    output=tmp_path/"cmd.json"
    execute(parser().parse_args(["board-command","--omc",str(root),"--prompt",str(tmp_path/"prompt.json"),
        "--board-model-dir","/data/test space","--board-output","/data/report.json",
        "--runtime-version","fake-version","--firmware-id","fake-fw","--max-new-tokens","8","--output",str(output)]))
    data=json.loads(output.read_text()); argv=data["argv"]
    assert argv[argv.index("--mode")+1]=="generate"
    assert argv[argv.index("--target")+1]=="/data/test space/target.omc"
    assert argv[argv.index("--pad-token-id")+1]=="0"
    assert argv[argv.index("--target-sha256")+1]==m["graphs"][0]["sha256"]
    import shlex
    assert shlex.split(data["shell"])==argv


def _smoke_args(root, tmp_path, output):
    return parser().parse_args(["board-command", "--mode", "smoke", "--omc", str(root),
        "--prompt", str(tmp_path / "prompt.json"), "--board-model-dir", "/data/models",
        "--board-output", "/data/smoke.json", "--runtime-version", "fake-version",
        "--firmware-id", "fake-fw", "--output", str(output)])


def test_smoke_command_needs_only_prompt_capacity_and_preserves_hashes(bundle, tmp_path):
    root, manifest = bundle
    # A full gear is valid for two-graph smoke despite max_new_tokens defaulting
    # to 32. EOS in this prompt must not introduce generation semantics.
    prompt = [10] * manifest["sequence_length"]
    (tmp_path / "prompt.json").write_text(json.dumps({"prompt_token_ids": prompt}))
    output = tmp_path / "smoke-command.json"
    execute(_smoke_args(root, tmp_path, output))
    data = json.loads(output.read_text()); argv = data["argv"]
    assert argv[argv.index("--mode") + 1] == "smoke"
    assert argv[argv.index("--prompt-token-ids") + 1] == ",".join(map(str, prompt))
    for name in ("target", "draft"):
        graph = next(g for g in manifest["graphs"] if g["name"] == name)
        assert argv[argv.index("--" + name + "-sha256") + 1] == graph["sha256"]
    assert data["manifest_sha256"] == hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()


@pytest.mark.parametrize("mutation", ["empty-prompt", "long-prompt", "corrupt-model", "old-abi"])
def test_smoke_command_retains_input_artifact_and_abi_gates(bundle, tmp_path, mutation):
    root, manifest = bundle
    if mutation in ("empty-prompt", "long-prompt"):
        tokens = [] if mutation == "empty-prompt" else [1] * (manifest["sequence_length"] + 1)
        (tmp_path / "prompt.json").write_text(json.dumps({"prompt_token_ids": tokens}))
    if mutation == "corrupt-model":
        (root / "target.omc").write_bytes(b"corrupt")
    if mutation == "old-abi":
        manifest["abi_version"] = 1
        (root / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "smoke-command.json"
    with pytest.raises(ValueError):
        execute(_smoke_args(root, tmp_path, output))
    assert not output.exists()


@pytest.mark.parametrize("mutation",["none","fake-runtime","wrong-weights","wrong-limit","wrong-tokens",
                                     "wrong-precision","wrong-quantization","wrong-abi"])
def test_comparison_gates(bundle,tmp_path,mutation):
    root,m=bundle
    sample={"generated_token_ids":[3,4],"stop_reason":"length"}
    mode={"stable_generated_token_ids":[3,4],"stable_stop_reason":"length","measurements":[sample]*10}
    board={"status":"PASS","evidence_kind":"device-hiai","source":{"kind":"TEST-DOUBLE"},
           "prompt_token_ids":[1,2],"max_new_tokens":2,"sequence_length":16,
           "ordinary":mode,"dflash":copy.deepcopy(mode),
           "target_sha256":m["graphs"][0]["sha256"],"draft_sha256":m["graphs"][1]["sha256"],
           "abi_version":2,"feature_dtype":"torch.float16"}
    reference={"status":"PASS","source":m["source"],"prompt_token_ids":[1,2],"max_new_tokens":2,
               "sequence_length":16,"ordinary":{"token_ids":[3,4],"stop_reason":"length"},
               "precision":"torch.float16","quantization":None}
    if mutation=="fake-runtime": board["evidence_kind"]="host-fake-hiai"
    if mutation=="wrong-weights": reference["source"]={"kind":"different"}
    if mutation=="wrong-limit": board["max_new_tokens"]=3
    if mutation=="wrong-tokens": board["dflash"]["stable_generated_token_ids"]=[3,5]
    if mutation=="wrong-precision": reference["precision"]="upstream-bf16"
    if mutation=="wrong-quantization": reference["quantization"]={"different":True}
    if mutation=="wrong-abi": board["abi_version"]=1
    (tmp_path/"board.json").write_text(json.dumps(board)); (tmp_path/"ref.json").write_text(json.dumps(reference))
    output=tmp_path/"comparison.json"
    args=parser().parse_args(["compare","--report",str(tmp_path/"board.json"),"--reference",str(tmp_path/"ref.json"),
                             "--omc",str(root),"--output",str(output)])
    if mutation=="none": execute(args)
    else:
        with pytest.raises(ValueError,match="comparison failed"): execute(args)
    assert json.loads(output.read_text())["status"] == ("PASS" if mutation=="none" else "FAIL")
