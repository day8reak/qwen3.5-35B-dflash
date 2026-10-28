"""Full topology export integration without opening real checkpoint payloads."""
from pathlib import Path

import pytest

from qwen35_moe_dflash.draft_standalone import _graph
from qwen35_moe_dflash.ge_cpp import emit_cpp
from qwen35_moe_dflash.ge_recipe_read import read_recipe


@pytest.mark.parametrize("tiny,rows", [(False, 64), (False, 8), (True, 8)])
def test_full_projection_policy_preserves_reference_graph_and_all_other_matmuls(tmp_path, tiny, rows):
    graph = _graph(rows, tiny=tiny)[2]
    before = [(n.kind, n.inputs, n.output, dict(n.attrs)) for n in graph.nodes]
    paths = {key: tmp_path / "weights" / f"w{i}.bin" for i, key in enumerate(graph.weights)}
    report = {}
    emit_cpp(graph, paths, external_weights=True, recipe_dir=tmp_path / "recipe",
             fold_constant_arithmetic=False, fp16_reductions="matmul", fp16_row_mul="compact",
             file_constant_policy="cpro-relative-int", file_constant_root=tmp_path,
             draft_fc_policy="conv2d", draft_fc_report=report)
    recipe_path = tmp_path / "recipe/graph.recipe"
    recipe = read_recipe(recipe_path)
    convs = [r for r in recipe["records"] if r["type"] == "Conv2D"]
    assert len(convs) == len(report["replaced"]) == (0 if tiny else 1)
    assert recipe_path.read_bytes()[:8] == (b"Q35GE001" if tiny else b"Q35GE002")
    if not tiny:
        assert report["replaced"][0]["input_view"] == [rows, 1024, 4, 4]
        assert convs[0]["inputs"][1]["shape"] == [2048, 1024, 4, 4]
        assert all(p["format"] == "NCHW" for p in convs[0]["inputs"] + convs[0]["outputs"])
    for node in recipe["records"]:
        if node["type"] == "MatMul":
            transpose = node["attrs"].get("transpose_x1", {}).get("value", False)
            assert node["inputs"][0]["shape"][-2 if transpose else -1] <= 8000
    # Input order and full-vocabulary head stay bound to the original graph.
    assert [recipe["records"][i]["name"] for i in recipe["inputs"]] == [v.name for v in graph.inputs]
    assert [recipe["records"][i]["name"] for i, _ in recipe["outputs"]] == [v.name for v in graph.outputs]
    head, = [r for r in recipe["records"] if r["name"] == "target.lm_head.weight"]
    assert head["outputs"][0]["shape"] == list(graph.weights["target.lm_head.weight"].shape)
    assert [(n.kind, n.inputs, n.output, dict(n.attrs)) for n in graph.nodes] == before
    assert not (tmp_path / "weights").exists()


def test_default_native_policy_retains_original_large_matmul(tmp_path):
    graph = _graph(64)[2]
    report = {}
    paths = {key: Path("/not-opened") / key for key in graph.weights}
    source = emit_cpp(graph, paths, external_weights=True, fold_constant_arithmetic=False,
                      draft_fc_report=report)
    assert '"__ge/v20/flat_output", "MatMul"' in source
    assert '"Conv2D"' not in source
    assert report["mode"] == "native" and report["replaced"] == []
