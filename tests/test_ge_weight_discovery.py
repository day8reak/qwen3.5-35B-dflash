"""Locate only required draft dependencies in partial/indexless local checkpoints."""
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from qwen35_moe_dflash.ge_weights import SafeTensorFiles

EMBED = "model.language_model.embed_tokens.weight"
HEAD = "lm_head.weight"


def shard(path, keys):
    header = {key: {"dtype": "BF16", "shape": [2], "data_offsets": [4*i, 4*i+4]}
              for i, key in enumerate(keys)}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + struct.pack("<HH", 0x3f80, 0x4000)*len(keys))
    return 8 + len(raw)


def test_partial_nine_of_fourteen_shards_discover_without_payload_reads(tmp_path, monkeypatch):
    starts = {}
    for i in range(1, 10):
        path = tmp_path / f"model.safetensors-{i:05}-of-00014.safetensors"
        starts[path] = shard(path, [EMBED, HEAD] if i == 9 else [f"decoder.{i}"])
    original = Path.open

    class HeaderOnly:
        def __init__(self, path, stream):
            self.path, self.stream = path, stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.stream.__exit__(*args)
        def read(self, size=-1):
            assert size >= 0 and self.stream.tell()+size <= starts[self.path], "payload read during discovery"
            return self.stream.read(size)

    def guarded(path, *args, **kwargs):
        stream = original(path, *args, **kwargs)
        return HeaderOnly(path, stream) if path in starts else stream

    with monkeypatch.context() as m:
        m.setattr(Path, "open", guarded)
        reader = SafeTensorFiles(tmp_path, allow_indexless_shards=True, required_keys=[EMBED, HEAD])
        for key in [EMBED, HEAD]:
            assert reader.tensor(key, (2,))[0].name == "model.safetensors-00009-of-00014.safetensors"
    assert reader.discovery["mode"] == "HEADER_SCAN" and len(reader.discovery["files"]) == 9
    assert reader.discovery["index_present"] is False
    assert set(reader.discovery["selected_mapping"]) == {EMBED, HEAD}
    assert all(len(row["header_sha256"]) == 64 for row in reader.discovery["files"])
    output = tmp_path / "head.bin"
    reader.write_fp16(HEAD, (2,), output)
    np.testing.assert_array_equal(np.fromfile(output, dtype="<f2"), [1, 2])
    assert not (tmp_path / "model.safetensors.index.json").exists()


def test_indexless_scan_requires_opt_in(tmp_path):
    shard(tmp_path / "model-00009-of-00014.safetensors", [HEAD])
    with pytest.raises(ValueError, match="missing"):
        SafeTensorFiles(tmp_path)


def test_missing_required_head_names_the_tensor(tmp_path):
    shard(tmp_path / "model.safetensors-00009-of-00014.safetensors", [EMBED])
    with pytest.raises(ValueError, match=r"Required checkpoint tensors missing.*lm_head.weight"):
        SafeTensorFiles(tmp_path, allow_indexless_shards=True, required_keys=[EMBED, HEAD])


@pytest.mark.parametrize("first", ["model.safetensors", "model-00001-of-00014.safetensors"])
def test_ambiguous_shards_are_never_silently_selected(tmp_path, first):
    shard(tmp_path / first, [HEAD])
    shard(tmp_path / "model-00009-of-00014.safetensors", [HEAD])
    with pytest.raises(ValueError, match="Ambiguous checkpoint tensor lm_head.weight"):
        SafeTensorFiles(tmp_path, allow_indexless_shards=True, required_keys=[HEAD])


@pytest.mark.parametrize("raw", [b"[]", b'{"x":{},"x":{}}', b"{not-json}"])
def test_invalid_headers_rejected(tmp_path, raw):
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw))+raw)
    with pytest.raises(ValueError):
        SafeTensorFiles(tmp_path, allow_indexless_shards=True)


def test_selected_tensor_truncation_detected_before_conversion(tmp_path):
    path = tmp_path / "model.safetensors"
    shard(path, [HEAD])
    path.write_bytes(path.read_bytes()[:-2])
    reader = SafeTensorFiles(tmp_path, allow_indexless_shards=True, required_keys=[HEAD])
    with pytest.raises(ValueError, match="invalid safetensors data offsets"):
        reader.tensor(HEAD, (2,))


def test_discovery_refuses_symlink_shards(tmp_path):
    path = tmp_path / "actual"
    shard(path, [HEAD])
    (tmp_path / "model.safetensors").symlink_to(path)
    with pytest.raises(ValueError, match="regular file"):
        SafeTensorFiles(tmp_path, allow_indexless_shards=True)


@pytest.mark.parametrize("limit,value,error", [
    ("MAX_DISCOVERY_SHARDS", 1, "shard limit"),
    ("MAX_DISCOVERY_ENTRIES", 1, "entry limit"),
    ("MAX_DISCOVERY_HEADER_BYTES", 10, "over-budget"),
])
def test_header_discovery_is_bounded(tmp_path, monkeypatch, limit, value, error):
    shard(tmp_path / "a.safetensors", [HEAD])
    shard(tmp_path / "b.safetensors", [EMBED])
    monkeypatch.setattr(SafeTensorFiles, limit, value)
    with pytest.raises(ValueError, match=error):
        SafeTensorFiles(tmp_path, allow_indexless_shards=True)


def test_indexed_lookup_never_reads_unselected_missing_shards(tmp_path):
    shard(tmp_path / "selected.safetensors", [HEAD])
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        HEAD: "selected.safetensors", "decoder": "not-downloaded.safetensors"}}))
    reader = SafeTensorFiles(tmp_path, allow_indexless_shards=True, required_keys=[HEAD])
    assert not reader.headers
    reader.tensor(HEAD, (2,))
    assert len(reader.headers) == 1 and reader.discovery["mode"] == "INDEX"


def test_present_index_is_not_overridden_by_header_guessing(tmp_path):
    shard(tmp_path / "selected.safetensors", [HEAD])
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {HEAD: "absent.safetensors"}}))
    reader = SafeTensorFiles(tmp_path, allow_indexless_shards=True, required_keys=[HEAD])
    with pytest.raises(ValueError, match="missing"):
        reader.tensor(HEAD, (2,))
