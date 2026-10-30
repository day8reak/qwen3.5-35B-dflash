"""Explicit experimental control inputs preserve all original bytes/identities."""
import json
import os
from pathlib import Path
import stat
import struct
import time

import pytest

from qwen35_moe_dflash import target_decode_trial_inputs as trial
from qwen35_moe_dflash.target_decode_smoke_plan import _decode_expected_files


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    source = tmp_path / "user-decode"
    source.mkdir()
    for index, row in enumerate(_decode_expected_files()):
        if row["file"] != "conv_state_pos.bin":
            (source / row["file"]).write_bytes(bytes([index + 1]) * row["expected_bytes"])
    (source / "golden_logits.bin").write_bytes(b"original-golden-not-an-input")
    (source / "unknown.weight").write_bytes(b"unrelated-weights-not-an-input")
    return {"source": source, "out": tmp_path / "trial"}


def snapshot(source):
    return {path.name: (path.read_bytes(), path.stat()) for path in source.iterdir() if path.is_file()}


def unchanged(source, before):
    assert {path.name for path in source.iterdir()} == set(before)
    for name, (raw, old) in before.items():
        path = source / name
        now = path.stat()
        assert path.read_bytes() == raw
        assert (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns, now.st_ctime_ns,
                now.st_mode, now.st_uid, now.st_gid, now.st_nlink) == (
            old.st_dev, old.st_ino, old.st_size, old.st_mtime_ns, old.st_ctime_ns,
            old.st_mode, old.st_uid, old.st_gid, old.st_nlink)


def prepare(case):
    return trial.prepare_decode_trial_inputs(case["source"], case["out"],
        experimental_conv_current_query_count=True)


def write_manifest(root, manifest):
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_explicit_trial_copies_only_twelve_required_originals_and_generates_one512byte_control(case):
    before = snapshot(case["source"])
    manifest = prepare(case)
    assert manifest["status"] == "TARGET_DECODE_TRIAL_INPUTS_PREPARED"
    assert manifest["assumption"]["contract_status"] == "NOT_A_VALIDATED_KERNEL_CONTRACT"
    assert manifest["assumption"]["current_query_count"] == 1
    assert manifest["numerical_validation"] == "NOT_RUN_NO_REFERENCE"
    assert manifest["model_compilation"] == manifest["device_execution"] == "NOT_RUN"
    assert manifest["prefill_state_feedback"] == "NOT_RUN"
    assert manifest["model_payload_access"] == manifest["weight_payload_access"] == "NONE"
    assert manifest["links_created"] == 0
    assert len(manifest["source_bindings"]) == len(manifest["copied_inputs"]) == 12
    assert manifest["input_dir"] == str(case["out"] / "inputs")
    generated = Path(manifest["input_dir"]) / "conv_state_pos.bin"
    assert generated.read_bytes() == struct.pack("<128i", 1, *([0] * 127))
    assert generated.stat().st_size == 512
    assert manifest["generated_control"]["sha256"] == "d839a3521723b8a55d09d8eed9848940b284828e4d09218202c3ee11046bc16d"
    assert manifest["explicit_copy_bytes"] == sum(row["bytes"] for row in manifest["copied_inputs"])
    assert manifest["explicit_copy_bytes"] < 6 << 20
    assert not (case["source"] / "conv_state_pos.bin").exists()
    assert {path.name for path in Path(manifest["input_dir"]).iterdir()} == {
        *[row["file"] for row in manifest["copied_inputs"]], "conv_state_pos.bin"}
    for row in manifest["copied_inputs"]:
        destination = case["out"] / row["path"]
        assert destination.read_bytes() == before[row["file"]][0]
        assert destination.stat().st_nlink == 1
        assert destination.stat().st_ino != (case["source"] / row["file"]).stat().st_ino
    assert trial.validate_decode_trial_inputs(case["out"]) == manifest
    unchanged(case["source"], before)


@pytest.mark.parametrize("flag", [False, None, 1, "true"])
def test_no_implicit_generation_without_explicit_boolean_true(case, flag):
    with pytest.raises(ValueError, match="explicit"):
        trial.prepare_decode_trial_inputs(case["source"], case["out"],
            experimental_conv_current_query_count=flag)
    assert not case["out"].exists()


@pytest.mark.parametrize("fault", ["missing", "wrong-size", "symlink"])
def test_other_original_file_problems_fail_before_output_creation(case, fault):
    path = case["source"] / "inputs_embeds.bin"
    if fault == "missing": path.unlink()
    elif fault == "wrong-size": path.write_bytes(b"wrong")
    else:
        path.rename(case["source"] / "original-embedding.bin")
        path.symlink_to(case["source"] / "original-embedding.bin")
    with pytest.raises((ValueError, OSError)):
        prepare(case)
    assert not case["out"].exists()


@pytest.mark.parametrize("existing", ["correct-size", "wrong-size", "symlink", "dangling"])
def test_existing_control_of_any_kind_is_never_overwritten(case, existing):
    control = case["source"] / "conv_state_pos.bin"
    if existing == "correct-size": control.write_bytes(bytes(512))
    elif existing == "wrong-size": control.write_bytes(b"x")
    elif existing == "symlink": control.symlink_to(case["source"] / "allQLen.bin")
    else: control.symlink_to(case["source"] / "missing")
    before = control.lstat()
    with pytest.raises(ValueError, match="absent"):
        prepare(case)
    assert control.lstat() == before
    assert not case["out"].exists()


@pytest.mark.parametrize("fault", ["source", "copy", "control", "appeared-control", "assumption", "source-path", "inventory", "scope", "missing-copy-stat", "unlisted-file"])
def test_readonly_validator_rejects_changed_original_copied_control_or_provenance(case, fault):
    manifest = prepare(case)
    if fault == "source":
        source = case["source"] / "output_pos.bin"
        source.write_bytes(bytes([99]) * source.stat().st_size)
    elif fault == "copy":
        copied = case["out"] / manifest["copied_inputs"][0]["path"]
        copied.write_bytes(bytes([99]) * copied.stat().st_size)
    elif fault == "control":
        (case["out"] / "inputs/conv_state_pos.bin").write_bytes(struct.pack("<128i", 65, *([0] * 127)))
    elif fault == "appeared-control":
        (case["source"] / "conv_state_pos.bin").write_bytes(bytes(512))
    elif fault == "unlisted-file":
        (case["out"] / "inputs/unlisted.bin").write_bytes(b"unknown")
    else:
        if fault == "assumption": manifest["assumption"]["contract_status"] = "VALIDATED"
        elif fault == "source-path": manifest["source_bindings"][0]["path"] = str(case["source"] / "golden_logits.bin")
        elif fault == "inventory": manifest["copied_inputs"].pop()
        elif fault == "missing-copy-stat": manifest["copied_inputs"][0].pop("stat")
        else: manifest["execution_scope"] = "PREFILL_CONTINUATION"
        write_manifest(case["out"], manifest)
    before = (case["out"] / "manifest.json").read_bytes()
    with pytest.raises((ValueError, OSError)):
        trial.validate_decode_trial_inputs(case["out"])
    assert (case["out"] / "manifest.json").read_bytes() == before


def test_trial_refuses_source_overlap_and_existing_output(case):
    for path in (case["source"], case["source"] / "child"):
        with pytest.raises(ValueError, match="separate"):
            trial.prepare_decode_trial_inputs(case["source"], path, experimental_conv_current_query_count=True)
    case["out"].mkdir()
    marker = case["out"] / "keep"
    marker.write_bytes(b"existing-result")
    with pytest.raises(FileExistsError): prepare(case)
    assert marker.read_bytes() == b"existing-result"


def test_source_mutation_after_preflight_is_detected_before_publishing(case, monkeypatch):
    original = trial._source_rows
    def mutate(source, identity, expected):
        rows = original(source, identity, expected)
        path = source / "output_pos.bin"
        path.write_bytes(b"\x00" * path.stat().st_size)
        return rows
    monkeypatch.setattr(trial, "_source_rows", mutate)
    with pytest.raises(ValueError, match="changed"):
        prepare(case)
    assert not (case["out"] / "manifest.json").exists()


def test_required_copy_paths_never_open_golden_or_weight_extras(case, monkeypatch):
    original = trial._hash
    def guarded(path, **kwargs):
        assert Path(path).name not in {"golden_logits.bin", "unknown.weight"}
        return original(path, **kwargs)
    monkeypatch.setattr(trial, "_hash", guarded)
    manifest = prepare(case)
    assert all(not row["file"].startswith("golden") for row in manifest["source_bindings"])


def test_single_token_recipe_records_original_stale_controls_and_replaces_only_four_derived_files(case):
    source = case["source"]
    original_values = {"output_pos.bin": ("<i", 255), "allQLen.bin": ("<q", 1224),
                       "effective_length.bin": ("<h", 256)}
    for filename, (encoding, value) in original_values.items():
        (source / filename).write_bytes(struct.pack(encoding, value))
    before = snapshot(source)
    manifest = trial.prepare_decode_trial_inputs(source, case["out"], experimental_single_token_controls=True)
    assert manifest["experimental_single_token_controls"] is True
    assert manifest["experimental_conv_current_query_count"] is False
    assert len(manifest["source_bindings"]) == 12
    assert len(manifest["copied_inputs"]) == 9
    assert len(manifest["generated_controls"]) == 4
    assert manifest["generated_control_bytes"] == 526
    assert manifest["generated_control"] == manifest["generated_controls"][0]
    changes = {row["file"]: row for row in manifest["assumption"]["control_replacements"]}
    assert changes["conv_state_pos.bin"]["old_values"] is None
    expected = {"output_pos.bin": ("<i", 0), "allQLen.bin": ("<q", 1), "effective_length.bin": ("<h", 1)}
    for filename, (encoding, value) in expected.items():
        assert changes[filename]["old_values"] == [original_values[filename][1]]
        assert changes[filename]["new_values"] == [value]
        assert changes[filename]["old_sha256"] == next(row["sha256"] for row in manifest["source_bindings"] if row["file"] == filename)
        assert (case["out"] / "inputs" / filename).read_bytes() == struct.pack(encoding, value)
    for row in manifest["copied_inputs"]:
        assert row["file"] not in expected
        assert (case["out"] / row["path"]).read_bytes() == before[row["file"]][0]
    assert trial.validate_decode_trial_inputs(case["out"]) == manifest
    unchanged(source, before)


@pytest.mark.parametrize("fault", ["one-generated-value", "old-value", "missing-generated-record", "copied-original-control", "mode", "claimed-device-run"])
def test_single_token_validator_reconstructs_full_recipe_and_old_source_values(case, fault):
    (case["source"] / "output_pos.bin").write_bytes(struct.pack("<i", 255))
    (case["source"] / "allQLen.bin").write_bytes(struct.pack("<q", 1224))
    (case["source"] / "effective_length.bin").write_bytes(struct.pack("<h", 256))
    manifest = trial.prepare_decode_trial_inputs(case["source"], case["out"], experimental_single_token_controls=True)
    if fault == "one-generated-value":
        (case["out"] / "inputs/output_pos.bin").write_bytes(struct.pack("<i", 255))
    else:
        if fault == "old-value": manifest["assumption"]["control_replacements"][1]["old_values"] = [0]
        elif fault == "missing-generated-record": manifest["generated_controls"].pop()
        elif fault == "copied-original-control": manifest["copied_inputs"].append(manifest["source_bindings"][0])
        elif fault == "mode": manifest["experimental_single_token_controls"] = False
        else: manifest["device_execution"] = "EXECUTED"
        write_manifest(case["out"], manifest)
    before = (case["out"] / "manifest.json").read_bytes()
    with pytest.raises((OSError, ValueError)):
        trial.validate_decode_trial_inputs(case["out"])
    assert (case["out"] / "manifest.json").read_bytes() == before


def touch_ctime(path):
    old = path.stat()
    time.sleep(.02)
    mode = stat.S_IMODE(old.st_mode)
    path.chmod(mode ^ stat.S_IXUSR)
    path.chmod(mode)
    new = path.stat()
    assert (old.st_dev, old.st_ino, old.st_size, old.st_mtime_ns) == (
        new.st_dev, new.st_ino, new.st_size, new.st_mtime_ns)
    assert old.st_ctime_ns != new.st_ctime_ns


def test_all13_derived_ctime_changes_revalidate_without_mutating_originals_manifest_or_recipe(case):
    manifest = trial.prepare_decode_trial_inputs(case["source"], case["out"], experimental_single_token_controls=True)
    before = snapshot(case["source"])
    raw_manifest = (case["out"] / "manifest.json").read_bytes()
    inputs = [case["out"] / row["path"] for row in manifest["copied_inputs"] + manifest["generated_controls"]]
    assert len(inputs) == 13
    for path in inputs:
        touch_ctime(path)
    observations = []
    assert trial.validate_decode_trial_inputs(case["out"], payload_revalidation_observations=observations) == manifest
    assert len(observations) == 13
    assert {row["path"] for row in observations} == {str(path) for path in inputs}
    assert sum(row["role"] == "decode-trial-copied" for row in observations) == 9
    assert sum(row["role"] == "decode-trial-generated" for row in observations) == 4
    assert all(row["kind"] == "PAYLOAD_CTIME_ONLY_REVALIDATED" and row["status"] == "SHA256_VERIFIED" for row in observations)
    assert (case["out"] / "manifest.json").read_bytes() == raw_manifest
    unchanged(case["source"], before)


@pytest.mark.parametrize("kind", ["copy", "generated"])
@pytest.mark.parametrize("change", ["mtime", "inode", "content", "hardlink"])
def test_derived_payload_revalidation_never_accepts_other_identity_or_recipe_drift(case, kind, change):
    manifest = trial.prepare_decode_trial_inputs(case["source"], case["out"], experimental_single_token_controls=True)
    path = case["out"] / (manifest["copied_inputs"][0] if kind == "copy" else manifest["generated_controls"][0])["path"]
    before, raw = path.stat(), path.read_bytes()
    if change == "mtime":
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1))
    elif change == "inode":
        other = case["out"] / "replacement"
        other.write_bytes(raw)
        os.utime(other, ns=(before.st_atime_ns, before.st_mtime_ns))
        other.replace(path)
    elif change == "content":
        path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    else:
        os.link(path, case["out"] / "extra-link")
    with pytest.raises((ValueError, OSError)):
        trial.validate_decode_trial_inputs(case["out"], payload_revalidation_observations=[])


def test_original_source_ctime_stays_strict_when_derived_reconciliation_is_enabled(case):
    prepare(case)
    touch_ctime(case["source"] / "position_ids.bin")
    with pytest.raises(ValueError, match="original.*identit"):
        trial.validate_decode_trial_inputs(case["out"], payload_revalidation_observations=[])


def test_derived_change_after_fresh_hash_is_rejected_at_final_stability_gate(case, monkeypatch):
    prepare(case)
    original, changed = trial._hash, False
    def mutate(path, **options):
        nonlocal changed
        result = original(path, **options)
        if not changed and Path(path).parent == case["out"] / "inputs":
            changed = True
            touch_ctime(Path(path))
        return result
    monkeypatch.setattr(trial, "_hash", mutate)
    with pytest.raises(ValueError, match="changed after fresh hash"):
        trial.validate_decode_trial_inputs(case["out"], payload_revalidation_observations=[])


def test_payload_revalidation_collector_type_is_explicit(case):
    prepare(case)
    with pytest.raises(ValueError, match="must be a list"):
        trial.validate_decode_trial_inputs(case["out"], payload_revalidation_observations={})
