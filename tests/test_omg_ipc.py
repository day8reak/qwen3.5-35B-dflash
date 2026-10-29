"""Real host IPC regression; this does not exercise a vendor compiler or NPU."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from qwen35_moe_dflash.compile import _omg_ipc_environment


@pytest.fixture
def long_scratch(tmp_path):
    path = tmp_path / ("long-model-run-" * 8) / "omg-scratch"
    path.mkdir(parents=True)
    assert len(os.fsencode(path)) > 108
    return path


def test_original_long_tmpdir_reproduces_unix_socket_failure(long_scratch):
    child = subprocess.run(
        [sys.executable, "-c", "from multiprocessing.connection import Listener; Listener(family='AF_UNIX')"],
        env=dict(os.environ, TMPDIR=str(long_scratch)), capture_output=True, text=True, timeout=20)
    assert child.returncode != 0
    assert "AF_UNIX path too long" in child.stderr
    assert not list(long_scratch.iterdir())


@pytest.mark.parametrize("method", ["fork", "spawn", "forkserver"])
def test_short_alias_supports_real_manager_and_descendant_socket(long_scratch, tmp_path, method):
    script = tmp_path / "ipc_child.py"
    script.write_text('''import json, multiprocessing as mp, os, sys, tempfile
from pathlib import Path
from multiprocessing.connection import Listener, Client

def worker(pipe):
    assert tempfile.gettempdir() == os.environ["TMPDIR"]
    assert os.path.realpath(os.environ["TMPDIR"]) == os.environ["TMPDIR"]
    with Listener(family="AF_UNIX") as server:
        pipe.send(server.address)
        with server.accept() as client:
            client.send(client.recv() + 1)

if __name__ == "__main__":
    ctx = mp.get_context(sys.argv[1])
    temp = Path(tempfile.gettempdir())
    assert str(temp) == os.environ["TMPDIR"]
    assert os.environ["TMP"] == os.environ["TEMP"] == str(temp)
    physical = temp.resolve()
    assert temp == physical
    # Equivalent to the observed CCE compile_cce/_check_tmpdir prefix check.
    with tempfile.TemporaryDirectory() as cce_temp:
        assert os.path.realpath(cce_temp).startswith(os.environ["TMPDIR"])
    assert mp.current_process()._config["tempdir"] != str(physical)
    with ctx.Manager() as manager:
        shared = manager.dict()
        shared["value"] = 42
        assert shared["value"] == 42
        recv, send = ctx.Pipe(duplex=False)
        child = ctx.Process(target=worker, args=(send,))
        child.start()
        send.close()
        assert recv.poll(15), "worker did not publish its socket"
        address = recv.recv()
        assert Path(address).resolve().is_relative_to(physical)
        with Client(address) as client:
            client.send(41)
            assert client.recv() == 42
        recv.close()
        child.join(15)
        assert child.exitcode == 0
        print(json.dumps({"socket": address, "physical": str(physical), "method": sys.argv[1]}))
''')
    parent_env = dict(os.environ)
    with _omg_ipc_environment(long_scratch, parent_env) as (env, record):
        physical = Path(record["physical_path"])
        alias = Path(record["path"])
        assert record["mode"] == "parent-proc-fd"
        assert env["TMPDIR"] == env["TMP"] == env["TEMP"] == str(physical)
        assert record["multiprocessing_tmpdir"] == str(alias)
        assert Path(env["PYTHONPATH"].split(os.pathsep)[0]) == physical / "python-ipc"
        assert physical.is_relative_to(long_scratch)
        child = subprocess.run([sys.executable, str(script), method], env=env, capture_output=True,
                               text=True, timeout=45, check=True)
        report = json.loads(child.stdout)
        assert len(os.fsencode(report["socket"])) < 108
        assert report["physical"] == str(physical)
        assert all(p.name == "python-ipc" or p.name.startswith("listener-") and p.is_socket()
                   for p in physical.iterdir())
    assert dict(os.environ) == parent_env
    assert not alias.exists() and not physical.exists()
    assert not list(long_scratch.iterdir())


@pytest.mark.parametrize("error", [RuntimeError("compiler failed"), subprocess.TimeoutExpired("omg", 1)])
def test_ipc_closes_descriptor_and_removes_owned_files_on_error(long_scratch, error):
    with pytest.raises(type(error)):
        with _omg_ipc_environment(long_scratch, {}) as (env, record):
            alias = Path(record["path"])
            physical = Path(record["physical_path"])
            (alias / "compiler-leftover").write_bytes(b"owned by this invocation")
            raise error
    assert not alias.exists() and not physical.exists()
    assert not list(long_scratch.iterdir())


def test_simultaneous_invocations_have_independent_storage(long_scratch):
    with _omg_ipc_environment(long_scratch, {}) as (first, one):
        with _omg_ipc_environment(long_scratch, {}) as (second, two):
            assert first["TMPDIR"] != second["TMPDIR"]
            assert one["physical_path"] != two["physical_path"]
            (Path(first["TMPDIR"]) / "probe").write_text("first")
            (Path(second["TMPDIR"]) / "probe").write_text("second")
            assert (Path(one["physical_path"]) / "probe").read_text() == "first"
            assert (Path(two["physical_path"]) / "probe").read_text() == "second"
        assert (Path(first["TMPDIR"]) / "probe").read_text() == "first"
    assert not list(long_scratch.iterdir())


def test_stale_alias_cannot_redirect_to_later_invocation(long_scratch):
    with _omg_ipc_environment(long_scratch, {}) as (env, record):
        old_alias = Path(record["path"])
    with _omg_ipc_environment(long_scratch, {}) as (env, record):
        new_alias = Path(record["path"])
        assert old_alias.name != new_alias.name
        assert not old_alias.exists()
        assert new_alias.is_dir()
        with pytest.raises(FileNotFoundError):
            (old_alias / "old-worker-write").write_text("must not reach the new invocation")
        assert not (new_alias / "old-worker-write").exists()


def test_unavailable_procfs_with_long_path_fails_without_global_fallback(long_scratch, monkeypatch):
    monkeypatch.setattr(Path, "samefile", lambda *args: False)
    with pytest.raises(RuntimeError, match="Enable procfs access"):
        with _omg_ipc_environment(long_scratch, {}):
            pytest.fail("must not pass an unusable long TMPDIR to the compiler")
    assert not list(long_scratch.iterdir())


def test_cce_realpath_check_reproduces_old_alias_rejection_and_accepts_physical_tmpdir(long_scratch):
    with _omg_ipc_environment(long_scratch, {}) as (env, record):
        candidate = os.path.realpath(Path(record["path"]) / "cce-owned-source")
        assert not candidate.startswith(record["path"]), "old TMPDIR alias reproduced the observed mismatch"
        assert candidate.startswith(env["TMPDIR"]), "canonical TMPDIR must satisfy the observed vendor check"
        assert record["multiprocessing_overlay"]["selected_ddk_execution"].startswith("NOT_PROVEN")


@pytest.mark.parametrize("tamper", ["TMPDIR", "TMP", "TEMP", "stale_alias"])
def test_import_rejects_changed_directory_binding_before_multiprocessing_use(long_scratch, tamper):
    with _omg_ipc_environment(long_scratch, {}) as (env, record):
        if tamper == "stale_alias":
            hook = Path(record["multiprocessing_overlay"]["path"])
            hook.write_text(hook.read_text().replace(record["path"], record["path"] + "-stale"))
        else:
            env[tamper] = str(long_scratch)
        result = subprocess.run([sys.executable, "-c", "import multiprocessing; print('child code must not run')"],
                                env=env, capture_output=True, text=True, timeout=10)
        assert result.returncode == 78
        assert "qwen35 OMG IPC import rejected" in result.stderr
        assert "child code must not run" not in result.stdout
    assert not list(long_scratch.iterdir())


def test_child_environment_preserves_inherited_paths_without_parent_mutation(long_scratch):
    parent = {"PYTHONPATH": "/retained/vendor/path", "SOME_VENDOR_OPTION": "same"}
    before = dict(parent)
    with _omg_ipc_environment(long_scratch, parent) as (env, record):
        assert parent == before
        assert env["PYTHONPATH"].split(os.pathsep)[1:] == ["/retained/vendor/path"]
        assert env["SOME_VENDOR_OPTION"] == "same"
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"
        assert Path(record["multiprocessing_overlay"]["path"]).is_file()
    assert parent == before


def test_multiprocessing_overlay_preserves_an_inherited_sitecustomize(long_scratch, tmp_path):
    inherited = tmp_path / "inherited-startup"
    inherited.mkdir()
    marker = tmp_path / "inherited-module-executed-normally"
    (inherited / "sitecustomize.py").write_text(
        "from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('preserved')\n")
    parent = {"PYTHONPATH": str(inherited)}
    with _omg_ipc_environment(long_scratch, parent) as (env, record):
        result = subprocess.run([sys.executable, "-c", "import multiprocessing; print('normal child')"],
                                env=env, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        assert marker.read_text() == "preserved" and result.stdout.strip() == "normal child"
        assert not (Path(record["multiprocessing_overlay"]["path"]).parents[1] / "sitecustomize.py").exists()
    assert not list(long_scratch.iterdir())


def test_overlay_refuses_a_custom_nonstdlib_multiprocessing_package(long_scratch, tmp_path):
    inherited = tmp_path / "custom-multiprocessing"
    inherited.mkdir()
    package = inherited / "multiprocessing"
    package.mkdir()
    marker = tmp_path / "foreign-multiprocessing-must-not-execute"
    (package / "__init__.py").write_text(
        "from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('changed')\n")
    with _omg_ipc_environment(long_scratch, {"PYTHONPATH": str(inherited)}) as (env, record):
        result = subprocess.run([sys.executable, "-c", "import multiprocessing; print('unreachable')"],
                                env=env, capture_output=True, text=True, timeout=10)
        assert result.returncode == 78
        assert "original stdlib package" in result.stderr and not result.stdout
        assert not marker.exists()
    assert not list(long_scratch.iterdir())


def test_overlay_preserves_original_package_identity_and_reports_hashed_origins(long_scratch):
    with _omg_ipc_environment(long_scratch, {}) as (env, record):
        result = subprocess.run([sys.executable, "-c", "import hashlib,json,multiprocessing as mp,sysconfig; "
            "from pathlib import Path; marker=mp._qwen35_ipc_overlay; "
            "assert Path(mp.__file__).resolve()==Path(sysconfig.get_path('stdlib'),'multiprocessing/__init__.py').resolve(); "
            "assert mp.__spec__.name=='multiprocessing'; assert mp.__path__==[str(Path(mp.__spec__.origin).parent)]; "
            "assert marker['stdlib_sha256']==hashlib.sha256(Path(marker['stdlib_path']).read_bytes()).hexdigest(); "
            "print(json.dumps(marker))"], env=env, capture_output=True, text=True, check=True, timeout=10)
        marker = json.loads(result.stdout)
        assert marker["path"] == record["multiprocessing_overlay"]["path"]
        assert marker["sha256"] == record["multiprocessing_overlay"]["sha256"]
        assert marker["physical_tmpdir"] == env["TMPDIR"]
        assert marker["multiprocessing_tmpdir"] == record["path"]
