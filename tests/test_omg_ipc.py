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
        assert physical.is_relative_to(long_scratch)
        child = subprocess.run([sys.executable, str(script), method], env=env, capture_output=True,
                               text=True, timeout=45, check=True)
        report = json.loads(child.stdout)
        assert len(os.fsencode(report["socket"])) < 108
        assert report["physical"] == str(physical)
        assert not list(physical.iterdir())
    assert dict(os.environ) == parent_env
    assert not alias.exists() and not physical.exists()
    assert not list(long_scratch.iterdir())


@pytest.mark.parametrize("error", [RuntimeError("compiler failed"), subprocess.TimeoutExpired("omg", 1)])
def test_ipc_closes_descriptor_and_removes_owned_files_on_error(long_scratch, error):
    with pytest.raises(type(error)):
        with _omg_ipc_environment(long_scratch, {}) as (env, record):
            alias = Path(env["TMPDIR"])
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
    with _omg_ipc_environment(long_scratch, {}) as (env, _):
        old_alias = Path(env["TMPDIR"])
    with _omg_ipc_environment(long_scratch, {}) as (env, _):
        new_alias = Path(env["TMPDIR"])
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
