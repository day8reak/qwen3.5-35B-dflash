"""Shared tbe environment utilities used by model preparation and execution."""
from __future__ import annotations
import ast
from pathlib import Path

_SOURCE = Path("tbe-compat/wheel-python/te-0.4.0.data/purelib/tbe/tvm/contrib/ccec.py")

_MAX_SOURCE = 512 << 10

_CHILD = r'''import hashlib, json, multiprocessing as mp, os, pathlib, sys, tempfile, traceback, types
from multiprocessing.connection import Client, Listener

def socket_worker(pipe):
    with Listener(family="AF_UNIX") as server:
        pipe.send(server.address)
        with server.accept() as client:
            client.send(client.recv() + 1)

if __name__ == "__main__":
    result = {"status": "FAILED", "scope": "SOURCE_PREDICATE_ONLY_NOT_NATIVE_TBE_BUILD",
              "vendor_package_import": "NOT_PERFORMED", "model_compilation": "NOT_RUN", "device_execution": "NOT_RUN"}
    target, source = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    try:
        physical, short = os.environ["TMPDIR"], sys.argv[3]
        assert all(os.environ[key] == physical for key in ("TMP", "TEMP"))
        assert os.path.realpath(physical) == physical
        assert tempfile.gettempdir() == physical
        assert mp.current_process()._config.get("tempdir") == short
        module_path = pathlib.Path(mp.__file__)
        result["multiprocessing_module"] = {"path": str(module_path),
            "sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
            "spec_origin": getattr(mp.__spec__, "origin", None)}
        overlay = getattr(mp, "_qwen35_ipc_overlay", None)
        assert isinstance(overlay, dict), "owned multiprocessing overlay was not observed"
        assert overlay.get("physical_tmpdir") == physical and overlay.get("multiprocessing_tmpdir") == short
        assert pathlib.Path(overlay["path"]).is_relative_to(pathlib.Path(physical))
        assert hashlib.sha256(pathlib.Path(overlay["path"]).read_bytes()).hexdigest() == overlay["sha256"]
        result["multiprocessing_overlay"] = overlay
        result["ipc_configuration"] = "BOUND_SHORT_ALIAS_OBSERVED_IN_ACTUAL_PROCESS_CONFIG"
        # Execute only the original directory-check function, never its module.
        for name in ("tbe", "tbe.common", "tbe.common.buildcfg", "tbe.common.buildcfg.buildcfg_mapping"):
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
        mapping = sys.modules["tbe.common.buildcfg.buildcfg_mapping"]
        mapping.tbe_debug_level = "debug"
        mapping.op_debug_config = "op_debug"
        mapping.random_cce_file_location = "random"
        config = {"debug": False, "op_debug": [], "random": True}
        def reject(code, message):
            raise RuntimeError(message)
        namespace = {"os": os, "sys": sys, "current_build_config": lambda: config,
                     "TBE_DEFAULT_PYTHON_ERROR_CODE": "EB9999", "raise_tbe_python_err": reject}
        exec(compile(source.read_text(), str(source), "exec"), namespace)
        check = namespace["_check_tmpdir"]
        with tempfile.TemporaryDirectory(prefix="cce_probe_", dir=physical) as directory:
            canonical = os.path.realpath(directory)
            os.environ["TMPDIR"] = short
            try:
                check(canonical)
            except RuntimeError as error:
                assert str(error) == "dir path is invalid", repr(error)
                result["old_alias_control"] = "REJECTED_AS_EXPECTED"
            else:
                raise AssertionError("old alias control did not reproduce directory rejection")
            finally:
                os.environ["TMPDIR"] = physical
            check(canonical)
            result["physical_tmpdir_predicate"] = "PASSED"
        result["controlled_build_flags"] = {"debug": False, "dump": False, "random_cce_file_location": True}
        ctx = mp.get_context("forkserver")
        recv, send = ctx.Pipe(duplex=False)
        child = ctx.Process(target=socket_worker, args=(send,))
        child.start()
        send.close()
        try:
            assert recv.poll(10), "forkserver child did not publish its socket"
            address = recv.recv()
            assert address.startswith(short + "/")
            assert len(os.fsencode(address)) < 108
            assert pathlib.Path(address).resolve().is_relative_to(pathlib.Path(physical))
            with Client(address) as client:
                client.send(41)
                assert client.recv() == 42
            child.join(10)
            assert child.exitcode == 0, child.exitcode
            result["forkserver_socket"] = {"status": "PASSED", "path": address, "bytes": len(os.fsencode(address))}
        finally:
            recv.close()
            if child.is_alive():
                child.terminate()
                child.join(2)
        result.update(status="PROBE_PASSED", physical_tmpdir=physical, multiprocessing_tmpdir=short,
                      embedded_omg_startup="NOT_TESTED", native_cce_compilation="NOT_RUN",
                      attribution="SOURCE_PATH_CONTRACT_REPRODUCTION; RUNTIME_DIRPATH_NOT_CAPTURED")
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    raise SystemExit(0 if result["status"] == "PROBE_PASSED" else 1)
'''

def _predicate(raw):
    tree = ast.parse(raw)
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_check_tmpdir"]
    compilers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "compile_cce"]
    if len(matches) != 1 or len(compilers) != 1:
        raise ValueError("source requires one top-level directory predicate and compile_cce")
    function = matches[0]
    if (function.decorator_list or function.args.posonlyargs or len(function.args.args) != 1
            or function.args.args[0].arg != "dirpath" or function.args.defaults
            or function.args.kwonlyargs or function.args.vararg or function.args.kwarg):
        raise ValueError("unsupported directory predicate signature or decorators")
    canonical = checks = None
    for index, node in enumerate(compilers[0].body):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "dirpath"
                and ast.dump(node.value) == ast.dump(ast.parse("os.path.realpath(dirpath)", mode="eval").body)):
            canonical = index
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "_check_tmpdir"
                and not node.value.keywords
                and len(node.value.args) == 1 and isinstance(node.value.args[0], ast.Name)
                and node.value.args[0].id == "dirpath"):
            checks = index
            break
    if canonical is None or checks is None or canonical + 1 != checks:
        raise ValueError("compile_cce must canonicalize dirpath before its directory check")
    return ast.unparse(function) + "\n"
