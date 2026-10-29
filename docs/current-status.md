# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A（DDK 服务器，另开窗口）：先读取目录校验源码和路径，再只结束本次 OMG 进程树，让原窗口继续完成输入审计及失败报告。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 - "$AI_RUN_DIR/target-pytorch-prefill64-compile" <<'PY'
import ast, hashlib, json, os, stat, sys
from pathlib import Path

comp = Path(sys.argv[1]).resolve() / "compilation"
record = json.loads((comp / "omg-environment.json").read_text())
command = json.loads((comp / "compile-command.json").read_text())
print(json.dumps({"environment": {key: record.get("environment", {}).get(key) for key in ("TMPDIR", "TMP", "TEMP")},
    "ipc_temporary_directory": record.get("ipc_temporary_directory"), "compiler_cwd": command.get("cwd")}, ensure_ascii=False))
paths = [command["cwd"]] + [record.get("ipc_temporary_directory", {}).get(key) for key in ("path", "physical_path")]
for value in paths:
    if not value:
        continue
    p = Path(value)
    observed = {"path": value, "path_bytes": len(os.fsencode(value)), "resolved": str(p.resolve())}
    try:
        info = p.stat()
        observed.update(exists=True, mode=oct(stat.S_IMODE(info.st_mode)), uid=info.st_uid, gid=info.st_gid)
    except OSError as error:
        observed.update(exists=False, error=str(error))
    print(json.dumps(observed, ensure_ascii=False))
p = comp / "tbe-compat/wheel-python/te-0.4.0.data/purelib/tbe/tvm/contrib/ccec.py"
with p.open("rb") as stream:
    raw = stream.read(524289)
if len(raw) > 524288:
    raise SystemExit("ccec.py exceeds the small-source budget; no vendor code executed")
text = raw.decode("utf-8")
lines = text.splitlines()
print(json.dumps({"source": str(p), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "vendor_import": "NOT_PERFORMED"}))
for node in ast.walk(ast.parse(text)):
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in {"_check_tmpdir", "compile_cce"}:
        continue
    chosen = set(range(node.lineno - 1, min(node.end_lineno, node.lineno + 39)))
    if node.name == "_check_tmpdir":
        chosen = set(range(node.lineno - 1, node.end_lineno))
    else:
        for i in range(node.lineno - 1, node.end_lineno):
            if any(word in lines[i] for word in ("dirpath", "tmpdir", "tempfile", "kernel_meta", "work_dir")):
                chosen.update(range(max(node.lineno - 1, i - 6), min(node.end_lineno, i + 9)))
    selected = sorted(chosen)[:180]
    print(json.dumps({"function": node.name, "omitted_selected_lines": max(0, len(chosen) - len(selected)),
        "source_lines": [str(i + 1) + ": " + lines[i] for i in selected]}, ensure_ascii=False))
PY
python3 "$REPO_DIR/targets/cpro/scripts/stop_target_compile.py"   --job "$AI_RUN_DIR/target-pytorch-prefill64-compile" --apply   --output "$AI_RUN_DIR/target-pytorch-prefill64-stop"
```

A（原权重机器）：将下面三个小文件手动上传到 DDK 的 `$AI_RUN_DIR/target-pytorch-decode1-transfer/`，平铺保存为 `export.air`、`inventory.json`、`result.json`。

```bash
ls -lh \
  "$AI_RUN_DIR/target-pytorch-decode1/canonical-export/export.air" \
  "$AI_RUN_DIR/target-pytorch-decode1/canonical-export/inventory/inventory.json" \
  "$AI_RUN_DIR/target-pytorch-decode1/canonical-export/result.json"
```

A（DDK 服务器，三个小文件上传完成后）：只核对 decode1 的文件哈希、特征声明及实际输入名，不读取大权重或创建硬链接。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/stage_target_export_reuse.py" \
  --inspect-transfer --sequence-length 1 \
  --export-receipt "$AI_RUN_DIR/target-pytorch-decode1-transfer/result.json" \
  --inventory "$AI_RUN_DIR/target-pytorch-decode1-transfer/inventory.json" \
  --air "$AI_RUN_DIR/target-pytorch-decode1-transfer/export.air"
```

B：55 档特征 AIR 和文件清单已完成，本轮保留备份，无需重复生成或编译。
