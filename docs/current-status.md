# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A（DDK 服务器，另开窗口）：prefill64 已启动，先采两次只读进程状态并查看最后非句点日志，保持当前编译运行。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 - "$AI_RUN_DIR/target-pytorch-prefill64-compile" <<'PY'
import json, shutil, subprocess, sys, time
from pathlib import Path

job = Path(sys.argv[1]).resolve()
comp = job / "compilation"
command = json.loads((comp / "compile-command.json").read_text())
exe = command["argv"][0]
model_arg = next(arg for arg in command["argv"] if arg.startswith("--model="))

def sample():
    raw = subprocess.run(
        ["ps", "-ww", "-eo", "pid=,ppid=,stat=,pcpu=,time=,etime=,wchan:28=,args="],
        capture_output=True, text=True, check=True).stdout
    rows = {}
    for line in raw.splitlines():
        values = line.split(None, 7)
        if len(values) == 8:
            rows[int(values[0])] = values
    roots = {pid for pid, row in rows.items()
             if row[7].startswith(exe + " ") and model_arg in row[7].split()}
    selected = set(roots)
    while True:
        children = {pid for pid, row in rows.items() if int(row[1]) in selected}
        added = children - selected
        if not added:
            break
        selected.update(added)
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "matching_omg_pids": sorted(roots),
        "columns": ["pid", "ppid", "state", "lifetime_cpu_percent", "cpu_time", "elapsed", "wchan", "args"],
        "processes": [rows[pid][:-1] + [rows[pid][-1][:320]] for pid in sorted(selected)]}

print(json.dumps({"sample_1": sample()}, ensure_ascii=False), flush=True)
time.sleep(5)
print(json.dumps({"sample_2": sample()}, ensure_ascii=False), flush=True)
log = comp / "compile.log"
info = log.stat()
with log.open("rb") as stream:
    stream.seek(max(0, info.st_size - 65536))
    tail = stream.read(65536).decode("utf-8", errors="replace")
lines = [line.strip() for line in tail.splitlines()
         if line.strip() and set(line.strip()) != {"."}]
print(json.dumps({"log_bytes": info.st_size,
    "log_idle_seconds": round(max(0, time.time() - info.st_mtime), 1),
    "tail_read_limit_bytes": 65536, "last_non_dot_lines": [line[:600] for line in lines[-35:]],
    "disk_free_bytes": shutil.disk_usage(comp).free,
    "scope": "Read-only samples; CPU percent is a lifetime average. Compare both CPU-time samples and child states; output alone does not prove progress."},
    ensure_ascii=False), flush=True)
PY
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
