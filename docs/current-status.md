# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A（DDK 服务器，另开窗口）：TBE/TVM 已报错且进程在等待，读取最后三条完整异常和前文，先不重启编译。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 - "$AI_RUN_DIR/target-pytorch-prefill64-compile/compilation/compile.log" <<'PY'
import json, sys
from pathlib import Path

log = Path(sys.argv[1])
size = log.stat().st_size
start = max(0, size - 131072)
with log.open("rb") as stream:
    stream.seek(start)
    raw = stream.read(131072)
# The first line may begin outside this bounded tail; label it separately.
text = raw.decode("utf-8", errors="replace")
lines = text.splitlines()
anchors = [i for i, line in enumerate(lines) if "TVMError:" in line or "EB9999" in line]
print(json.dumps({"log": str(log), "snapshot_bytes": size, "tail_start_byte": start,
    "tail_read_bytes": len(raw), "matching_error_lines": len(anchors),
    "first_tail_line_may_be_partial": start != 0,
    "scope": "Read-only log tail; full selected lines, no per-line truncation, no model or process control"}, ensure_ascii=False))
if anchors:
    for i in anchors[-3:]:
        print("\n--- complete error and preceding context ---")
        print("\n".join(lines[max(0, i - 12):i + 2]))
else:
    print("\n".join(line for line in lines[-100:] if line.strip() and set(line.strip()) != {"."}))
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
