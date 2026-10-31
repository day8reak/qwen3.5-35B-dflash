# 当前阶段测试命令

<a id="用户现在执行什么"></a>

文字输入、原Prefill和常驻Draft／Decode循环已跑通。当前优先Draft实现、独立精度和性能，Target由用户另行调试，批量verify后置。本轮先生成增量Draft AIR：C16新特征块、192行KV容量，支持15／7／3／1候选档位。此命令复用原六层71份权重，生成新的图，不调用板端。

## 导出一个候选档位

在原DDK主机沿用项目环境。先导出15档；需要其他档位时将变量改为7、3或1，并使用新的输出目录。四档是不同计算形状，不能只修改旧OMC的manifest。

```bash
git -C "$REPO_DIR" pull --ff-only
export MAX_DRAFT_TOKENS=15
export INCREMENTAL_AIR="$AI_RUN_DIR/draft-kv-c16-cap192-k${MAX_DRAFT_TOKENS}-air-v1"

PYTHONDONTWRITEBYTECODE=1 python3 "$REPO_DIR/targets/cpro/scripts/export_incremental_draft.py" \
  --job "$AI_RUN_DIR/full-draft" \
  --context-rows 16 \
  --cache-capacity 192 \
  --max-draft-tokens "$MAX_DRAFT_TOKENS" \
  --output "$INCREMENTAL_AIR"
```

预期摘要为`DRAFT_INCREMENTAL_AIR_EXPORTED`、17输入、14输出。首次会只读校验约2.8GB原权重，并编译GE导出器；不复制权重、不编译OMC、不运行模型。保留并回传摘要；失败时同目录保留日志，下次使用新目录。

## 查看导出结果

成功后进入OMG编译与实际runtime ABI绑定，再接入板端12份缓存。当前`run_board_serial_dflash.py`仍使用旧S64；不要把新AIR目录传给`--draft-bundle`。Draft缓存扩容也不代表Target的长prompt接入已经完成。

```bash
python3 - <<'PYCODE'
import json, os
from pathlib import Path
p = Path(os.environ["INCREMENTAL_AIR"]) / "manifest.json"
d = json.loads(p.read_text())
keys = ("status", "context_rows", "cache_capacity", "max_draft_tokens",
        "model_compilation", "device_execution", "runtime_integration", "error")
print(json.dumps({k: d.get(k) for k in keys}, ensure_ascii=False, indent=2))
print("manifest:", p)
PYCODE
```
