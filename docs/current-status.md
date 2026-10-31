# 当前阶段测试命令

增量Draft的C16／缓存192／K15 AIR已在DDK主机导出成功：17输入、14输出、六层12份KV缓存，不复制原权重。当前执行下一步OMG编译、接口检查和内嵌权重打包。Target精度继续独立调试，新批量verify暂不开发。

## 编译已经导出的K15

复用本次成功的`air-v1`目录，无需重新导出。脚本从原`full-draft/residual-fix`记录选择相同DDK，在新目录编译；原权重仅建立逐文件引用并检查前后完整性。至少预留两份Draft权重大小加1 GiB空间，供新编译权重、打包OMC与临时文件使用；编译器实际峰值可能更高。不要同时编译其他档位。

```bash
git -C "$REPO_DIR" pull --ff-only
export MAX_DRAFT_TOKENS=15
export INCREMENTAL_AIR="$AI_RUN_DIR/draft-kv-c16-cap192-k15-air-v1"
export INCREMENTAL_COMPILED="$AI_RUN_DIR/draft-kv-c16-cap192-k15-compiled-v1"

PYTHONDONTWRITEBYTECODE=1 python3 "$REPO_DIR/targets/cpro/scripts/compile_incremental_draft.py" \
  --air-export "$INCREMENTAL_AIR" \
  --output "$INCREMENTAL_COMPILED"
```

预期`DRAFT_INCREMENTAL_OMC_COMPILED_ABI_BOUND`，包含17输入、14输出。编译后检查FP32计算路径及KV形状／类型，再生成新`model/model.omc`。每30秒显示编译等待状态，完整日志保留在`compilation/compile.log`。失败时保留目录，不覆盖重试。

## 查看并回传结果

这一步不运行板端，也不表示新KV版本已接通。拿到实际编译接口后接入12路缓存，再跑完整板端文字→Prefill→Draft／decode1串行验证→文字闭环。整体保通暂不按精度或接受率验收；Draft独立精度另测。当前新编译目录不能直接传给旧流程的`--draft-bundle`。

```bash
python3 - <<'PYCODE'
import json, os
from pathlib import Path
p = Path(os.environ["INCREMENTAL_COMPILED"]) / "manifest.json"
d = json.loads(p.read_text())
keys = ("status", "stage", "context_rows", "cache_capacity", "max_draft_tokens",
        "model_compilation", "compiled_precision", "compiled_abi",
        "device_execution", "runtime_integration", "error")
print(json.dumps({k: d.get(k) for k in keys}, ensure_ascii=False, indent=2))
print("manifest:", p)
PYCODE
```
