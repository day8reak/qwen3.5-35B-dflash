# 当前阶段测试命令

增量Draft的C16／缓存192／K15 AIR已在DDK主机导出成功：17输入、14输出、六层12份KV缓存，不复制原权重。当前执行下一步OMG编译、接口检查和内嵌权重打包。Target精度继续独立调试，新批量verify暂不开发。

## 采集当前S64 Draft算子耗时

复用`real-feature-draft-plan`的真实Prefill64特征和现有内嵌权重Draft；不重编模型或runner、不重新上传模型。启动前会核对旧输入和元数据的完整SHA256，在新输出目录内重新生成输入计划；允许内容相同的文件身份变化，不改旧报告。板端模型缓存须仍在。通过已观察到的`model_run_tool_internal --enable_item=1`，用V1执行一次，再由同目录`data_proc_tool_internal`转CSV。可与主机上的增量OMG编译并行；板上模型调用使用同一把排他锁。

```bash
git -C "$REPO_DIR" pull --ff-only

python3 "$REPO_DIR/targets/cpro/scripts/profile_draft.py" \
  --prepared "$AI_RUN_DIR/real-feature-draft-plan" \
  --tool-inspection "$AI_RUN_DIR/target-board-tool-inspection-writable" \
  --output "$AI_RUN_DIR/draft-s64-profile-v2" \
  --apply
```

返回`DRAFT_OPERATOR_PROFILE_STARTED`后，稍后回收同一次采集。断线也用下面命令，不重跑模型；若返回`PENDING`，等待后换一个新的回收目录再执行。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/profile_draft.py" \
  --recover "$AI_RUN_DIR/draft-s64-profile-v2" \
  --output "$AI_RUN_DIR/draft-s64-profile-recovery-v2"

cat "$AI_RUN_DIR/draft-s64-profile-recovery-v2/summary/report.md"
```

回收目录保留原始`prof_data`、工具日志、PID和退出码；`summary/operators.csv`包含按耗时排序的全部算子，`operator-types.csv`和`devices.csv`分别按类型、原始设备标签汇总。首次采集需核对厂商导出的CSV是否混入历史模型记录，因此状态标为`DRAFT_PROFILE_CAPTURED_REQUIRES_ATTRIBUTION_REVIEW`，不能把所有导出行自动当成Draft热点。

这是新工具进程中的一次profiling诊断，输入是已有64行真实特征，不是18-token文字请求中的同一次Draft调用；也不是稳定时延分布或精度验收。算子耗时总和可能含重叠、嵌套执行；不能据此直接推算50ms。工具使用方法参考[官方示例](https://developer.huawei.com/consumer/jp/doc/hiai-Guides/examples-0000001053605660)，实际参数以本机internal工具帮助为准。

## 编译已经导出的K15

复用本次成功的`air-v1`目录，无需重新导出。脚本从原`full-draft/residual-fix`记录选择相同DDK，在新目录编译；原权重仅建立逐文件引用并检查前后完整性。至少预留两份Draft权重大小加1 GiB空间，供新编译权重、打包OMC与临时文件使用；编译器实际峰值可能更高。不要同时编译其他档位。

```bash
git -C "$REPO_DIR" pull --ff-only
export MAX_DRAFT_TOKENS=15
export INCREMENTAL_AIR="$AI_RUN_DIR/draft-kv-c16-cap192-k15-air-v1"
export INCREMENTAL_COMPILED="$AI_RUN_DIR/draft-kv-c16-cap192-k15-compiled-v2"

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
