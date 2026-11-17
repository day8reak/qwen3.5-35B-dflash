# Draft 完整精度验证与 Profiling

复用已编译的六层 Draft，先对照 CPU 参考验证候选、hidden 和逐层 KV，再采集同一模型的算子耗时。

## 前提

先完成[环境配置](environment-setup.md)中的工具链与项目配置，确保 HDC 可连接板端；以下命令在服务器仓库根目录执行。

- 已有 `draft/source/prepared`、`draft/air`、`draft/omc`；缺少时按[完整模型执行](run-end-to-end.md)第 2 节准备。
- 已有 `runtime/runtime.json` 及其引用的 assets 配置、`runtime/native/build/qwen35_draft_accuracy` 运行器；沿用此前运行配置，运行器构建见完整流程第 4 节。
- 板端已有 `/data/local/tmp/tools_sysdbg/model_run_tool_internal` 和 `data_proc_tool_internal`。

## 1. 加载配置

```bash
export REPO_DIR="$(git rev-parse --show-toplevel)"
source "$REPO_DIR/targets/cpro/scripts/env.sh"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

export DRAFT_RUNTIME_CONFIG="${DRAFT_RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}"
export DRAFT_ACCURACY_RUN="$AI_RUN_DIR/draft/tests/accuracy-run"
export DRAFT_ACCURACY="$AI_RUN_DIR/draft/tests/accuracy-result"
export PROFILE_RUN="$AI_RUN_DIR/draft/tests/profile-run"
export PROFILE_RESULT="$AI_RUN_DIR/draft/tests/profile-result"
```

## 2. 完整 Draft 精度验证

C16／cap192／K15 配置运行 19 个用例的独立输入对照与连续 KV 反馈，覆盖空缓存、部分新增、重复调用、候选拒绝和容量边界；输入为确定性合成特征，不代表 Target 端到端或文本质量验证。

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --compiled "$AI_RUN_DIR/draft/omc" \
  --air-export "$AI_RUN_DIR/draft/air" \
  --source "$AI_RUN_DIR/draft/source/prepared" \
  --config "$DRAFT_RUNTIME_CONFIG" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$DRAFT_ACCURACY_RUN" --apply
```

启动后回收结果；若为 `DRAFT_ACCURACY_PENDING`，稍后只重复下面的回收命令。

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --recover "$DRAFT_ACCURACY_RUN" --output "$DRAFT_ACCURACY"
```

结果在 `$DRAFT_ACCURACY/result.json`：`PASSED` 表示通过，`NUMERICAL_FAILED` 表示已执行并回收但数值未通过。已有同一 OMC 的完整结果、原 fixture 和板端模型缓存时，可跳过本节。

## 3. 采集 Profiling

沿用 case 18：满 192 行缓存、零新增 context、15 个候选；工具自动绑定输入文件及输入／输出类型，通过 V1 执行一次，无预热。

```bash
python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --accuracy "$DRAFT_ACCURACY" --case-index 18 \
  --allow-numerical-failure --modelmanager-itf V1 \
  --model-tool /data/local/tmp/tools_sysdbg/model_run_tool_internal \
  --data-proc-tool /data/local/tmp/tools_sysdbg/data_proc_tool_internal \
  --output "$PROFILE_RUN" --apply
```

当前基线数值尚未通过，`--allow-numerical-failure` 仅允许性能诊断，保留原精度失败结论。Profiling 复用精度验证部署的板端模型缓存，不会重新编译或自动上传 OMC。

## 4. 回收 CSV

```bash
python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --recover "$PROFILE_RUN" --output "$PROFILE_RESULT"
```

若为 `DRAFT_OPERATOR_PROFILE_PENDING`，只重复回收；完成状态为 `DRAFT_PROFILE_CAPTURED_REQUIRES_ATTRIBUTION_REVIEW`。

原始 `*_op.csv`、`*_model.csv` 位于 `$PROFILE_RESULT/prof_data/`，汇总在 `result.json`。按本次模型／进程筛选记录，分别查看加载与推理耗时；单次 Profiling 不代表常驻时延，也不要把可能重叠的算子时间相加当总时延。
