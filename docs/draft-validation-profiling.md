# Draft 完整精度验证与 Profiling

复用已编译的六层 Draft，先对照 CPU 参考验证候选、hidden 和逐层 KV，再采集同一模型的算子耗时。

## 前提

先完成[环境配置](environment-setup.md)中的工具链与项目配置，确保 HDC 可连接板端；以下命令在服务器仓库根目录执行。

- 精度验证使用 `draft/source/prepared`、`draft/air`、`draft/omc`。三者齐全时可跳过 TBE 诊断和编译。首次编译需在 DDK 服务器安装 StableArgMax。[权重与 AIR 生成步骤](run-end-to-end.md#draft-build) · [StableArgMax 安装步骤](../targets/cpro/ops/draft-top1/README.md)
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
export UT_RUN="$AI_RUN_DIR/draft/tests/accuracy-untouched-run"
export UT_RESULT="$AI_RUN_DIR/draft/tests/accuracy-untouched-result"
export PROFILE_RUN="$AI_RUN_DIR/draft/tests/profile-run"
export PROFILE_RESULT="$AI_RUN_DIR/draft/tests/profile-result"
```

<a id="tbe-diagnosis"></a>

### 首次编译 Draft OMC 时生成 TBE 诊断

在 **DDK 服务器**的 `custom_pypto` 环境加载上面的配置。已有当前 DDK 的完整诊断时，将 `TBE_DIAGNOSIS` 指向其原目录即可复用。

```bash
export TBE_DIAGNOSIS="${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}"
if [ ! -d "$TBE_DIAGNOSIS" ]; then
  python3 "$REPO_DIR/targets/cpro/scripts/direct_model_air.py" check-tbe-platform \
    --ddk-root "$CPRO_DDK_ROOT" --python "$(command -v python3)" \
    --diagnose --output "$TBE_DIAGNOSIS"
fi
```

`$TBE_DIAGNOSIS/report.json` 的 `DIAGNOSIS_COMPLETE` 表示诊断已结束。编译 packed Draft 时，把整个诊断目录传给 `--tbe-diagnosis`。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/compile_incremental_draft.py" \
  --air-export "$AI_RUN_DIR/draft/air" --source "$AI_RUN_DIR/draft/source/prepared" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --tbe-diagnosis "$TBE_DIAGNOSIS" --rope-fp32-table --cpu-feature-pack \
  --output "$AI_RUN_DIR/draft/omc"
```

<a id="draft-accuracy"></a>

## 2. 完整 Draft 精度验证

C16／cap192／K15 配置用固定的合成特征运行 19 个用例，比较独立输入和连续 KV 反馈两种模式。已有同一 OMC 的完整精度结果及原始输入、golden 时，直接从 2.2 开始。

### 2.1 首次生成并回收基线

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --compiled "$AI_RUN_DIR/draft/omc" \
  --air-export "$AI_RUN_DIR/draft/air" \
  --source "$AI_RUN_DIR/draft/source/prepared" \
  --config "$DRAFT_RUNTIME_CONFIG" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$DRAFT_ACCURACY_RUN" --apply
```

启动后回收结果。若为 `DRAFT_ACCURACY_PENDING`，稍后重复对应的回收命令。

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --recover "$DRAFT_ACCURACY_RUN" --output "$DRAFT_ACCURACY"
```

`$DRAFT_ACCURACY/result.json` 状态为 `PASSED` 或 `NUMERICAL_FAILED` 时均可重放。保留该结果及其引用的输入和 golden。

### 2.2 使用 untouched 重放精度验证

复用同一 OMC、输入和 golden，让 runner 跳过对输出缓冲区的显式清零。两次独立冷启动实测中，第 0 例首次返回的 14 个输出均与老 OMC 基准逐位一致。

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --config "$DRAFT_RUNTIME_CONFIG" --replay "$DRAFT_ACCURACY" \
  --output-initialization untouched --output "$UT_RUN" --apply
```

重放启动后，从 `$UT_RUN` 回收结果。

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --recover "$UT_RUN" --output "$UT_RESULT"
```

精度结果见 `$UT_RESULT/result.json`。当前对 CPU golden 的整套验证仍为 `NUMERICAL_FAILED`，两种模式各有 83/86 个 Top1 匹配。

## 3. 采集 Profiling

沿用 case 18：满 192 行缓存、零新增 context、15 个候选；工具自动绑定输入文件及输入／输出类型，通过 V1 执行一次，无预热。

```bash
python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --accuracy "$UT_RESULT" --case-index 18 \
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
