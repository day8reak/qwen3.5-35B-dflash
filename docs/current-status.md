# 当前阶段：将 StableArgMax 接入完整 Draft

2026-11-05：单算子两种shape共11组、110次输出全部INT32精确一致、重复稳定。Qwen配置同步API均值1.232ms；17×8193尾部配置0.515ms。两轮均为单NPU子图，详情见[结果](results.md)。单算子验证已收敛。

本阶段在现有C16／cap192／K15、host embedding候选上选择 `StableArgMax`。权重、attention、hidden和KV计算保持原表达；使用有限FP16 logits，并列取最小索引。新的Torch导出接线已通过主机检查，完整AIR／OMC及板端结果待验证。产物放固定候选目录，现有运行模型继续保留。

## 用户现在执行什么

先将两台服务器仓库更新到最新版本。以下 `REPO_DIR`、`AI_RUN_DIR` 及SDK路径沿用各自项目配置；`draft/source/prepared` 为已准备的FP16权重。无需重建allops或重复单算子测试。

**权重服务器（已有TorchAir环境）**：导出一次完整Draft。

```bash
source "$REPO_DIR/targets/cpro/scripts/env.sh"

python3 "$REPO_DIR/targets/cpro/scripts/export_incremental_draft.py" \
  --prepared "$AI_RUN_DIR/draft/source/prepared" --frontend pytorch \
  --optimization packed-spatial-host-embedding --top1 stable-arg-max \
  --context-rows 16 --cache-capacity 192 --max-draft-tokens 15 \
  --output "$AI_RUN_DIR/draft/tests/top1-integration/export"
```

需要返回 `PYTORCH_DRAFT_AIR_EXPORTED`，17输入／14输出。导出检查要求一个 `StableArgMax` 直接连接token输出、无原生ArgMax；缺失的中间shape交给OMG推导，明确冲突会拒绝。将整个 `draft/tests/top1-integration/export` 文件夹（含 `weight/` 和元数据）复制到DDK服务器对应位置。

**DDK服务器（无需TorchAir）**：复用已安装的StableArgMax及TBE诊断，编译候选。

```bash
source "$REPO_DIR/targets/cpro/scripts/env.sh"

python3 "$REPO_DIR/targets/cpro/scripts/compile_incremental_draft.py" \
  --air-export "$AI_RUN_DIR/draft/tests/top1-integration/export" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --tbe-diagnosis "$AI_RUN_DIR/common/tbe-diagnosis" \
  --output "$AI_RUN_DIR/draft/tests/top1-integration/compiled"
```

需要 `PYTORCH_DRAFT_OMC_COMPILED`，且 `top1_placement.status=NPU_OUTPUT_BOUNDARY_PASSED`。该检查确认token的INT32输出直接来自NPU任务；它不证明整图所有算子均在NPU或内部指令精度。

**编译通过后，同一DDK／HDC服务器**：运行完整Draft数值对照，沿用原容差。配置须提供此前成功的设备连接及运行器，默认公共 `runtime/runtime.json`。

```bash
export VECTOR_CONFIG="${VECTOR_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}"

python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --compiled "$AI_RUN_DIR/draft/tests/top1-integration/compiled" \
  --air-export "$AI_RUN_DIR/draft/tests/top1-integration/export" \
  --source "$AI_RUN_DIR/draft/source/prepared" \
  --ddk-root "$CPRO_DDK_ROOT" --config "$VECTOR_CONFIG" \
  --output "$AI_RUN_DIR/draft/tests/top1-integration/run" --apply

python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --recover "$AI_RUN_DIR/draft/tests/top1-integration/run" \
  --output "$AI_RUN_DIR/draft/tests/top1-integration/result"
```

出现PENDING时继续回收同一请求。验证包含独立输入与连续KV反馈，golden仍由原模型产生；检查Top1、hidden和每层KV。旧完整Draft已有数值失败，单算子通过不会消除这些偏差，本轮也不放宽门槛。回传导出、编译和最终验证摘要后，再判断整图落点、精度与实测延迟。
