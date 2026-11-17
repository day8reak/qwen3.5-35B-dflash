# Target Prefill：分别验证 Linear Attention 与 Full Attention

用同一次真实 Prefill 的模块输入和旧缓存，分别运行 L0–L2 完整 GDN 与 L3 完整 FullAttention；每张图独立重放，定位模块自身的误差，不包含外部残差、MoE 或跨层误差反馈。

先按[Prefill 输入与参考生成](target-prefill-fixture.md)完成环境配置和 `--capture-boundaries` 采集，复用其中的 `REPO_DIR`、`AI_RUN_DIR`、`TARGET_CONFIG`、`PREFILL_FIXTURE`。以下流程要求 `capture/manifest.json` 为 `CAPTURED`，同次 `attention_audit` 为 `PASSED`。保留完整 `PREFILL_FIXTURE`（含父 manifest 和 `attention-audit/`），无需再次执行整网 Prefill。

## 1. 导出两组 AIR

切到此前成功导出 CPro AIR 的 Python 环境。`CPRO_SOURCE` 指向原私有 CPro 源码根目录；GDN 导出器在进程内将原 `linear_attention_convstate_update` 展开为已验证的 concat/tail，因此应使用保留该原调用的源码副本。不要指向已手工应用整网 ConvState patch 的副本。权重 YAML 使用原 checkpoint／量化权重，路径规则见采集文档。

```bash
export CPRO_SOURCE="${CPRO_SOURCE:-$TARGET_SOURCE}"
export CPRO_EXPORT_SCRIPT="$CPRO_SOURCE/export_glm_model_qwen3_5_moe.py"
export PAIR_ROOT="$AI_RUN_DIR/target/tests/attention-pair"
export GOLDEN_CAPTURE="$PREFILL_FIXTURE/capture"
export PYTHONDONTWRITEBYTECODE=1

python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" export \
  --capture "$GOLDEN_CAPTURE" \
  --source-root "$CPRO_SOURCE" --target-config "$TARGET_CONFIG" \
  --converter-script "$CPRO_EXPORT_SCRIPT" \
  --points gdn_layer --phases prefill \
  --qlinear-scales fp16 --gdn-state-dtype float16 --gdn-convstate reference \
  --output "$PAIR_ROOT/linear/air"

python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" export \
  --capture "$GOLDEN_CAPTURE" \
  --source-root "$CPRO_SOURCE" --target-config "$TARGET_CONFIG" \
  --converter-script "$CPRO_EXPORT_SCRIPT" \
  --points full_attention_layer --phases prefill \
  --qlinear-scales fp16 --rope-precision fp32 \
  --full-attention-reference captured-audited \
  --output "$PAIR_ROOT/full/air"
```

预期 Linear 为 3 张图、每图 hidden／ConvState／recurrent state；Full 为 1 张图，内部串联 QKV、Q/K Norm、RoPE、CacheUpdate、Attention、gate、o_proj，输出 hidden／K cache／V cache。Full 使用已验证的 FP16 QLinear scale 和 FP32 RoPE 旋转，并保留原输出 dtype。

这里显式选用 `captured-audited`：Full 的主参考是**同次原 NZ 模块的 hidden／K／V**。导出前会核验原文件、CacheUpdate→core→模块返回的 K/V 链路，并重新计算独立 FP64 Attention core 参考、转 FP16 后按原精度门槛比较；失败则停止。该审计只验证核心，同次模块输出仍是原实现参考，不代表全模块独立数学验证或整网精度通过。

若辅助模块重放因 K/V 不一致而显示 `UNAVAILABLE`，上述入口仍可使用通过审计的同次捕获，失败记录不会被改写。CLI 默认仍是 `math-replay`，要求辅助重放 `COMPUTED` 且 K/V 逐位一致；两个模式不自动切换。

## 2. 编译

在原 DDK 编译环境执行，沿用已成功的 TBE 兼容诊断目录；两组均要求新输出目录。

```bash
: "${CPRO_DDK_ROOT:?沿用已验证的 DDK 根目录}"
export CPRO_DDK_ID="${CPRO_DDK_ID:-100.520.020.010}"
export TBE_DIAGNOSIS="${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}"
(
set -e
for group in linear full; do
  python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" compile \
    --batch "$PAIR_ROOT/$group/air" --output "$PAIR_ROOT/$group/omc" \
    --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
    --compiler-profile isolated-kirinx90 \
    --tbe-diagnosis "$TBE_DIAGNOSIS" --use-origin-format
done
)
```

## 3. 上板、回收、比较

沿用[板端执行环境](target-batch-rerun.md#4-上板与回收)的 HDC、`BIP`、`SN` 和 `runtime.json`。跨主机复制整个 `PAIR_ROOT`，保留各组 `air/omc` 的相对位置。分别令 `GROUP=linear`、`GROUP=full` 执行；前一组回收完成后再提交下一组。

```bash
export GROUP=linear
export RUNTIME_CONFIG="${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}"
export BATCH_ROOT="$PAIR_ROOT/$GROUP"

python "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --compiled "$BATCH_ROOT/omc" --config "$RUNTIME_CONFIG" \
  --ddk-root "$CPRO_DDK_ROOT" --output "$BATCH_ROOT/execution" --apply

python "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --recover "$BATCH_ROOT/execution" --output "$BATCH_ROOT/recovery"

python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" compare \
  --fixture "$BATCH_ROOT/execution/fixture" --native "$BATCH_ROOT/recovery/native" \
  --output "$BATCH_ROOT/comparison"

cat "$BATCH_ROOT/comparison/summary.csv"
```

`PENDING` 时仅重复 recover；已有 native 输出后，即使精度失败也继续 compare。每图预热 3 次、采样 10 次，每次恢复同一输入。Linear 应有 9 行，Full 应有 3 行。Full CSV 的 `reference_basis` 标明参考来源；上述模式中常规指标与 `captured_*` 都对比同次原输出。K/V 另外要求本次有效写入区域满足相同精度门槛，未写区域逐位等于旧输入。

回传两组 `summary.csv` 和 `report.json`。只有两组各自完成执行和比较后，才能判断是 GDN、FullAttention 或两者存在偏差；模块通过也不代表 MoE、整网 logits 或连续状态反馈已经通过。
