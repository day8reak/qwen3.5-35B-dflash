# Target Prefill：分别验证 Linear Attention 与 Full Attention

用同一次真实 Prefill 的模块输入和旧缓存，分别运行 L0–L2 完整 GDN 与 L3 完整 FullAttention；每张图独立重放，定位模块自身的误差，不包含外部残差、MoE 或跨层误差反馈。

T64 Prefill 的四个已测投影均通过 CPU 对照。CPro L3 Full Attention 的 hidden、K、V 对数学重放均通过。L0–L2 Linear Attention 对同一份捕获，分块 GDR 通过 6/9 项，逐 token GDR 通过 8/9 项，剩余超差项为 L1 hidden。[CPU 与 310P 对照](target-golden-verification.md#t64-golden) · [T64 GDR 对照](#t64-gdr)

先按[Prefill 输入与参考生成](target-prefill-fixture.md)完成环境配置和 `--capture-boundaries` 采集，复用其中的 `REPO_DIR`、`AI_RUN_DIR`、`TARGET_CONFIG`、`PREFILL_FIXTURE`。保留完整 fixture，确认 `capture/manifest.json` 的状态为 `CAPTURED`。Full 的 `captured-audited` 模式还要求同次 core 审计为 `PASSED`。本次 T64 使用[模块数学重放](#t64-full)。

## 1. 导出两组 AIR

切到此前成功导出 CPro AIR 的 Python 环境。`CPRO_SOURCE` 指向原私有 CPro 源码根目录；GDN 导出器在进程内将原 `linear_attention_convstate_update` 展开为已验证的 concat/tail，因此应使用保留该原调用的源码副本。不要指向已手工应用整网 ConvState patch 的副本。权重 YAML 使用原 checkpoint／量化权重，路径规则见采集文档。

```bash
conda activate qwen3_5_cpro
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

Full 导出会在主机将 `new_kv_cache_pos` 从捕获的整数转为精确 FP32 输入，形状仍为 `[T,1]`。这是原 `update()` 第一条转换的前移，用于绕过本轮 `Cast_27` 的 INT64→FP32 编译失败；原浮点索引公式、位置值及 golden 不变，`allQLen` 仍为 INT64。导出器检查源码使用位置、数值回转、实际 AIR 输入 dtype，并在 `export.json` 的 `source_contract.cache_position_carrier` 保存两份描述。T8 时该输入由 64 字节变为 32 字节，输入数量不变；必须重新导出、编译并生成执行 fixture，不能复用旧输入文件或槽位。T8 批次的 hidden／K／V 对原 310P 捕获均超差，K/V 未写区逐位保留。

这里显式选用 `captured-audited`：Full 的主参考是**同次原 NZ 模块的 hidden／K／V**。导出前会核验原文件、CacheUpdate→core→模块返回的 K/V 链路，并重新计算独立 FP64 Attention core 参考、转 FP16 后按原精度门槛比较；失败则停止。该审计只验证核心，同次模块输出仍是原实现参考，不代表全模块独立数学验证或整网精度通过。

Mask 核验保留模块入口和核心入口的实际数据。零符号或因果屏蔽值 `-inf → -65504` 的表示差异，还要求同一组 Q/K/V 下两份独立参考的最终 FP16 输出逐位一致；屏蔽位置改变或其它内容差异仍拒绝，并输出 `mask_diagnostics`。原输入和 golden 不改写。

若辅助模块重放因 K/V 不一致而显示 `UNAVAILABLE`，上述入口仍可使用通过审计的同次捕获，失败记录不会被改写。CLI 默认仍是 `math-replay`，要求辅助重放 `COMPUTED` 且 K/V 逐位一致；两个模式不自动切换。

<a id="t64-full"></a>

### T64 Full Attention 使用模块数学重放

在此前成功导出 AIR 的 `qwen3_5_cpro` 环境复用 T64 边界和第 1 节的 CPro 源码、converter。主参考取自同一模块输入上的重放，只将 Attention core 换成 FP64 数学计算后转 FP16，投影、gate 等其余计算仍使用原模块。导出时会再次核对输入和 K/V 的文件记录。

板端本次三项通过且重复稳定。hidden 对数学重放的相对 L2 为 **1.6221%**，对同次 310P 原模块输出为 **99.6910%**。原模块和重放的 K/V 逐位相同，板端 K/V 有效写入区的相对 L2 分别为 **0.2449% / 0.2245%**，未写区均逐位保留。[三项板端比较结果](../specs/target-full-attention-t64-metrics.csv)

```bash
conda activate qwen3_5_cpro
T64_FIXTURE="$AI_RUN_DIR/target/tests/prefill-fixture-t64"
export PAIR_ROOT="$AI_RUN_DIR/target/tests/attention-pair-t64"
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" export \
  --capture "$T64_FIXTURE/capture" --points full_attention_layer --phases prefill \
  --source-root "$CPRO_SOURCE" --target-config "$TARGET_CONFIG" \
  --converter-script "$CPRO_EXPORT_SCRIPT" --qlinear-scales fp16 --rope-precision fp32 \
  --full-attention-reference math-replay --output "$PAIR_ROOT/full/air"
```

按第 2 节只编译 `full`，第 3 节使用 `GROUP=full`。比较报告的 `relative_l2` 和 `cosine` 对应板端对数学 core 的模块重放，`captured_relative_l2` 和 `captured_cosine` 对应板端对原模块输出。Linear 两组沿用下方[分块与逐 token 对照](#t64-gdr)。

若导出在 `npu_quant_matmul` 的 FakeTensor 检查时报 A8W4 的 `scale` 要求 INT64，先核对 traceback 中的 Python 环境。已复现的失败使用了 310P 采集环境 `qwen3_5_w4`。切回此前成功的 CPro AIR Python、CANN 和 TorchAir 配置，保留 `--qlinear-scales fp16` 与原 fixture。重试可将 `PAIR_ROOT` 设为 `$AI_RUN_DIR/target/tests/attention-pair-t64-cpro`，后续编译和上板沿用该目录。

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

`PENDING` 时仅重复 recover；已有 native 输出后，即使精度失败也继续 compare。每图预热 3 次、采样 10 次，每次恢复同一输入。Linear 应有 9 行，Full 应有 3 行。Full CSV 的 `reference_basis` 标明参考来源；`captured-audited` 的两组指标均与原捕获比较，`math-replay` 的常规指标与重放比较。K/V 另外要求本次有效写入区域满足相同精度门槛，未写区域逐位等于旧输入。

回传两组 `summary.csv` 和 `report.json`。只有两组各自完成执行和比较后，才能判断是 GDN、FullAttention 或两者存在偏差；模块通过也不代表 MoE、整网 logits 或连续状态反馈已经通过。

<a id="t64-gdr"></a>

## T64 Prefill 的 GDR 分块与逐 token 对照

两组均使用同一份 L0–L2 捕获。分块组按原路径运行一次 GDR，逐 token 组在每层的同一张静态图内运行 64 次 S=1、chunk_size=1 的 GDR。每步消费对应 token 的 Q/K/V、g、beta 和上一步的 recurrent state，有效长度为 1。按时间顺序拼接 core 后执行原有 gated norm 和输出投影。两组都使用 FP16 状态接口，逐步回传 FP16 状态会引入与分块计算不同的舍入。

T64 的三项输出分别是当前序列的 `hidden`、下一次卷积使用的 QKV 历史、以及第 64 个 token 更新后的递归状态。原模块返回 `(output, past_key_values)`，整层探针将后者展开为两项。hidden 的计算路径为输入投影、带历史的因果深度卷积与 SiLU、GDR 逐位置读出、按 head 的 RMSNorm 与 SiLU(z) 门控、合并各 head、最后执行 `out_proj`。decoder 的残差加法在本探针之外。

| 层 | 分块 hidden 相对 L2 | 逐 token hidden 相对 L2 | 逐 token recurrent state 相对 L2 |
| --- | ---: | ---: | ---: |
| L0 | 26.96% · 失败 | 2.66% · 通过 | 0.96% · 通过 |
| L1 | 48.66% · 失败 | 21.08% · 失败 | 1.38% · 通过 |
| L2 | 53.70% · 失败 | 4.80% · 通过 | 0.94% · 通过 |

两组的 ConvState 均为 3/3 通过。逐 token 的三个末态相对 L2 均略有上升，板端输出为 FP16，捕获的 raw state 参考为 FP32。各层单独重放本层捕获输入。L1 hidden 的 cosine 为 0.98031，平均相对误差为 1.05456，两者和相对 L2 均超限。要区分误差从何处进入 hidden，需要在同一 T64 条件下比较 L1 的逐位置 core、gated norm 输出和输出投影。数值来自板端回传的两份 summary.csv。[分块与逐 token 的 18 项结果](../specs/target-gdn-t64-metrics.csv)

**310P 采集。** 已有同次 T64 Prefill 的完整边界捕获时，直接将 `T64_FIXTURE` 指向它。当前 YAML 的单条 prompt 为 8 个 token，下面示例将原文重复 8 次并校验实际分词数为 64。已有 T8 golden 可单独用于 8 步对照。

```bash
T64_FIXTURE="$AI_RUN_DIR/target/tests/prefill-fixture-t64"
python "$REPO_DIR/targets/cpro/tools/generate_target_prefill.py" \
  --source-root "$TARGET_SOURCE" --config "$TARGET_CONFIG" --device npu:0 \
  --prompt-repeat 8 --sequence-length 64 --capture-boundaries \
  --output "$T64_FIXTURE"
```

**AIR 导出。** 在 AIR 服务器配置上述 `T64_FIXTURE`，确认其 `capture/manifest.json` 为 `CAPTURED`。使用第 1 节的 CPro 源目录和 converter。在后续各主机沿用同一 `PAIR_ROOT` 相对目录结构。

```bash
conda activate qwen3_5_cpro
export PAIR_ROOT="$AI_RUN_DIR/target/tests/gdr-t64"
for mode in chunk single-token; do
  python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" export \
    --capture "$T64_FIXTURE/capture" --source-root "$CPRO_SOURCE" \
    --target-config "$TARGET_CONFIG" --converter-script "$CPRO_EXPORT_SCRIPT" \
    --points gdn_layer --phases prefill --qlinear-scales fp16 \
    --gdn-state-dtype float16 --gdn-convstate reference --gdr-mode "$mode" \
    --output "$PAIR_ROOT/$mode/air" || break
done
```

核对两组 AIR manifest 的 `capture_sha256` 相同。每份 `export.json` 的 `candidate_graph_check` 应分别显示 `gdr_calls=1, chunk_size=64` 和 `gdr_calls=64, chunk_size=1`。逐 token 组的 `step_length_const_edges` 记录供 64 步使用的长度常量，导出时校验其 INT16 值为 1。若已有旧版 AIR 在 `OnesLike` 推形处失败，仅需重新导出和编译逐 token 组，使用新的输出路径。沿用第 2 节的 DDK 编译命令，将循环组名改为 `chunk single-token`。执行第 3 节时分别设置 `GROUP=chunk` 和 `GROUP=single-token`，各自回收 3 层、9 项输出。

逐层对比两组板端输出。先确认重复稳定的 `golden_new_conv_state` 对应实际输出哈希在 A/B 间一致，再看 hidden 和 recurrent state 的变化，以及各自对同一捕获的误差。两组都偏离 310P 时，还需结合 CPU 与 310P 的投影对照。仅切换 GDR 无法证明其它模块全部正确。

## 四层 T64 Prefill 导出

在 AIR 服务器使用同一次 T64 捕获和本地 CPro 源码。L0–L2 各执行 64 次单 token GDR，这三层的输入投影、卷积和输出投影仍一次处理 64 个 token，L3 使用 full attention。四层之后取最后一个位置的 logits，同时输出八个状态张量。

```bash
conda activate qwen3_5_cpro
T64_FIXTURE="$AI_RUN_DIR/target/tests/prefill-fixture-t64"
JOB="$AI_RUN_DIR/target/tests/cpro-prefill-t64"
python "$REPO_DIR/targets/cpro/tools/prepare_cpro_prefill_t64.py" \
  --source-root "$CPRO_SOURCE" --config "$TARGET_CONFIG" \
  --fixture "$T64_FIXTURE" --output "$JOB" --export
```

导出文件为 `$JOB/air/export.air`。在 310P 上使用同一份 fixture 生成四层参考输出。脚本核对原推理的 L3 attention 输入，再将独立 attention 数学结果接回原模块并运行到 LM head。

```bash
conda activate qwen3_5_w4
python "$JOB/reference_t64.py" --repo "$REPO_DIR" \
  --source-root "$TARGET_SOURCE" --config "$TARGET_CONFIG" \
  --fixture "$T64_FIXTURE" --device npu:0 --output "$JOB/reference"
```
