# Target Prefill：分别验证 Linear Attention 与 Full Attention

## L0 完整 Linear Attention 的 CPU SiLU 对照

78 上运行以下脚本。它从原始四个 attention 的 AIR 批次提取完整 L0、权重、实测输入和 310P 的 `golden_hidden`。可用 `ATTENTION_AIR` 指定已有批次的 `air` 目录。默认批次已清理时，脚本沿用 `TARGET_SOURCE`、`CPRO_SOURCE` 及原有 conda 环境重新采集和导出。

```bash
cd "${REPO_DIR:?}"
git pull --ff-only origin main
bash targets/cpro/scripts/l0_attention_cpu_silu_78.sh
```

两台主机都会打印 `L0 AIR constants`。其中 `inline tensors` 统计 AIR 内含张量数据的常量，`external files` 统计图引用的外置文件。权重全部内嵌时，外置文件数为 0。两端打印的 AIR SHA-256 应相同。

53 上用已有 DDK、GDR 和 HDC 环境运行以下脚本。脚本从 78 拉取该批次，编译原版和 CPU SiLU 版各一张完整的 L0 OMC，调用板上的 `model_run_tool_internal` 各执行一次。实验版替换卷积后的 SiLU 和门控 SiLU。编译后会检查两处 `sigmoid(x) * x` 的 CPUCL 归属和 FP32 dtype。

```bash
cd "${REPO_DIR:?}"
git pull --ff-only origin main
bash targets/cpro/scripts/l0_attention_cpu_silu_53.sh
```

运行目录在开头的 `L0 attention A/B` 一行。两组按相同输入和 310P golden 比较 `out_proj` 的前 19 行。终端显示 valid19 和 token18，`comparison.csv` 保存每个有效 token 的五项指标、MAE、最大绝对误差、超差元素数及比例。两个 `delta_*_pp` 列表示实验组减原版的百分点差。逐元素容差为 `rtol=1e-3, atol=1e-5`，允许超差比例为 0.1%。

L0–L2 的 310P 与 CPro 逐张量对比见 [GDR、RMSNorm 和门控乘法](target-gdr-prefill-stages.md)。

用同一次真实 Prefill 的模块输入和旧缓存，分别运行 L0–L2 完整 GDN 与 L3 完整 FullAttention；每张图独立重放，定位模块自身的误差，不包含外部残差、MoE 或跨层误差反馈。

物理 T64、19 个有效 token 的未缩放基线中，L0–L2 的整层 hidden 均超差，L0 已定位到 GDR core 的小数值偏差。[19 个有效 token 的逐层结果与 L0 对照](#t64-valid19-linear)

在 64 个 token 全部有效的历史用例中，四个已测投影均通过 CPU 对照。CPro L3 Full Attention 的 hidden、K、V 对数学重放均通过。L0–L2 Linear Attention 对同一份捕获，分块 GDR 通过 6/9 项，逐 token GDR 通过 8/9 项，剩余超差项为 L1 hidden。[CPU 与 310P 对照](target-golden-verification.md#t64-golden) · [T64 GDR 对照](#t64-gdr)

先按[Prefill 输入与参考生成](target-prefill-fixture.md)完成环境配置和 `--capture-boundaries` 采集，复用其中的 `REPO_DIR`、`AI_RUN_DIR`、`TARGET_CONFIG`、`PREFILL_FIXTURE`。保留完整 fixture，确认 `capture/manifest.json` 的状态为 `CAPTURED`。Full 的 `captured-audited` 模式还要求同次 core 审计为 `PASSED`。本次 T64 使用[模块数学重放](#t64-full)。

## 用纯 T-matrix 版 GDR 复测四层 prefill

L0–L2 分别重放 linear attention，L3 full attention 用作对照。沿用环境中的 `REPO_DIR`、`AI_RUN_DIR`，本轮目录为 `target/tests/attention4-t64-valid19-tmatrix-01`。脚本优先使用上次 Q 缩放后移的四图 AIR。若 OMC 机保留了该 AIR，可直接执行板端步骤。

AIR 机（10.136.180.78）缺少原 AIR 时，用原来的 `TARGET_SOURCE`、`CPRO_SOURCE`、`TARGET_CONFIG` 从同一提示词重新采集 310P、进行 CPU 检查并导出。复用和新导出的四图批次均会校验文件哈希并准备传输权限。

```bash
conda activate qwen3_5_w4
git -C "$REPO_DIR" pull --ff-only origin main &&
bash "$REPO_DIR/targets/cpro/scripts/check_attention4_tmatrix.sh" capture &&
conda activate qwen3_5_cpro &&
bash "$REPO_DIR/targets/cpro/scripts/check_attention4_tmatrix.sh" air
```

OMC 机（10.136.181.35）将 `GDR_SOURCE` 指向改好的 `allops/op_kernel/chunk_gated_delta_rule.cpp`。板端步骤按需调用 `pull_artifacts.py --from air`，检查本地源码与两份 DDK 源码一致，再重新编译并执行。

```bash
conda activate custom_pypto
git -C "$REPO_DIR" pull --ff-only origin main &&
bash "$REPO_DIR/targets/cpro/scripts/check_attention4_tmatrix.sh" board
```

输出 `tmatrix-accuracy.csv`。每个稳定层有 valid19 汇总和 token0–18，共 20 行。上次 Q 后移的 fixture 和回读仍在且输入、golden 匹配时，`baseline_*` 记录历史指标，`delta_*` 为本次减去历史值。逐元素超差比例使用 rtol=1e-3、atol=1e-5。任何层数值失败时，命令以非零状态结束，已计算的 CSV 可直接查看。

<a id="first-decode-valid19"></a>

## 19-token 用例的首步 Decode

在 AIR 机的 310P 环境执行原 `inference_segment.py`。采集器核对 64 行输入与保留的 Prefill 逐 token 一致，第一步 Decode 使用该模型生成的 token、位置 19、`effective_length=[1]` 和 `allQLen=[20]`。报告逐项列出本步真实输入状态与历史 Prefill 输出是否逐位一致。L0–L2 使用各自的模块输入、ConvState 和 recurrent state，L3 使用自己的模块输入和已填充的分页 KV。CPU 在相同输入上独立计算四个模块的投影后 hidden，逐层报告相对 L2 和 cosine。

```bash
conda activate qwen3_5_w4
export DECODE="$AI_RUN_DIR/target/tests/decode-t1-after-valid19"
python "$REPO_DIR/targets/cpro/tools/check_segment_prefill.py" \
  --source-root "$TARGET_SOURCE" --decode-after "$AI_RUN_DIR/target/tests/prefill-4l-t64-valid19" \
  --output "$DECODE/reference"
```

在同一台 AIR 机切换到导出环境。每层有独立 AIR 和 `golden_hidden[1,1,2048]`，参考取自通过固定输入 CPU 检查的 310P 模块。这里使用原 Decode 分支，GDR 单次处理一个 token。导出成功后脚本将 AIR 目录和文件设为 `0777`，并为运行目录内的祖先目录补齐进入权限。

```bash
conda activate qwen3_5_cpro
python "$REPO_DIR/targets/cpro/tools/prepare_cpro_prefill_t64.py" \
  --source-root "$CPRO_SOURCE" --config "$TARGET_CONFIG" --attention-layers \
  --fixture "$DECODE/reference" --output "$DECODE/cpro" --export
```

在 DDK 机的 `custom_pypto` 环境执行传输和编译。

```bash
conda activate custom_pypto
export BATCH_ROOT="$AI_RUN_DIR/target/tests/decode-t1-after-valid19/cpro"
export PROBE="$REPO_DIR/targets/cpro/tools/target_layer_probe.py"
python "$REPO_DIR/targets/cpro/scripts/pull_artifacts.py" "$BATCH_ROOT/air" &&
python "$PROBE" compile --batch "$BATCH_ROOT/air" --output "$BATCH_ROOT/omc" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" --compiler-profile isolated-kirinx90 \
  --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}" --use-origin-format
```

四张图均为 `COMPILED` 后，使用同一个 `BATCH_ROOT` 上板。

```bash
conda activate custom_pypto
VALIDATE="$REPO_DIR/targets/cpro/tools/validate_draft_vector.py"
python "$VALIDATE" --config "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}" \
  --compiled "$BATCH_ROOT/omc" --ddk-root "$CPRO_DDK_ROOT" --output "$BATCH_ROOT/execution" --apply
```

启动后回收同一次执行。即使出现数值超差，成功取回的张量仍可用 `compare` 输出逐层结果。

```bash
conda activate custom_pypto
python "$VALIDATE" --recover "$BATCH_ROOT/execution" --output "$BATCH_ROOT/recovery"
python "$PROBE" compare --fixture "$BATCH_ROOT/execution/fixture" \
  --native "$BATCH_ROOT/recovery/native" --output "$BATCH_ROOT/comparison"
cat "$BATCH_ROOT/comparison/summary.csv"
```

### 首步 Decode 的 L3 CacheUpdate 对照

同一次真实 19-token prefill 后，四层分别使用 310P 模块输入和已有状态运行首步 decode。L0–L2 的 hidden 通过五项指标，相对 L2 分别为 0.02777、0.03046、0.02620，cosine 分别为 0.999616、0.999539、0.999662。L3 hidden 相对 L2 为 757.9691，cosine 为 −0.052076，十次输出稳定。早期独立 Decode CacheUpdate 的 V 样本在应保留区有 130,966 个差异，golden 变化区有 0 个差异。[历史单测记录](target-accuracy-status.md#decode-cache-原因分析)

在 AIR 机的 `qwen3_5_w4` 环境，CPU 用本次 L3 输入和原始 W4 权重计算第 19 号位置的新 K、V，拷贝调用前的真实分页缓存并写入该位置。脚本核对逻辑位置、物理页、所有未写位置的位值及本次 CPU/310P hidden 参考，然后把输出目录设为 0777。

```bash
conda activate qwen3_5_w4
git -C "$REPO_DIR" pull --ff-only origin main
D="$AI_RUN_DIR/target/tests/decode-t1-after-valid19"
python "$REPO_DIR/targets/cpro/tools/decode_cache_control.py" cpu \
  --fixture "$D/reference" --output "$D/cache-control/cpu"
```

在 DDK 机的 `custom_pypto` 环境拉取 CPU 结果，复用已传入的原 L3 AIR。三个新 case 分别绕过 K、V 或两路 CacheUpdate，选中的缓存输入改为 CPU 写好的完整缓存。Q、attention、gate、o_proj 继续在板端执行。hidden 参考沿用原 310P 输出。

```bash
conda activate custom_pypto
git -C "$REPO_DIR" pull --ff-only origin main
D="$AI_RUN_DIR/target/tests/decode-t1-after-valid19"
C="$D/cache-control"
PROBE="$REPO_DIR/targets/cpro/tools/target_layer_probe.py"
VALIDATE="$REPO_DIR/targets/cpro/tools/validate_draft_vector.py"
python3 "$REPO_DIR/targets/cpro/scripts/pull_artifacts.py" "$C/cpu" &&
python "$REPO_DIR/targets/cpro/tools/decode_cache_control.py" prepare \
  --batch "$D/cpro/air" --cpu "$C/cpu" --output "$C/air" &&
python "$PROBE" compile --batch "$C/air" --output "$C/omc" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --compiler-profile isolated-kirinx90 --use-origin-format \
  --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}"
```

三项均为 `COMPILED` 后启动一次板端执行。

```bash
conda activate custom_pypto
python "$VALIDATE" --config "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}" \
  --compiled "$C/omc" --ddk-root "$CPRO_DDK_ROOT" --output "$C/execution" --apply
```

启动得到确认后，回收并打印各组的 relative L2、cosine 和其他指标。

```bash
conda activate custom_pypto
python "$VALIDATE" --recover "$C/execution" --output "$C/recovery"
python "$PROBE" compare --fixture "$C/execution/fixture" \
  --native "$C/recovery/native" --output "$C/comparison"
cat "$C/comparison/summary.csv"
```

若只换 V 即恢复，说明相同 query 和原 K 下替换 V 路径足以消除本例的 hidden 超差；只换 K 的解释对称。若仅同时替换通过，需联合检查两路。CPU 也计算了当前 token 的投影行，要进一步区分投影与写回，可在原图测量实际写入行，以及更新后 0–18 号位置与输入历史是否逐位一致。若两路都换仍失败，继续检查 query、有效长度、mask、core 和 o_proj 输入。

### L3 首步 Decode 的缓存替换结果

每张图独立重放位置 19 的同一份输入，比较投影后的 hidden 和经过 CPU 检查的 310P hidden。四种图的重复输出均稳定。

| Attention 使用的缓存 | relative L2 | cosine | RMSE |
|---|---:|---:|---:|
| 原 CPro K/V | 757.96909 | −0.052076 | 9.31366 |
| CPU K、CPro V | 5975.50695 | 0.046523 | 73.42493 |
| CPro K、CPU V | 0.733593 | 0.721121 | 0.009014 |
| CPU K/V | 0.366521 | 0.930884 | 0.004504 |

只换 V 就消除了大部分绝对误差，继续换 K 又使 RMSE 约减半。双换后 relative L2、cosine 和平均相对误差仍超限。替换同时提供独立投影的新行和保留历史的完整缓存，单凭 hidden 无法区分当前行计算、历史保留和图编译差异。

在 AIR 机生成 CPU 的 Q、attention core 和 o_proj 输入，复用同一次 capture。输出目录自动设为 0777。

```bash
conda activate qwen3_5_w4
git -C "$REPO_DIR" pull --ff-only origin main &&
D="$AI_RUN_DIR/target/tests/decode-t1-after-valid19" &&
python "$REPO_DIR/targets/cpro/tools/decode_cache_control.py" cpu --stages \
  --fixture "$D/reference" --output "$D/cache-trace/cpu"
```

在 DDK 机拉取 CPU 张量，再从保留的原图和双换 K/V 图各生成一张带中间输出的 AIR。

```bash
conda activate custom_pypto
D="$AI_RUN_DIR/target/tests/decode-t1-after-valid19"
X="$D/cache-trace"
TOOL="$REPO_DIR/targets/cpro/tools/decode_cache_control.py"
PROBE="$REPO_DIR/targets/cpro/tools/target_layer_probe.py"
VALIDATE="$REPO_DIR/targets/cpro/tools/validate_draft_vector.py"
git -C "$REPO_DIR" pull --ff-only origin main &&
python3 "$REPO_DIR/targets/cpro/scripts/pull_artifacts.py" "$X/cpu" &&
python "$TOOL" trace --baseline "$D/cpro" --control "$D/cache-control" --cpu "$X/cpu" --output "$X/air" &&
python "$PROBE" compile --batch "$X/air" --output "$X/omc" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" --compiler-profile isolated-kirinx90 \
  --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}" --use-origin-format
```

两张图均为 `COMPILED` 后，在同一 DDK 终端启动一次板端执行。

```bash
conda activate custom_pypto
python "$VALIDATE" --config "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}" \
  --compiled "$X/omc" --ddk-root "$CPRO_DDK_ROOT" --output "$X/execution" --apply
```

看到 `VECTOR_VALIDATION_STARTED` 后回收该次执行。数值失败也能检查已取回的稳定输出。

```bash
conda activate custom_pypto
python "$VALIDATE" --recover "$X/execution" --output "$X/recovery"
python "$TOOL" inspect --baseline "$D/cpro" --control "$D/cache-control" --trace "$X"
```

先确认两张图的 hidden 各自与原板端输出逐位相同。原图报告 attention 收到的缓存在 0–18 号位置及所有未写区的 FP16 位差，并把位置 19 的值与板端写入操作数比较。双换图核对实际 K/V 和 CPU 张量，比较 Q、core 和 o_proj 输入，再用读回的 Q/K/V、捕获的 mask 和页号、长度 20 独立计算 core。所有浮点对比同时报告 relative L2 和 cosine。

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

如果 AIR 导出机和 DDK 编译机使用独立存储且两端目录同名，在编译机的 `custom_pypto` 环境拉取导出的 `air/`。首次用两台机器的 SSH 地址初始化本地配置，按提示输入账号和密码。设置保存在 `~/.config/qwen35-cpro/transfer.json`。

```bash
python "$REPO_DIR/targets/cpro/scripts/pull_artifacts.py" --init "$AIR_HOST" "$OMC_HOST"
```

新导出的 AIR 已设置传输权限。以前由 Docker 的 root 生成的目录，可以在导出机补设权限。此操作只需本地 Python 和 chmod。

```bash
python "$REPO_DIR/targets/cpro/scripts/pull_artifacts.py" --prepare \
  "$PAIR_ROOT/linear/air" "$PAIR_ROOT/full/air"
```

每次导出完成后，在编译机将 `PAIR_ROOT` 设为与导出机相同的路径，再传本轮生成的目录。只测 Full 时，命令只传 `"$PAIR_ROOT/full/air"`。

```bash
python "$REPO_DIR/targets/cpro/scripts/pull_artifacts.py" \
  "$PAIR_ROOT/linear/air" "$PAIR_ROOT/full/air"
```

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

<a id="t64-valid19-linear"></a>

## T64、19 个有效 token 的 Linear Attention 未缩放基线

本次重放使用物理 64 行、19 个有效 token、零初始状态。GDR 的 `effective_length=[19]`，单次执行 `chunk_size=64`。L0–L2 分别使用 310P 分段推理捕获的本层输入，整层输出 `golden_hidden` 取自同次 310P 调用，并通过了固定输入的 CPU 对照。下表报告 CPro 对 310P 的相对 L2；验收分别检查前 19 行和最后一个有效 token（索引 18）。三层均在独立执行本层时复现了偏差。

| Linear Attention 层 | 前 19 个 token 的 hidden 相对 L2 | token 18 的 hidden 相对 L2 | 整层判定 |
| --- | ---: | ---: | --- |
| L0 | 22.17% | 22.92% | 失败 |
| L1 | 36.24% | 55.30% | 失败 |
| L2 | 35.35% | 56.06% | 失败 |

**L0 的误差在门控 RMSNorm 后明显放大。** 增加中间张量回读后，前 19 行的 hidden 与原板端结果逐位一致。Q、K、V、g、beta 和门控输入均通过 310P 对照。板端 GDR core 对 310P core 的整体相对 L2 为 0.235%，但按 head 做门控 RMSNorm 后，`out_proj` 输入的相对 L2 达到 12.60%，最终 hidden 达到 22.17%。

固定 CPro 的 GDR 输入和门控值，两次都在 CPU 上使用同一套门控 RMSNorm 和量化输出投影，仅替换送入 RMSNorm 的 core。两种 core 的整体相对 L2 仅为 0.0844%，整层 hidden 对 310P 的结果如下。

| 送入 CPU 门控 RMSNorm 的 core | 前 19 个 token 的 hidden 相对 L2 | token 18 的 hidden 相对 L2 | 五项指标 |
| --- | ---: | ---: | --- |
| 板端回读的 CPro core | 22.13% | 22.92% | 失败 |
| 在相同 CPro 输入上用 CPU 重算的 core | 1.93% | 4.75% | 通过 |

**L0 最明显的数值特征是 FP16 次正规数对应的板端输出大量为零。** CPU 在相同输入上重算的 core 中有 37,424 个 FP16 次正规数，其中 37,371 个（99.86%）在 CPro 回读中为零。仅将 CPU core 输出中的次正规数置零，再用同一门控值归一化，CPU 与板端 core 所得归一化结果之间的 L2 差距降至置零前的 10.90%；单看 token 18，仍剩 41.62%。具体在哪一步变零、token 18 的剩余误差来自哪里，以及 L1、L2 是否具有相同机制，仍待验证。[离线回读与 core 替换工具](../targets/cpro/tools/inspect_segment_gdn_trace.py)

算子侧可用 L0 的相同 Q/K/V、g、beta、初态和 `effective_length=[19]` 复现。优先检查 GDR 读出累加到 FP16 core 写回之间的小值，逐项对比 CPU 次正规数与板端零值，再用原 gate 和 out_proj 检查整层 hidden。外部归一化的 q1/q16 对照在编译阶段遇到 `ReduceSumCpro_1` 的 tiling 限制，最近一次二维图报 `innerSize>16`，此次没有板端缩放精度结果。

ConvState 另有一项已观察到的差异。310P 返回物理 64 行末尾 4 行的卷积历史，CPU 的有效序列参考取截至 token 18 的末尾 4 行。本轮独立 prefill 从零状态开始；上面的 L0 core 替换对照使用同一份板端输入。

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

导出文件为 `$JOB/air/export.air`。在 310P 服务器上用同一份 T64 输入生成 CPU 和原生 310P 两套输出。已有 JOB 可以直接调用仓库脚本。

```bash
conda activate qwen3_5_w4
python "$REPO_DIR/targets/cpro/tools/verify_target_prefill_t64.py" \
  --source-root "$TARGET_SOURCE" --config "$TARGET_CONFIG" \
  --fixture "$T64_FIXTURE" --device npu:0 --output "$JOB/cpu-310p"
```

`cpu/outputs/` 和 `native-310p/outputs/` 各保存末位置 logits 与八个状态。CPU 按同一批 token ID 从原始 embedding 表取行，核对输入后，逐层计算 GDN、全注意力、MoE 和 LM head，量化投影从原始 W4 checkpoint 读取。每层 decoder hidden 和九个最终输出的精度指标在 `report.json`。logits 还要核对 top-1，K/V 分别检查有效写入行和未写区域。

当四层 hidden、九项输出、原生 L3 core 审核均通过，且第二次原生运行的九个输出逐位一致并通过比较时，`selected_golden` 才指向本次原生 310P 输出及其 SHA256。这个结论适用于本次四层、空初态、T64 输入。`comparison.l3_math_310p` 单独记录把独立注意力结果接回原模块后的误差，便于定位 L3 对整链结果的影响。精度失败时脚本保留两套输出和报告，退出码为 2。
