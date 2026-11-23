# 310P 与 CPro 的 GDR、RMSNorm 和门控乘法逐张量对比

<a id="gdn-tail"></a>

## 用已有 GDR 输出定位 L0 从 core 到 out_proj 的误差

在 78 的 310P 环境准备数据。三个 bin 均按 FP16 的 token、head、channel 顺序读取，每个 token 有 32 个 head，每个 head 有 128 个通道。可提供 19 或 64 行；只有 19 行时补齐物理 T64，比较 token 0–18。两端共用同一份门控投影输出 `z`。若手上有 310P 实测的 GDR core，可用它替换 `--gold-core` 的 CPU golden。两组参考的后处理均由 310P 原生 L0 模块执行。

```bash
conda activate qwen3_5_w4
git -C "${REPO_DIR:?}" pull --ff-only origin main
T="${AI_RUN_DIR:?}/target/tests"
D="$T/GDN-acc"
R="$T/gdn-tail-l0-valid19-01"
A="${ATTENTION_AIR:-$T/attention4-t64-valid19-v1024-qlate-01/air}"

# 旧 attention AIR 已删除时，用原采集和导出入口恢复。
if [[ ! -f "$A/models/l00_c00_gdn_layer_hidden_o0/export.air" ]]; then
  A="$R/original/air"
  S="$REPO_DIR/targets/cpro/scripts/check_attention4_tmatrix.sh"
  bash "$S" capture "$R/original" &&
    conda activate qwen3_5_cpro &&
    bash "$S" air "$R/original" &&
    conda activate qwen3_5_w4
fi

python "$REPO_DIR/targets/cpro/tools/gdn_tail_accuracy.py" prepare \
  --run "$R" --module-air "$A" --source-root "${TARGET_SOURCE:?}" \
  --config "${NZ_CONFIG:-config/qwen3_5_moe.yaml}" \
  --core "$D/gdr/core_attn.bin" --gold-core "$D/golden/golden_core_attn.bin" \
  --z "${GDN_Z:-$D/inputs/z.bin}"
```

78 会核对原 310P L0 投影输出的 19 行与源 AIR 的 golden 逐位一致，再用指定的 core、z 重放原生 norm 和 W4 投影。CPro 端从源 AIR 保留各输出所依赖的算子、类型转换、gamma、epsilon 和投影权重。输入文件的哈希和真实 310P 模块信息在 `air/manifest.json`，各截取点的源边和外部权重记录在对应的 `export.json`。

在 35 的 DDK 环境执行。同一运行目录在两台服务器上的路径不同时，给 `board` 增加 `--remote-air /78上的运行目录/air`。

```bash
conda activate custom_pypto
git -C "${REPO_DIR:?}" pull --ff-only origin main
R="${AI_RUN_DIR:?}/target/tests/gdn-tail-l0-valid19-01"
python "$REPO_DIR/targets/cpro/tools/gdn_tail_accuracy.py" board \
  --run "$R" --ddk-root "${CPRO_DDK_ROOT:?}" \
  --runtime-config "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}" \
  --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}"
```

六张 OMC 各只返回一个结果。沿 core 分支依次读 `rms_output`（含 gamma）、`mul_output`（与 SiLU 门值相乘）、`gated_return`（门控值转回投影输入 dtype）、`out_proj`（含 DynamicQuant 和 W4 矩阵乘）。另两张分别读 `silu_output`（只依赖 z）和 `out_proj_only`（固定投影输入）。每张图预热三次，再回读十次。板端结果保存在 35 的 `$R/recovery`。重算指标可执行 `python "$REPO_DIR/targets/cpro/tools/gdn_tail_accuracy.py" report --run "$R"`。

| accuracy.csv 的 comparison | 对比 | 回答的问题 |
| --- | --- | --- |
| own_core | CPro 用实测 core，310P 用指定 golden core | 从 raw_core 开始，各阶段实际相差多少 |
| same_core | 两端均用 CPro 的 core 和同一份 z | 对齐输入后，从哪一段开始仍有差异 |
| 310p_core_change | 只在 310P 上替换 core | 已知的 core 差异在同一套后处理下被放大多少 |
| same_proj_input | 两端直接投影同一份 310P 门控结果 | 固定激活时，动态量化和投影共引入多少差异 |

`$R/accuracy.csv` 给出 valid19 和每个 token 的五项指标、MAE、最大绝对误差、参考 RMS 和逐元素超差率。逐元素对比使用 `rtol=1e-3`、`atol=1e-5`，允许超差元素占比最多 0.1%，单独记录通过情况。`delta_L2_pp` 是相对于上一个 core 分支输出的相对 L2 变化（百分点）。终端标出 valid19 和 token18 增幅最大的阶段。各阶段独立编译，跨阶段的增幅可用来选定位点；固定输入对照可检验该段在相同输入下的偏差。

## 汇总 L0 的 dump 和精度

在 10.136.181.35 的 DDK 终端执行。命令从已回收的 `gamma-readout`、`norm-single-l0`、`l0-silu-pair` 和原始 310P capture 中读取 L0 张量，校验后写入 `l0-dump-summary/`。

```bash
conda activate custom_pypto
G="$AI_RUN_DIR/target/tests/gdr-stages-t64-valid19-fresh01"
git -C "$REPO_DIR" pull --ff-only origin main &&
python "$REPO_DIR/targets/cpro/tools/gdr_prefill_stages.py" summarize --group "$G" &&
cat "$G/l0-dump-summary/summary.md"
```

| 文件 | 内容 |
|---|---|
| `summary.md` | 按 L0 计算顺序排列。每行同时显示 valid19、token18 或全张量的相对 L2 / cosine 和精度判定 |
| `summary.csv` | 每项检查的全部指标、非有限值和零值计数、FP16 次正规数对端变零计数。先看 `measurement` 和 `result`，`passed` 只表示该读数与参考的数值比较 |
| `tensor-map.csv`、`manifest.json` | 每个 bin 的步骤、两端 dtype、shape、文件位置、板端 case、槽位和偏移；清单另含 SHA256、AIR 来源及有效 hidden 复现标志 |
| `l0/{310p,cpro}/inputs/` | 实际整层入口，包括 hidden、卷积历史、初始 state 和有效长度。若 310P 的某个参数被编译器裁掉，只保存采集侧文件 |
| `l0/{310p,cpro}/*.bin` | GDR、RMSNorm、SiLU、门控乘积、转型后的 out_proj 输入，以及未经探针修改的整层 `golden_hidden` |
| `silu-result.json` | L0 同次调用的直接 SiLU 校验。汇总中的 `silu_input`、`silu_output`、`mul_rhs` 都来自这张图，后两个名字共享一个板端槽 |

本次实测 L0 在有效 19 个 token 上，SiLU 输出 CPro/310P 相对 L2 为 **0.00160568**，cosine 为 **0.99999883**。CPro 输出对自身实测输入的 SiLU 公式为 **0.00132394 / 0.999999**。这张图的有效 hidden 与原始 CPro 图逐位相等。后续实测门控乘积为 **0.125974 / 0.992034**，整层输出为 **0.221701 / 0.976573**。汇总另外用两端各自的实测左右操作数重建各自乘积，并在 CSV 中单列检查结果。

## 从零生成 L0 GDR 输入输出和 CPU golden

本例只依赖原模型源码、权重和现有 DDK、TBE、板端及传输配置。两台机器的 `REPO_DIR`、`AI_RUN_DIR` 指向各自相同的绝对路径。以下目录可从空目录开始；保留上次结果时可在两端改成同一个新名字。

**10.136.180.78** 的 310P 环境采集原始 valid19 样本。沿用此前的 `TARGET_SOURCE`（含 inference_segment.py），默认采集 YAML 是该目录下的 `config/qwen3_5_moe.yaml`。有自定义采集 YAML 时用 `NZ_CONFIG` 指定。导出环境使用原 `CPRO_SOURCE` 和 `TARGET_CONFIG`。

```bash
conda activate qwen3_5_w4
G="$AI_RUN_DIR/target/tests/l0-gdr-io-t64-valid19-01"
RUN="$REPO_DIR/targets/cpro/scripts/check_l0_gdr_io.sh"
git -C "$REPO_DIR" pull --ff-only origin main &&
bash "$RUN" capture "$G" &&
conda activate qwen3_5_cpro &&
bash "$RUN" air "$G"
```

整层 AIR 仍沿用原采集和导出入口。传输批次仅含 L0 原始 hidden 图和同时返回 hidden、七个 GDR 输入、两个输出的探针图。**10.136.181.35** 在 `custom_pypto` 环境拉取并编译。脚本校验编译前后两处 DDK GDR 源码的 SHA256，保存在本轮的 `gdr-source.sha256`。

```bash
conda activate custom_pypto
G="$AI_RUN_DIR/target/tests/l0-gdr-io-t64-valid19-01"
RUN="$REPO_DIR/targets/cpro/scripts/check_l0_gdr_io.sh"
git -C "$REPO_DIR" pull --ff-only origin main &&
python "$REPO_DIR/targets/cpro/scripts/pull_artifacts.py" --from air "$G/reference" "$G/air" &&
bash "$RUN" compile "$G" &&
bash "$RUN" run "$G" &&
python "$REPO_DIR/targets/cpro/tools/gdr_prefill_stages.py" pack-gdr \
  --dump "$G/dump" --cpu-golden --output "$G/l0-gdr-valid19.tar.gz"
```

两个 case 完整执行且回读成功即可分析数值超差。打包还会验证探针与未插桩图的入口和有效 hidden 一致。包内 `cpu/valid19/output/core.bin` 为 FP16 [1,19,32,128]，`cpu/tensor/output/final_state.bin` 为 FP32 [1,32,128,128]。二者使用 310P 的算子入口与初态，在 CPU 上用 FP32 归一化、逐 token 递推，core 按 query dtype 返回。每端实测相对 CPU 的结果在 `cpu-comparison.csv`。CPro 与 310P 使用各自产生的 Q、K、V、g、beta，CPro 对 CPU 的差异可能含上游误差。

## 从已保存的 L0 dump 打包 GDR 算子输入和输出

在保存了 `l0-dump-summary` 的 OMC 服务器上执行。归档使用所选 dump 的历史读数，脚本会校验 GDR 探针的整层入口和有效 hidden 复现记录，以及每个源 bin 的 SHA256。

```bash
conda activate custom_pypto
G="$AI_RUN_DIR/target/tests/gdr-stages-t64-valid19-fresh01"
git -C "$REPO_DIR" pull --ff-only origin main &&
python "$REPO_DIR/targets/cpro/tools/gdr_prefill_stages.py" pack-gdr \
  --dump "$G/l0-dump-summary" --output "$G/l0-gdr-valid19.tar.gz" &&
tar -xOzf "$G/l0-gdr-valid19.tar.gz" gdr-l0-valid19/README.md
```

每端 15 个 bin。`full64/` 和 `valid19/` 各含 query、key、value、g、beta 和 core。`tensor/` 含完整 initial_state、effective_length 和 final_state。每个目录再按 input、output 分开。用 full64 与 tensor 复现物理调用，用 valid19 分析有效区间。两端各自的算子输入可能已有差异。包内 `comparison.csv` 含前 19 个 token 的汇总、每个有效位置的误差以及 FP16 次正规数在另一端为零的计数。

## L0 用新版 S=1 重跑

在 10.136.181.35 的 DDK 终端执行。脚本读取原实验的 310P capture 和三组已验证的 AIR 探针。若本机缺少输入，脚本会通过 `pull_artifacts.py` 拉取所需的原始 capture 或整层 AIR，在 DDK 端重建缺失的探针。脚本在新目录生成 9 个 L0 AIR，使用当前 `CPRO_DDK_ROOT` 编译为 OMC，再下板、等待回收、生成两端的 bin 和对照表。

```bash
conda activate custom_pypto
G="$AI_RUN_DIR/target/tests/gdr-stages-t64-valid19-fresh01"
RUN="$G/l0-s1-v2"
git -C "$REPO_DIR" pull --ff-only origin main &&
bash "$REPO_DIR/targets/cpro/scripts/check_prefill_gdr_s1.sh" "$RUN" &&
cat "$RUN/l0-dump-summary/chunk-vs-s1.csv"
```

此图专用于整层入口是 T64、有效长度是 19 的首轮 prefill。第 0 步读取原初始 state，第 1–18 步逐次读取上一调用的 FP16 state 输出。每步的五个逐 token 输入都切成一行，`chunk_size=1`，导出前先核对实测整层长度为 INT16 [19]，19 次调用都直接读取 INT16 常量 [1]。旧版为最后一步计算 `19 - 18` 时，OMG 在 `l0s1_valid_length` 的 CastT 拒绝 INT16 输入。新版省去了这段长度运算。顺序拼回 19 个 core，在末尾补 45 个零行供原 T64 后续计算。`final_state` 取 token 18 的返回值。310P 参考继续使用原来实测的 chunk64。

`l0-dump-summary/chunk-vs-s1.csv` 并列显示两个 CPro 版本相对同一 310P 的 L2、cosine、精度判定，以及参考 FP16 次正规数在 CPro 变为零的元素数。`summary.md`、`summary.csv` 和 `tensor-map.csv` 给出新版的完整结果和 bin 索引。每个阶段的有效 hidden 都与此次 S=1 未插桩图逐位核对。若 OMG 裁掉未插桩图的整层长度输入，`entry_effective_length` 在对照表里显示 `NOT_BOUND`。GDR 探针单列读回的整层长度 [19]。`ddk-gdr.json` 记录本次编译使用的 DDK 路径、ID 和可按名称识别的 GDR 文件哈希。再次更换算子包时，给 `RUN` 换一个新目录。

## 重新采集和执行

从原始分段推理重新采集同一个 19-token prompt 的首次 prefill，物理 T64，最后一个有效 token 的下标为 18。L0–L2 各执行一次未缩放的 chunk-64 GDR。每层 CPro 图使用本轮 310P 同层入口的完整 hidden、初始 ConvState、recurrent state 和有效长度，在图内连续计算投影、GDR、RMSNorm、SiLU、门控乘法及输出投影。所有参考张量来自这次 310P 执行。

| 位置 | 保存的张量 |
|---|---|
| GDR | query、key、value、g、beta、initial_state、effective_length、core、final_state |
| RMSNorm | rms_input、gamma、rms_output；epsilon 在清单里。输出取源代码后续使用的归一化激活 |
| 门控 | gate、silu_input、silu_output、mul_lhs、mul_rhs、mul_output、gated_return。mul_output 在返回值转型前采集 |

**1. 10.136.180.78，重新采集 310P 参考**

两台机器使用相同的实验目录名。本次使用 `gdr-stages-t64-valid19-fresh01`。如需再次完整重跑，两边同时换一个新后缀。下列命令使用已有的模型权重、DDK、板端 runtime 和传输账号配置。

```bash
conda activate qwen3_5_w4
G="$AI_RUN_DIR/target/tests/gdr-stages-t64-valid19-fresh01"
git -C "$REPO_DIR" pull --ff-only origin main &&
python "$REPO_DIR/targets/cpro/tools/check_segment_prefill.py" \
  --source-root "$TARGET_SOURCE" --gdr-stages --output "$G/reference"
```

看到 `CAPTURED`、`physical_rows=64`、`valid_rows=19`、`output_pos=18` 后继续。本例补齐后的 64 个 int32 token ID 的 SHA256 为 `ec5119e4b49de3c9e0cd48b7cff7faf13b18a0a10dda89c2a5180b194f05eaf7`，可核对采集日志。`reference/tensor-map.csv` 说明模块入口和每个 310P 观测点。

**2. 同一台 10.136.180.78，生成整层 AIR 和中间张量诊断 AIR**

```bash
conda activate qwen3_5_cpro
G="$AI_RUN_DIR/target/tests/gdr-stages-t64-valid19-fresh01"
TOOLS="$REPO_DIR/targets/cpro/tools"

python "$TOOLS/prepare_cpro_prefill_t64.py" \
  --source-root "$CPRO_SOURCE" --config "$TARGET_CONFIG" \
  --fixture "$G/reference" --attention-layers --output "$G/export" --export &&
python "$TOOLS/gdr_prefill_stages.py" prepare \
  --baseline "$G/export" --reference "$G/reference" \
  --include-baseline --output "$G/air" &&
ls -lh "$G"/air/models/*/export.air
```

第一条命令用本轮采集重新导出三个原始整层 hidden 图。第二条从这些图增加实际算子输出端口，把三张整层图、三张 GDR 图和三张 RMSNorm/门控图写入 `air/manifest.json`。最后应列出九个 `export.air`。采集和导出脚本已将待传输的 `reference`、`air` 目录及内容设为 0777。

三张 RMSNorm 诊断图用运行时 INT32 索引 `gamma_read_indices.bin`（0 到 127）顺序读出 CPro 的 gamma。GatherV2 从原 RMSNorm 的权重边取值，权重仍由该模型提供。采集脚本校验索引的内容、编译后的输入绑定和 gamma 输出槽的连接。最终 `dump/l*/cpro/gamma.bin` 来自板端返回的权重槽。

**3. 10.136.181.35，传输并编译这九个 AIR**

```bash
conda activate custom_pypto
G="$AI_RUN_DIR/target/tests/gdr-stages-t64-valid19-fresh01"
TOOLS="$REPO_DIR/targets/cpro/tools"

git -C "$REPO_DIR" pull --ff-only origin main &&
python "$REPO_DIR/targets/cpro/scripts/pull_artifacts.py" "$G/reference" "$G/air" &&
python "$TOOLS/target_layer_probe.py" compile \
  --batch "$G/air" --output "$G/omc" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --compiler-profile isolated-kirinx90 \
  --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}" --use-origin-format
```

**4. 同一台 10.136.181.35，执行板端模型并回收**

九个 case 均编译成功后，在同一终端执行。首次命令下发九个模型，第二条等待并回收同一次执行。

```bash
conda activate custom_pypto
python "$TOOLS/validate_draft_vector.py" \
  --config "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}" \
  --compiled "$G/omc" --ddk-root "$CPRO_DDK_ROOT" --output "$G/execution" --apply &&
python "$TOOLS/validate_draft_vector.py" \
  --recover "$G/execution" --wait --output "$G/recovery"
```

若在 `stage=prepare` 收到 `compiled output … shape or dtype is incompatible`，先在 DDK 服务器读取本次产物的输出槽。下面以 L0 的 RMSNorm/门控图为例。命令列出每个槽的 310P 参考、AIR 源边和 OMC NetOutput 描述，以及能从 OMC 元数据追溯的直接生产者。末行给出与下发流程相同的绑定检查结果。

```bash
conda activate custom_pypto
G="$AI_RUN_DIR/target/tests/gdr-stages-t64-valid19-fresh01"
python "$REPO_DIR/targets/cpro/tools/gdr_prefill_stages.py" inspect-abi \
  --compiled "$G/omc" --ddk-root "$CPRO_DDK_ROOT" \
  --case l00_c00_norm_prefill_stages_o0
```

如果旧 OMC 的 `gamma` 输出边为空、`shape=null`，DDK 服务器上已有的 `fresh01/air` 批次内含三个原始整层 AIR，可以在该机直接重建诊断图。下列命令从同一次 310P 采集生成新的九图批次。通过 ABI 检查后，在第 4 步使用 `$FIX/omc`、`$FIX/execution` 和 `$FIX/recovery`，第 5 步使用 `--reference "$G/reference" --trace "$FIX" --output "$FIX/dump"`。

```bash
conda activate custom_pypto
G="$AI_RUN_DIR/target/tests/gdr-stages-t64-valid19-fresh01"
FIX="$G/gamma-readout"
TOOLS="$REPO_DIR/targets/cpro/tools"
git -C "$REPO_DIR" pull --ff-only origin main &&
python "$TOOLS/gdr_prefill_stages.py" prepare \
  --baseline "$G" --reference "$G/reference" --include-baseline --output "$FIX/air" &&
python "$TOOLS/target_layer_probe.py" compile \
  --batch "$FIX/air" --output "$FIX/omc" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --compiler-profile isolated-kirinx90 \
  --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}" --use-origin-format &&
(
  set -e
  for layer in 00 01 02; do
    python "$TOOLS/gdr_prefill_stages.py" inspect-abi \
      --compiled "$FIX/omc" --ddk-root "$CPRO_DDK_ROOT" --case "l${layer}_c00_norm_prefill_stages_o0"
  done
)
```

**5. 同一台 10.136.181.35，导出两端 bin 并逐张量对比**

回收结果中 `device_execution=EXECUTED_OUTPUTS_RETRIEVED` 表示已经取得板端计算结果。即使此时 `numerical_validation=FAILED`，仍可执行下面的命令定位误差。

```bash
conda activate custom_pypto
python "$TOOLS/gdr_prefill_stages.py" compare \
  --reference "$G/reference" --trace "$G" --output "$G/dump" &&
head -n 8 "$G/dump/tensor-map.csv"
```

`dump/l0/{310p,cpro}/mul_output.bin` 等文件按后端保留实际 dtype，统一为清单中声明的逻辑形状。这些 .bin 按 C 连续顺序写入，无文件头。CPro 使用回读包中的第一次调用，校验各次调用有效区的稳定性。两个设备的逐 token 张量保留完整 64 行。`dump/summary.csv` 用 310P 对照 CPro 的各个浮点张量，分别给出 relative L2、cosine、最大绝对误差、RMSE、KL 和平均相对误差，也统计参考 FP16 次正规数在对端变为零的个数。逐 token 张量分别统计 0–18 和 token 18。状态和 gamma 使用全张量，effective_length 按整数精确比较。

`dump/manifest.json` 记录数组的路径、shape、dtype 和 SHA256、CPro 的实际取数边，同一条边的别名，以及每张诊断图的有效区 hidden 是否与本次同批整层基线的 19 行有效输出逐位相同。解释原图误差时使用 `valid_hidden_equal_bits=true` 的图。

### 如何辨认一个 bin

`dump/tensor-map.csv` 每行对应一个导出文件，给出中文计算步骤、张量含义、后端、相对路径、逻辑形状、数据类型、字节数以及原始文件位置。`module_edge` 指向原计算图中被读取的张量，`air_edge` 指向诊断图实际返回的边。gamma 的这两项分别对应 RMSNorm 的权重输入和顺序 Gather 的输出。`source_offset_bytes` 是从零开始的字节偏移。在 CPro 原始 `recovery/native/<case>/0.bin` 中，`0` 是调用序号，文件按输出槽依次拼接。槽 0 是用于核对基线的整层 hidden，后面的槽是本组采集的中间张量。用 `source_slot`、`source_offset_bytes` 和 `source_length_bytes` 可以取出对应槽的原始字节。多个语义名称读同一条 AIR 边时，表中列出相同的槽和偏移。分拆后的 `dump/l*/cpro/*.bin` 直接从文件头读取。板端执行目录中的 `input_N.bin` 是整图输入，`golden_N.bin` 是该输出槽的参考值。

例如读取 L0 的 CPro 乘法结果并查看最后一个有效 token。以下命令在生成 dump 的 DDK 终端执行。

```bash
conda activate custom_pypto
python - "$G/dump" <<'PYTHON'
import csv
import json
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1])
with (root / 'tensor-map.csv').open(encoding='utf-8') as f:
    row = next(r for r in csv.DictReader(f)
               if (r['layer'], r['backend'], r['tensor']) == ('0', 'cpro', 'mul_output'))
x = np.fromfile(root / row['path'], dtype=row['dtype']).reshape(json.loads(row['shape']))
print(row['step'], row['meaning'], x.shape, x.dtype)
print(x[0, 18])
PYTHON
```

### RMSNorm 诊断图的 hidden 与原图不同

先用同批九个模型已有的回收数据，量化每个诊断图的 hidden 与原 CPro 图在有效区的差异。命令核验已导出的 CPro 张量与回收包一致，并在各后端内部用已保存的操作数核对 RMSNorm、SiLU 和乘法。用于 CPro 的 core 与 rms_input 来自两张图，这项对比单独标星。其它公式使用同一张 norm 图内的张量。统计量包括相对 L2、cosine、非零和非有限元素数。模型参考继续使用 310P。

```bash
conda activate custom_pypto
G="$AI_RUN_DIR/target/tests/gdr-stages-t64-valid19-fresh01"
FIX="$G/gamma-readout"
TOOLS="$REPO_DIR/targets/cpro/tools"
git -C "$REPO_DIR" pull --ff-only origin main &&
python "$TOOLS/gdr_prefill_stages.py" audit-dump --dump "$FIX/dump" --trace "$FIX"
```

如果需要重新取得 L0 的 RMSNorm/门控读数，在当前 DDK 服务器直接用已有原始整层 AIR 构建对照。每张图返回原 hidden 和一个被观测张量。使用同一条边的乘法输入复用该槽。仅观测 gamma 的图会添加顺序索引输入。每个诊断图仍然从同一份 310P 层入口计算到输出投影。

```bash
conda activate custom_pypto
G="$AI_RUN_DIR/target/tests/gdr-stages-t64-valid19-fresh01"
FIX="$G/gamma-readout"
S="$G/norm-single-l0"
TOOLS="$REPO_DIR/targets/cpro/tools"

python "$TOOLS/gdr_prefill_stages.py" prepare \
  --baseline "$G" --reference "$G/reference" --split-norm-layers 0 --output "$S/air" &&
python "$TOOLS/target_layer_probe.py" compile \
  --batch "$S/air" --output "$S/omc" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --compiler-profile isolated-kirinx90 \
  --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}" --use-origin-format &&
python "$TOOLS/validate_draft_vector.py" \
  --config "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}" \
  --compiled "$S/omc" --ddk-root "$CPRO_DDK_ROOT" --output "$S/execution" --apply &&
python "$TOOLS/validate_draft_vector.py" \
  --recover "$S/execution" --wait --output "$S/recovery"
```

回收显示 `EXECUTED_OUTPUTS_RETRIEVED` 后执行下面的命令。GDR 张量与整层基线使用前一次九图测量，每个 norm 读数使用对应的新图。先看每项 `baseline_hidden_reproduced`，再看通过复现检查的张量在 valid19 和 token18 的精度。L0 验证后，可另设目录并用 `--split-norm-layers 1 2` 处理其余两层。

```bash
python "$TOOLS/gdr_prefill_stages.py" compare \
  --baseline "$FIX" --reference "$G/reference" --trace "$S" --output "$S/dump"
```

### 单独读出 SiLU 仍改变原始 hidden

L0 单输出测量中，GDR core、RMSNorm 输出和门控乘积在有效 19 行的相对 L2 / cosine 分别为 0.00234627 / 0.999999、0.013495 / 0.999909 和 0.125974 / 0.992034。这些图与原始 CPro 整层图有效 hidden 逐位相同。直接导出 SiLU 输出的图未复现 hidden。该图的 SiLU 和共用此边的 mul_rhs 文件只描述该诊断图。

先在 10.136.181.35 离线核对现有分拆数据。该命令校验 dump 中文件的长度和 SHA256，只用复现 hidden、模块输入相同的 CPro 图做乘积分解。310P 使用实测乘法右操作数，CPro 使用实测 silu_input 计算公式值。这个值只用于分析，原始 bin 保留实测来源。分析打印 valid19 和 token18 的 L2/cos，报告只替换左操作数、只替换门值、两者交互及重建残差在实测误差方向上的投影，以及 310P FP16 core 为次正规数且 CPro 为零的位置占乘积平方误差的份额。

```bash
conda activate custom_pypto
G="$AI_RUN_DIR/target/tests/gdr-stages-t64-valid19-fresh01"
FIX="$G/gamma-readout"
S="$G/norm-single-l0"
TOOLS="$REPO_DIR/targets/cpro/tools"
git -C "$REPO_DIR" pull --ff-only origin main &&
python "$TOOLS/gdr_prefill_stages.py" audit-product --dump "$S/dump" --layer 0
```

### L0 实测：门控乘积的偏差集中在 GDR core 的小值上

未缩放的 L0 使用 310P 整层入口，物理 T64、有效 19、GDR chunk=64。下表每格为相对 L2 / cosine。前三行以 310P 实测乘积为参考，第四行以 CPro 实测乘积为参考。左操作数是含 gamma 的 RMSNorm 输出，右操作数是 SiLU 门值。

| 门控乘积对照 | 有效 19 个 token | 最后一个有效 token（18） |
|---|---:|---:|
| CPro 实测乘积 | 0.125974 / 0.992034 | 0.087697 / 0.996147 |
| 只换 CPro 实测左操作数，保留 310P 实测右操作数 | 0.125960 / 0.992035 | 0.0876601 / 0.996151 |
| 保留 310P 左操作数，只用 CPro 门输入按 SiLU 公式求右操作数 | 0.000830871 / 1 | 0.000468806 / 1 |
| CPro 实测左操作数 × SiLU(CPro 实测门输入)，对比 CPro 实测乘积 | 0.00164140 / 0.999999 | 0.00272336 / 0.999996 |

左操作数变化在实测乘积误差方向上的有符号投影占有效 19 行的 **99.9789%**，token 18 为 **99.9117%**。乘积重建说明门值公式能够解释这些位置对最终乘积的整体影响。直接导出 CPro SiLU 的图改变了有效 hidden，其读数不能代表原图的逐元素 SiLU 值。

按同一坐标筛选 **310P core 为 FP16 非零次正规数且 CPro core 为零** 的元素，有效区命中 37,759 / 77,824 个，聚集了实测乘积**平方误差的 99.0114%**。token 18 命中 1,847 / 4,096 个，占该 token 平方误差的 94.2066%。这里按坐标累加平方误差，和上面的方向投影是两个统计量。

同事可在 `$S/dump/l0/{310p,cpro}/` 对照 `core.bin`（GDR 读出）、`rms_input.bin`（core 转成 FP32）、`mul_lhs.bin`（RMSNorm 输出）和 `mul_output.bin`（门控乘积）。张量按 [batch, token, head, channel] 排列，shape 和 dtype 见 `$S/dump/tensor-map.csv`。先看 token 6 / head 2（占有效区乘积平方误差 6.34%）和 token 18 / head 17（占最后一个有效 token 乘积平方误差 28.07%）。在算子内复用同一份 CPro GDR 操作数，比较首次降精度前的读出值与最终 FP16 core，可以区分小值在块内计算时已经消失，还是在转换或写回时变成零。当前 dump 只观测到 GDR 输出边界。

### 直接读出板端 SiLU 的补充实验

要验证 L0 的 SiLU，在 DDK 主机 10.136.181.35 上执行下列命令。脚本从该机现有的原始 AIR 和 310P 采集生成一张整层图，一次调用返回原始 hidden、Swish 的 FP32 输入和经过恒等索引 Gather 的 FP32 输出。图内原始乘法仍直接消费同一个 Swish 输出。中断后用同一命令回收已有请求。

```bash
conda activate custom_pypto
git -C "$REPO_DIR" pull --ff-only origin main &&
bash "$REPO_DIR/targets/cpro/scripts/check_prefill_silu.sh"
```

结果在 `gdr-stages-t64-valid19-fresh01/l0-silu-pair/dump/`。两端实测的门输入和输出在 `l0/{310p,cpro}/{silu_input,silu_output}.bin`，`tensor-map.csv` 记录文件含义和源槽。公式只用于校验实测输出。`silu-result.json` 的 `PASSED` 要求模块输入与基线一致、有效 hidden 与原 CPro 图逐位相同，并且 valid19 和 token18 的跨端输入、输出及各端对自身输入的 SiLU 公式检查都通过五项指标。`NUMERICAL_FAILED` 可从 `failed_checks` 区分失败的比较。`READOUT_NOT_REPRESENTATIVE` 表示读出图未复现基线条件，原图的 SiLU 精度仍待核实。

#### 只增加 SiLU 输出端口的实验

为取得直接的板端 SiLU 输出，可在同一 DDK 终端再编译一张 L0 图。该图从同一原始 AIR 引出 Swish 的输出，reshape 成 [2048,128]，用动态索引 0 到 2047 沿第 0 维逐行 Gather，保持数据类型和元素顺序。观察分支以外的原始门控乘法继续读取原来的数据边。

```bash
conda activate custom_pypto
T="$G/l0-silu-gather"
CASE=l00_c00_norm_silu_output_prefill_stages_o0
(
  set -e
  python "$TOOLS/gdr_prefill_stages.py" prepare \
    --baseline "$G" --reference "$G/reference" --indexed-silu-layer 0 --output "$T/air"
  python "$TOOLS/target_layer_probe.py" compile \
    --batch "$T/air" --output "$T/omc" \
    --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
    --compiler-profile isolated-kirinx90 \
    --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}" --use-origin-format
  python "$TOOLS/gdr_prefill_stages.py" inspect-abi \
    --compiled "$T/omc" --ddk-root "$CPRO_DDK_ROOT" --case "$CASE"
  python "$TOOLS/validate_draft_vector.py" \
    --config "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}" \
    --compiled "$T/omc" --ddk-root "$CPRO_DDK_ROOT" --output "$T/execution" --apply
  python "$TOOLS/validate_draft_vector.py" --recover "$T/execution" --wait --output "$T/recovery"
)
```

回收显示 `EXECUTED_OUTPUTS_RETRIEVED` 后运行下面两条命令。在新目录汇总原 GDR、已有单输出图和本次测得的 SiLU。检查 `norm_silu_output valid hidden equals original CPro run`。若为 True，audit-product 额外比较实测 CPro 右操作数与 SiLU 公式、两侧实测操作数的乘积与板端实测乘积。若为 False，审计继续使用公式门值，并保留该图的诊断标记。`summary.csv` 记录误差，`tensor-map.csv` 说明各个 bin 的计算步骤、来源及原始输出槽。

```bash
python "$TOOLS/gdr_prefill_stages.py" compare \
  --baseline "$FIX" --reference "$G/reference" --trace "$S" \
  --silu-readout "$T" --output "$T/dump" &&
python "$TOOLS/gdr_prefill_stages.py" audit-product --dump "$T/dump" --layer 0
```

## L0 三个单算子的 CPU、310P、CPro 对比矩阵

下面的 `review` 只读已有的三张 AIR，在 78 或拥有上一轮 tail AIR 的主机均可运行。它打印原图的输入边、输出边、形状、引用权重及截取后每个 GE 节点的名称、类型和输入。OMG 编译可能融合节点。

```bash
conda activate qwen3_5_w4
git -C "${REPO_DIR:?}" pull --ff-only origin main
T="${AI_RUN_DIR:?}/target/tests"
A="${TAIL_AIR:-$T/gdn-tail-l0-valid19-01/air}"
P="$REPO_DIR/targets/cpro/tools/gdn_op_matrix.py"
python "$P" review --air "$A"
```

| 单算子 | 三端使用的 310P 原生 L0 输入 | 输出边界 |
| --- | --- | --- |
| RMSNorm | GDR 的 core，按 128 通道归一化 | 原层 gamma 和 epsilon 参与计算后的 RMSNorm 输出 |
| SiLU | 原层门控投影的 z | SiLU(z)，在门控乘法之前 |
| out_proj | 310P 原层门控乘法、回转 FP16、reshape 后的投影输入 | DynamicQuant、QuantBatchMatmulV4444 及原图输出视图后的 attention 投影，止于残差相加之前 |

在 78 运行 `prepare` 后，三个 `air/models/*/export.air` 均可单独编译。复用六图批次会保留这三张原 AIR 的字节，也可把 `--source-air` 指向包含 `l00_c00_gdn_layer_hidden_o0` 的四层 attention AIR 批次重新截取。每张图使用原生 T64 输入，指标统计第 0–18 行。CPU 投影使用原 checkpoint 的 group-256 W4 权重和 scale，逐 token 动态量化，int32 点积、FP32 分组累加、FP16 输出。

```bash
R="$T/gdn-op-matrix-l0-valid19-01"
python "$P" prepare --run "$R" --source-air "$A" \
  --source-root "${TARGET_SOURCE:?}" --config "${NZ_CONFIG:-config/qwen3_5_moe.yaml}"
cat "$R/air/boundaries.json"
```

在 35 执行板端测试，`--remote-air` 可指定 78 的新批次路径。每张 OMC 预热三次，回读十次。复用已有结果时执行 `report --run "$R"`。

```bash
conda activate custom_pypto
git -C "${REPO_DIR:?}" pull --ff-only origin main
R="${AI_RUN_DIR:?}/target/tests/gdn-op-matrix-l0-valid19-01"
python "$REPO_DIR/targets/cpro/tools/gdn_op_matrix.py" board \
  --run "$R" --ddk-root "${CPRO_DDK_ROOT:?}" \
  --runtime-config "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}" \
  --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}"
```

`$R/matrices.csv` 每三行是一张矩阵，行是 actual，列是 reference。覆盖 valid19 和每个有效 token。每个方向分别计算 cosine、relative L2、KL、RMSE、平均相对误差、MAE、最大绝对误差、逐元素超差数、总数、超差率和两个判定。超差按 `np.isclose(actual, reference, rtol=1e-3, atol=1e-5, equal_nan=True)`，超差率门限为 0.1%。脚本退出状态沿用 CPro 对 310P 在 valid19 和 token18 的五指标门限。

### 核对本次 310P 输出投影的量化路径

78 的采集器调用原 `inference_segment.py` 时显式传入 `--quant_mode=enable --quant_bits=4`。三端的投影输入都是 310P 捕获的同一份门控输出。在 78 用已有矩阵运行目录复核实际量化操作数，只执行 L0 的 310P 采集和 CPU 对照。

```bash
R="${AI_RUN_DIR:?}/target/tests/gdn-op-matrix-l0-valid19-01"
python "${REPO_DIR:?}/targets/cpro/tools/gdn_op_matrix.py" audit-proj \
  --run "$R" --source-root "${TARGET_SOURCE:?}" --config "${NZ_CONFIG:-config/qwen3_5_moe.yaml}"
```

先看 `matches_previous_matrix`。三个值为 true 表示完整投影输入和 valid19 的 310P、CPU 输出均复现原批次。`VERIFIED_A8_PACKED_W4` 表示在这次原生矩阵乘入口抓到 INT8 激活，打包权重及分组 scale 与原 checkpoint 的 W4 逐位吻合。`a8_different` 统计 310P 和 CPU 动态量化产生不同 INT8 码的个数，`token_scale_max_abs` 比较两端逐 token scale。`native_vs_cpu_on_native_codes_relative_l2` 用 310P 的实际 A8 码和 scale 在 CPU 重算分组点积，再与 310P 输出比较。新一轮 `prepare` 也会把这组记录保存在 `air/manifest.json` 的 `native_310p.out_proj_quantization` 下。

## L0 卷积后的 SiLU 和 L0/L3 attention 的通用截取

在 78 准备卷积后的 SiLU，在 35 用同一份输入独立上板。两条命令见 [单测 L0 的 F.silu(conv_out)](guides/attention-step-example.md)。原生输入形状是 `[1,8192,64]`，沿最后一维统计 valid19 和 token18。

### 按 GE 端口截取其他步骤

在 78 用 `list` 枚举实际端口。每行列出一个节点、它的输入边、各个输出的 `edge`、dtype 和 shape。层号支持 `0`（linear attention）和 `3`（full attention）。`--match` 可省略。

```bash
P="${REPO_DIR:?}/targets/cpro/tools/attention_step.py"
T="${AI_RUN_DIR:?}/target/tests"
A="${ATTENTION_AIR:-$T/attention4-t64-valid19-v1024-qlate-01/air}"
python "$P" list --source-air "$A" --layer 3 --match RmsNorm
```

例如要独立导出上面列出的某个 RMSNorm，复制该节点第一个激活输入的完整 `上游节点名:端口`，以及该节点归一化输出的完整 `节点名:0`。指定该边实际的 dtype 和静态维度。源 AIR 的空 shape 表示该文件没有声明维度，须按对应模块的实测 shape 填写。下列模板中的变量由选定节点决定。

```bash
S="${AI_RUN_DIR:?}/target/tests/l3-selected-step-01"
python "$P" extract --source-air "$A" --layer 3 --run "$S" \
  --input "${INPUT_EDGE:?}" "${INPUT_DTYPE:?}" "${INPUT_SHAPE:?例如1,64,2048}" \
  --output-edge "${OUTPUT_EDGE:?}"
cat "$S/slice/slice.json"
```

每个 `--input EDGE DTYPE DIMS` 是一个新的运行时输入，多个输入重复给出。保留从输出回溯到这些输入之间的原节点和常量，相关 FileConstant 随 AIR 一起导出。当前接口支持一个输出端口、静态逻辑形状、无嵌套子图的片段。参数控制具体的图边界，所选节点上游的常量参数可以留在模型中。

在 35 用相同的 `S` 相对目录执行，工具会拉取 78 的 `slice/`，编译并打印生成的 OMC 路径。可用 `--remote-slice` 指定 78 上的绝对路径。

```bash
S="${AI_RUN_DIR:?}/target/tests/l3-selected-step-01"
python "$P" compile --run "$S" --ddk-root "${CPRO_DDK_ROOT:?}" \
  --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}"
```

通用入口生成所选步骤的 AIR、边界清单和 OMC。L0 的 `qkv_conv_silu`、`gate_silu`、`gated_rmsnorm` 和 `out_proj` 支持按步骤采集 310P 输入并生成三端矩阵。两台机器的命令见[单步骤执行示例](guides/attention-step-example.md)。

可直接传给 `--step` 的命名步骤见[步骤说明表](../specs/attention-step-presets.csv)。在 78 执行以下命令可导出本次真实 AIR 的完整步骤表，包含两个层的命名片段和按节点输出端口选取的步骤。

```bash
python "$P" catalog --source-air "$A" --output "$T/attention-l0-l3-steps.csv"
```

表中的 `runtime_inputs` 给出各输入的边名、dtype、shape，`constant_inputs` 列出保留在图内的常量边，`output` 给出输出规格。每行都有源 AIR 哈希。`ready` 表示截取参数齐全，具体编译结果以 OMG 为准。`supply_input_abi` 表示需要用 `--input EDGE DTYPE DIMS` 补足表中列出的输入规格。`manual_edges` 表示涉及控制依赖，需要审核输入切点。`unresolved_preset` 表示这张 AIR 的结构未匹配到对应预设。

例如在 78 选择 L3 的完整输出投影，或者把 `--step` 换成同层 CSV 的某个 `节点名:输出端口`。若表中提示输入规格缺失，在命令后追加该输入的 `--input` 参数。之后在 35 运行上面的 `compile`。

```bash
S="${AI_RUN_DIR:?}/target/tests/l3-out-proj-only-01"
python "$P" extract --source-air "$A" --layer 3 --step out_proj --run "$S"
```

### 按原 forward 查看步骤

静态步骤表按层排列。L0 覆盖四个输入投影、卷积窗口、depthwise conv、卷积 SiLU、Q/K/V 拆分与 head 整理、beta、衰减 g、GDR 的 core 和 final state、门控归一化、门控乘法及输出投影。L3 覆盖 Q/gate 拆分、Q/K norm、cos/sin 查表、partial RoPE、Q 与 KV 布局、缓存索引与写入、融合 attention、解包、sigmoid 门控及输出投影。每个投影都可通过 `_a8`、`_token_scale`、`_matmul` 选择量化激活、逐 token 尺度、矩阵乘三个边界。矩阵乘接收整数激活和动态尺度，使用 AIR 中的原权重。

在 78 导出便于 review 的源码步骤表。去掉 `--presets-only` 可同时查看这份图的 GE 节点端口。以区分 out_proj 量化误差和矩阵乘误差为例，每个输出使用独立目录。

```bash
python "$P" catalog --source-air "$A" --presets-only --output "$T/attention-forward-steps.csv"
for step in out_proj_a8 out_proj_token_scale out_proj_matmul; do
  python "$P" extract --source-air "$A" --layer 0 --step "$step" --run "$T/l0-step-$step"
done
```

`reason` 记录未匹配的原因，例如两个逻辑步骤在导出时共用同一条边、输出被裁剪或名称对应多个分支。审查每个切片的 `slice/slice.json` 可核对实际输入端口顺序、输出边和保留的 GE 节点。GDR 内部的 T-matrix 和融合 full attention 内部的 softmax 需要算子内读出或另行导出诊断图。
