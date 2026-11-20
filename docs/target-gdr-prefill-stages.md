# 310P 与 CPro 的 GDR、RMSNorm 和门控乘法逐张量对比

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
