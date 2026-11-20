# 310P 与 CPro 的 GDR、RMSNorm 和门控乘法逐张量对比

复用 19 个有效 token 的首次 prefill，物理 T64，L0–L2 各执行一次未缩放的 chunk-64 GDR。310P 采集原始分段推理的模块入口和中间张量。CPro 的每层用本次 310P 捕获的完整 hidden、初始 ConvState、recurrent state 和有效长度启动，在图内连续执行投影、GDR、RMSNorm、SiLU 和门控乘法。CPro 在原来三张整层 hidden AIR 上读出算子边，每层两张诊断图，分别读 GDR 和归一化/门控张量。

| 位置 | 保存的张量 |
|---|---|
| GDR | query、key、value、g、beta、initial_state、effective_length、core、final_state |
| RMSNorm | rms_input、gamma、rms_output；epsilon 在清单里。输出取源代码后续使用的归一化激活 |
| 门控 | gate、silu_input、silu_output、mul_lhs、mul_rhs、mul_output、gated_return。mul_output 在返回值转型前采集 |

**1. 在 10.136.180.78 采集 310P 参考张量**

```bash
conda activate qwen3_5_w4
G="$AI_RUN_DIR/target/tests/gdr-stages-t64-valid19"
git -C "$REPO_DIR" pull --ff-only origin main &&
python "$REPO_DIR/targets/cpro/tools/check_segment_prefill.py" \
  --source-root "$TARGET_SOURCE" --gdr-stages --output "$G/reference"
```

已有完整的 `reference/manifest.json` 且含有 L0–L2 的 `gdr_stages` 时可以直接复用采集。新采集只跑 310P 分段推理。`reference/tensor-map.csv` 逐文件给出计算步骤和 dtype。采集目录自动设置传输权限 0777。

**2. 在同一台 10.136.180.78 生成 6 个 CPro AIR**

`prepare` 读取 `$PAIR/air` 下已有的 L0–L2 整层基线和刚采集的 `$G/reference`，为每层生成 GDR、RMSNorm/门控两张诊断图。输入和参考输出也随 AIR 一起写入新批次。

```bash
conda activate qwen3_5_cpro
G="$AI_RUN_DIR/target/tests/gdr-stages-t64-valid19"
PAIR="$AI_RUN_DIR/target/tests/attention-layers-t64-valid19"

git -C "$REPO_DIR" pull --ff-only origin main &&
python "$REPO_DIR/targets/cpro/tools/gdr_prefill_stages.py" prepare \
  --baseline "$PAIR" --reference "$G/reference" --output "$G/air" &&
ls -lh "$G"/air/models/*/export.air
```

输出应列出 6 个 `export.air`。脚本会给生成的 `$G/air` 设置传输权限 0777。

**3. 在 10.136.181.35 拉取 AIR、编译 OMC 并下发**

```bash
conda activate custom_pypto
G="$AI_RUN_DIR/target/tests/gdr-stages-t64-valid19"
PAIR="$AI_RUN_DIR/target/tests/attention-layers-t64-valid19"
TOOLS="$REPO_DIR/targets/cpro/tools"

git -C "$REPO_DIR" pull --ff-only origin main &&
python "$REPO_DIR/targets/cpro/scripts/pull_artifacts.py" "$G/reference" "$G/air" &&
python "$TOOLS/target_layer_probe.py" compile \
  --batch "$G/air" --output "$G/omc" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --compiler-profile isolated-kirinx90 \
  --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}" --use-origin-format &&
python "$TOOLS/validate_draft_vector.py" \
  --config "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}" \
  --compiled "$G/omc" --ddk-root "$CPRO_DDK_ROOT" --output "$G/execution" --apply
```

**在同一 DDK 终端回收并对比**。板端执行完成后运行。诊断图里的数值超阈值会使 recover 返回非零，取得实际输出后仍可生成两端报告。

```bash
conda activate custom_pypto
python "$TOOLS/validate_draft_vector.py" --recover "$G/execution" --output "$G/recovery"
python "$TOOLS/gdr_prefill_stages.py" compare \
  --baseline "$PAIR" --reference "$G/reference" --trace "$G" --output "$G/dump"
```

`dump/l0/{310p,cpro}/mul_output.bin` 等文件按后端保留实际 dtype，统一为清单中声明的逻辑形状。这些 .bin 按 C 连续顺序写入，无文件头。CPro 使用回读包中的第一次调用，校验各次调用有效区的稳定性。两个设备的逐 token 张量保留完整 64 行。`dump/summary.csv` 用 310P 对照 CPro 的各个浮点张量，分别给出 relative L2、cosine、最大绝对误差、RMSE、KL 和平均相对误差，也统计参考 FP16 次正规数在对端变为零的个数。逐 token 张量分别统计 0–18 和 token 18。状态和 gamma 使用全张量，effective_length 按整数精确比较。

`dump/manifest.json` 记录数组的路径、shape、dtype 和 SHA256、CPro 的实际取数边，同一条边的别名，以及每张诊断图的有效区 hidden 是否与之前未缩放的板端结果逐位相同。解释原图误差时使用 `valid_hidden_equal_bits=true` 的图。

### 如何辨认一个 bin

`dump/tensor-map.csv` 每行对应一个导出文件，给出中文计算步骤、张量含义、后端、相对路径、逻辑形状、数据类型、字节数以及原始文件位置。`source_offset_bytes` 是从零开始的字节偏移。在 CPro 原始 `recovery/native/<case>/0.bin` 中，`0` 是调用序号，文件按输出槽依次拼接。槽 0 是用于核对基线的整层 hidden，后面的槽是本组采集的中间张量。用 `source_slot`、`source_offset_bytes` 和 `source_length_bytes` 可以取出对应槽的原始字节。多个语义名称读同一条 AIR 边时，表中列出相同的槽和偏移。分拆后的 `dump/l*/cpro/*.bin` 直接从文件头读取。板端执行目录中的 `input_N.bin` 是整图输入，`golden_N.bin` 是该输出槽的参考值。

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
