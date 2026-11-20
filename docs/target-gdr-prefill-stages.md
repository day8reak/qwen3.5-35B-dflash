# GDR、RMSNorm 和门控乘法的三端逐张量对比

复用 19 个有效 token 的首次 prefill，物理 T64，L0–L2 各执行一次未缩放的 chunk-64 GDR。CPU 使用每层各自捕获的 310P 模块入口 hidden 和初始状态，独立计算该层的投影、卷积、GDR 递推及后续门控。310P 采集原始分段推理的实参、返回值和表达式中间值。CPro 在原来三张整层 hidden AIR 上读出算子边，每层两张诊断图，分别读 GDR 和归一化/门控张量。

| 位置 | 保存的张量 |
|---|---|
| GDR | query、key、value、g、beta、initial_state、effective_length、core、final_state |
| RMSNorm | rms_input、gamma、rms_output；epsilon 在清单里。输出取源代码后续使用的归一化激活 |
| 门控 | gate、silu_input、silu_output、mul_lhs、mul_rhs、mul_output、gated_return。mul_output 在返回值转型前采集 |

**310P 采集机**（10.136.180.78）

```bash
conda activate qwen3_5_w4
G="$AI_RUN_DIR/target/tests/gdr-stages-t64-valid19"
git -C "$REPO_DIR" pull --ff-only origin main &&
python "$REPO_DIR/targets/cpro/tools/check_segment_prefill.py" \
  --source-root "$TARGET_SOURCE" --gdr-stages --output "$G/reference"
```

此命令先跑真实 310P，再在另一个进程跑 CPU。完成后，`reference/tensor-map.csv` 已列出 310P 和 CPU 实际导出的各个 bin，目录自动设置传输权限 0777。

**DDK 编译和板端下发**（10.136.181.35）

```bash
conda activate custom_pypto
G="$AI_RUN_DIR/target/tests/gdr-stages-t64-valid19"
PAIR="$AI_RUN_DIR/target/tests/attention-layers-t64-valid19"
TOOLS="$REPO_DIR/targets/cpro/tools"

git -C "$REPO_DIR" pull --ff-only origin main &&
python "$REPO_DIR/targets/cpro/scripts/pull_artifacts.py" "$G/reference" &&
python "$TOOLS/gdr_prefill_stages.py" prepare \
  --baseline "$PAIR" --reference "$G/reference" --output "$G/air" &&
python "$TOOLS/target_layer_probe.py" compile \
  --batch "$G/air" --output "$G/omc" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --compiler-profile isolated-kirinx90 \
  --tbe-diagnosis "${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}" --use-origin-format &&
python "$TOOLS/validate_draft_vector.py" \
  --config "${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}" \
  --compiled "$G/omc" --ddk-root "$CPRO_DDK_ROOT" --output "$G/execution" --apply
```

**在同一 DDK 终端回收并对比**。板端执行完成后运行。诊断图里的数值超阈值会使 recover 返回非零，取得实际输出后仍可生成三端报告。

```bash
conda activate custom_pypto
python "$TOOLS/validate_draft_vector.py" --recover "$G/execution" --output "$G/recovery"
python "$TOOLS/gdr_prefill_stages.py" compare \
  --baseline "$PAIR" --reference "$G/reference" --trace "$G" --output "$G/dump"
```

`dump/l0/{cpu,310p,cpro}/mul_output.bin` 等文件按后端保留实际 dtype，统一为清单中声明的逻辑形状。这些 .bin 按 C 连续顺序写入，无文件头。CPro 使用回读包中的第一次调用，校验各次调用有效区的稳定性。CPU 保留 64 行输入投影，仅推进 19 步状态；逐 token 计算输出保留 19 行，两个设备的完整输出保留 64 行。`dump/summary.csv` 对所有浮点张量报告三组两两比较，分别给出 relative L2、cosine、最大绝对误差、RMSE、KL 和平均相对误差，也统计参考 FP16 次正规数在对端变为零的个数。逐 token 张量分别统计 0–18 和 token 18。状态和 gamma 使用全张量，effective_length 按整数精确比较。

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
