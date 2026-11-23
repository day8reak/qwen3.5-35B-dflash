# 单测 attention 的一个步骤

在 78 和 35 的仓库根目录分别执行 `git pull --ff-only origin main`。沿用原有的 310P、DDK 和 HDC 环境变量。下面两条命令都只测试 L0 的 `F.silu(conv_out)`。

```bash
# 78 · 截取所选步骤的 AIR，保存实测 310P 输入和同输入的 CPU、310P 输出
bash targets/cpro/scripts/conv_silu.sh prepare --step qkv_conv_silu

# 35 · 拉取这份数据，编译所选 OMC，上板并打印 CPU、310P、CPro 三端矩阵
bash targets/cpro/scripts/conv_silu.sh board --step qkv_conv_silu
```

换模块时，两台机器填同一个 `--step`。省略时默认 `qkv_conv_silu`。

| L0 的 `--step` | 独立 OMC 的输入和输出 |
|---|---|
| `qkv_conv_silu` | 原始卷积输出 → `F.silu(conv_out)`，到 Q/K/V 拆分前 |
| `gate_silu` | 原始门控张量 z → `SiLU(z)` |
| `gated_rmsnorm` | GDR core → 用 L0 的 gamma、epsilon 算出的 RMSNorm 输出，到门控相乘前 |
| `out_proj` | 门控乘积转换精度后的投影输入 → 量化输出投影结果 |

`prepare` 再次执行会校验并复用已准备的数据。每次 `board` 默认新建运行目录，完整指标写入终端打印的 `matrices.csv`。续跑时在同一条 `board` 命令后加上之前打印的运行目录。原 attention AIR 在自定义位置时，在 78 设置 `ATTENTION_AIR`。两机的运行根目录不同时，在 35 设置 `AIR_ON_78` 为 78 已准备的 `air/` 绝对路径。

## 查看实际命令和板端 runner

`conv_silu.sh` 默认把展开后的命令打印到终端。78 的 `prepare` 会显示源 AIR 的生成或复用，以及 Python 截取的输入边、输出边和 AIR 路径。35 的 `board` 会显示 OMG 的工作目录、完整参数、HDC 上传和回收命令，以及发给板端的 `launch.sh` 内容。复用已有结果会显示对应的记录路径。

单步测试使用仓库的 `qwen35_tensor_probe`。它以 `tensor-probe` 为文件名随模型上传，按 `plan.txt` 设置输入，调用 HIAI 同步推理。普通模式和 `--cpu-silu` 共用这个 runner。

已有运行可以直接在 35 查看保存的记录。把 `R` 设成终端打印的 board run 目录：

```bash
cat "$R"/omc/models/*/omc/compile-command.json
cat "$R/execution/launch.sh"
ls "$R/execution/calls" "$R/recovery/calls"
```

## 用同一份输入在板端 CPU 上对照 SiLU

在 35 运行。脚本先查 35，再查 78。门控 SiLU 可直接复用旧三算子目录 `$AI_RUN_DIR/target/tests/gdn-op-matrix-l0-valid19-01/air`，只取该步的 AIR、输入和参考输出。自定义路径可在 35 设置 `AIR_ON_35` 或 `AIR_ON_78`。两边都已清理时，先在 78 运行 `bash targets/cpro/scripts/conv_silu.sh prepare --step gate_silu`。卷积分支用 `conv_out.bin`，门控分支用 `tail_z.bin`。

```bash
bash targets/cpro/scripts/conv_silu.sh board --step qkv_conv_silu --cpu-silu
bash targets/cpro/scripts/conv_silu.sh board --step gate_silu --cpu-silu
```

`--cpu-silu` 构建单独的诊断 OMC，沿用原 fixture 的输入类型、形状和字节。该图以 FP32 计算 `x * sigmoid(x)`，按该步骤的输出类型写回。脚本在运行前解码成品并检查两项算术都落在 CPUCL，实际端口类型和执行引擎记录在 `omc/compile.json` 的 `cpu_silu_placement`。这次矩阵的 CPro 行是板端 CPUCL 的结果。

## 查看其余可截取的步骤

```bash
bash targets/cpro/scripts/conv_silu.sh list
```

列表中的 `[prepare/board]` 表示已接入三端同输入对比。`[extract/compile]` 提供独立 AIR 和 OMC，做三端对比还需要该步的实测输入和参考输出。例如截取 L3 的 Q 归一化：

```bash
# 78
bash targets/cpro/scripts/conv_silu.sh extract --layer 3 --step q_norm

# 35
bash targets/cpro/scripts/conv_silu.sh compile --layer 3 --step q_norm
```

用 78 生成的 `slice/slice.json` 核对输入、输出端口和保留节点。两机运行根目录不同时，在 35 设置 `SLICE_ON_78` 为 78 的 `slice/` 绝对路径。各步骤的可截取边界见[步骤表](../../specs/attention-step-presets.csv)。
