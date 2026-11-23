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
