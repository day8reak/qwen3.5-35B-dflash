# 完整 decoder 的三端精度对比

在同一版本的仓库根目录运行。每台机器只执行与自己角色对应的一条命令。以 40 层逐层比较、S=64 为例。

**310P / 导出机（78）**

```bash
bash targets/cpro/scripts/decoder_compare.sh prepare --mode layerwise --phase prefill
```

**DDK / 上板机（35）**

```bash
bash targets/cpro/scripts/decoder_compare.sh board --mode layerwise --phase prefill --recovery
```

| `--mode` | CPro 计算范围 | 每个 phase 的 OMC 数 |
|---|---|---:|
| `linear` | L0 完整 decoder，包括线性 attention、MoE、残差，接原模型最终 norm/head | 1 |
| `full` | 用实测 L3 入口运行 L3 完整 decoder，接最终 norm/head | 1 |
| `4` | L0–L3 连续运行，接最终 norm/head | 1 |
| `40` | L0–L39 连续运行，接最终 norm/head | 1 |
| `layerwise` | 按全局层号加载权重，每个完整 decoder 一个 OMC；L39 内含最终 norm/head | 40 |

`--phase prefill` 使用原 19-token 提示词、物理长度 64、logits 读出位置 18。`--phase decode` 让原 310P 驱动先跑这个 prefill，再捕获第一个生成 token、位置 19、有效长度 1、累计长度 20，以及该次调用每层实际收到的历史状态。两种形状各自导出静态 AIR。小范围的 logits 是在所选范围的末层接最终 head 得到的诊断读数。

只看将要导出的层和图数时，执行 `bash targets/cpro/scripts/decoder_compare.sh plan --mode 4 --phase decode`，普通 Python 3 即可。

## 配置一次

脚本读取已导出的环境变量。也可以在每台主机的 `~/.config/qwen35-cpro/decoder.env` 保存该机的 Bash 变量；如已设置 `XDG_CONFIG_HOME`，配置文件位于其下的 `qwen35-cpro/decoder.env`。可用 `DECODER_ENV` 指定其它文件。权重、DDK 和设备信息由各自的环境提供。

| 主机 | 必需配置 | 可按环境调整 |
|---|---|---|
| 两机 | `AI_RUN_DIR`，本机可写的绝对路径 | `--run DIR` 或 `DECODER_RUN` 选择该根下的批次 |
| 78 | `TARGET_SOURCE` 指向含 `inference_segment.py`、`inference.py` 的原 310P 工程；`CPRO_SOURCE` 指向原 CPro 工程。两个 YAML 指向同一模型及 W4 权重 | `NZ_CONFIG`、`TARGET_CONFIG` 默认分别为源码下的 `config/qwen3_5_moe.yaml`。两个 Python 环境默认是 `qwen3_5_w4`、`qwen3_5_cpro`，可用 `DECODER_NZ_ENV`、`DECODER_EXPORT_ENV` 改名 |
| 35 | 加载目标 DDK 环境，设置 `CPRO_DDK_ROOT`、`BIP`（HDC 服务端 host:port）、`SN`。DDK 中已安装 FP16 → FP16 的 SiluPatch、SoftplusPatch 及原模型算子 | `HDC`、`CPRO_BOARD_TOOL`、`DECODER_BOARD_ENV`（默认 `custom_pypto`）、`RUNTIME_CONFIG`（默认 `$AI_RUN_DIR/runtime/runtime.json`）、`TBE_DIAGNOSIS`（默认 `$AI_RUN_DIR/common/tbe-diagnosis`） |

已激活合适的 Python 环境时，其环境名也可设为 `current`。35 默认用设备上的 `/data/local/tmp/tools_sysdbg/model_run_tool_internal` 的 HIAI V2 接口。使用不同路径时把 `CPRO_BOARD_TOOL` 设为设备上工具的绝对路径，其所需库应能从工具目录或系统库目录加载。

35 从已配置的 SSH `air` 端点读取 78 的产物。首次可在 35 用 `python targets/cpro/scripts/pull_artifacts.py --init AIR主机名 OMC主机名` 建立私有连接配置。如果两台机器的批次路径不同，在 35 的命令末加 `--remote-run /78上的批次绝对路径`。共享文件系统可直接使用同一批次目录。DDK 和 HDC 的初始化见[环境配置](environment-setup.md)。

## 输入、输出和续跑

- `40 + prefill` 复用原 CPro wrapper 和已有 `aligned40` 入口。CPro 接收原 INT8 embedding 和 FP32 scale。报告记录反量化到 FP16 后与 310P 层入口的差异。默认沿用 `ALIGNED40_RUN` 或原批次目录。参考数据保存在批次根目录，新导出的整图、OMC 及 `replay-*/matrices.csv` 和 `comparison.json` 存在其 `conv2d/` 子目录。[整网图说明](target-prefill40-single.md)
- 其它组合从同一个实测 310P decoder 入口 hidden 和每层实测输入状态起步。CPU 和 CPro 各自连续计算。逐层模式直接将上一张 OMC 的板端 hidden 文件作为下一张 OMC 的输入。Prefill 在原 310P 两段衔接处对齐其 padding 处理。Decode 的 `package/fixture/reference.json` 记录实际输入 token 和每份输入 cache 是否与前一次 prefill 的返回值逐字节一致。
- 每个完整 decoder 都保留原 attention、MoE 及残差。CPro 在原源码临时副本中应用已核对的 W4 scale、RoPE、专家量化和卷积历史更新修复，使用 FP16 的两种自定义激活，在 GDN 需要 FP32 的乘法前升精度。导出时检查 W4/GDR 端口、GDR 的 chunk size 及激活的实际图节点和原生形状。
- 除原 `aligned40` 目录外，默认批次为 `$AI_RUN_DIR/target/tests/decoder-<mode>-<phase>`。35 的 `board/matrices.csv` 给出 CPU、310P、CPro 三端的 3×3 指标矩阵，行是 actual，列是 reference。`board/comparison.json` 给出 top1 token ID、文本及未达精度阈值的张量。逐层矩阵也保存在 `board/Lxx/matrices.csv`。
- `hidden_i` 是第 i 层最终残差；单层图还返回 `out_proj_i`，即进入第一次残差相加的 attention 输出。Prefill 比较前 19 行，decode 比较当前行。线性层比较整份 `conv_state_i`、`recurrent_state_i`。Full 层比较 K/V 的有效写入位置，并检查物理写入范围外的槽位是否逐字节保持。静态 T64 可写前两页；S1 本次只写第 0 页的 offset 19。
- 逐层执行中断可在 35 原样重跑上面的 `--recovery` 命令。脚本核对编译产物、板卡、输入及已回收输出的哈希，复用已经完成的层。78 重新执行相同的 `prepare` 会复用完整参考和已导出的层。变更权重、DDK、自定义算子或模型代码后，为新实验指定新的 `--run`。
- 35 下载前显示实际 AIR 包大小和所在文件系统的可用空间。OMG 还需要写 OMC、编译后权重和临时文件。仅凭 69 GiB 可用容量无法确定编译峰值；整网 prefill 入口会打印按当前包估算的剩余空间。
