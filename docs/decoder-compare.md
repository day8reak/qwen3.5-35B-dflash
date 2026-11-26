# Decoder 的 CPU、310P、CPro 精度对比

四个执行环境使用同一版代码。在仓库根目录运行以下脚本。环境可以独立部署，也可以两两同机。每台主机的个人配置、可共享安装和传输端点见[环境配置](environment-setup.md)。

## 两台机器各执行一次

在 CPU、310P golden 和 CPro AIR 共用的服务器执行

```bash
bash targets/cpro/scripts/decoder_prepare.sh
```

成功后，在 CPro OMC 生成和板端执行共用的服务器执行

```bash
bash targets/cpro/scripts/decoder_run.sh
```

第一条依次调用 `decoder_golden.sh`、`decoder_air.sh`。第二条依次调用 `decoder_transfer.sh air`、`decoder_omc.sh`、`decoder_board.sh`。同机的两个角色使用相同的 case 目录。任何一步失败都会停止。可单独重跑失败的入口。

## 四个环境独立部署

| 顺序 | 执行服务器 | 在该机运行，工作目录为仓库根目录 |
|---|---|---|
| 1 | CPU、310P golden 生成服务器 | `bash targets/cpro/scripts/decoder_golden.sh` |
| 2 | CPro AIR 生成服务器 | `bash targets/cpro/scripts/decoder_transfer.sh golden` 成功后运行 `bash targets/cpro/scripts/decoder_air.sh` |
| 3 | CPro OMC 生成服务器 | `bash targets/cpro/scripts/decoder_transfer.sh air` 成功后运行 `bash targets/cpro/scripts/decoder_omc.sh` |
| 4 | CPro 板端执行服务器 | `bash targets/cpro/scripts/decoder_transfer.sh omc` 成功后运行 `bash targets/cpro/scripts/decoder_board.sh` |

传输脚本在**接收方**运行。参数填上游的产物类型。按生产端清单接收文件，字节数齐全后发布到个人 case 目录。服务器之间和板端收发都跳过逐文件哈希校验。使用共享的同一个 case 时可直接执行下一阶段；不同挂载位置可传 `--local-source /挂载点/40-prefill-t64`。OMC 交接包含编译模型、外置权重、输入、参考输出和端口描述。运行时使用已有 OMC，无需在上板机安装 OMG。

## 选择层数、阶段和输入

配置 `DECODER_MODE`、`DECODER_PHASE`，或给各入口传相同的 `--mode`、`--phase`。预览当前选择可运行 `bash targets/cpro/scripts/decoder_compare.sh plan`。

| 模式 | OMC 数 | 计算范围 | Prefill 目录 |
|---|---:|---|---|
| `linear` | 1 | L0 完整 linear decoder 和最终 norm/head | `1-linear-prefill-t64` |
| `full` | 1 | 从实测 L3 入口运行完整 full decoder 和最终 norm/head | `1-full-prefill-t64` |
| `4` | 1 | L0–L3 连续执行，再接最终 norm/head | `4-prefill-t64` |
| `40` | 1 | L0–L39 连续执行，再接最终 norm/head | `40-prefill-t64` |
| `layerwise` | 40 | 每层一张完整 decoder OMC，层间传递板端 hidden，L39 接最终 norm/head | `40x1-prefill-t64` |

`prefill` 使用 T64，`decode` 使用 T1，目录后缀换为 `decode-t1`。在 golden 端编辑 `NZ_CONFIG` 的 `prompts[0]`。310P 使用原 chat template 和 tokenizer，格式化后长度 N 支持 1–64。Prefill 对比前 N 行，有效输出 logits 的位置为 N−1。Decode 取实际 prefill 的预测 token 和缓存，运行位置 N 的一步。

## 对照 GDN 卷积和自定义源码

默认 `--gdn-conv conv2d`。要在整图绕开 Conv2D，两个组合脚本都加 `--mode 40 --phase prefill --gdn-conv taps`，或在各角色配置 `DECODER_GDN_CONV=taps`。CPro 用原权重顺序对四个窗口切片逐通道乘加，FP32 累加后转回卷积权重 dtype，再进入 FP16 SiluPatch。40 层 AIR 导出时检查 30 处卷积 SiLU 各自有四个乘积输入，并检查 AIR 中卷积节点数为零。源文件在 `export/source-taps/models/` 下可供 review。对照的目标是验证整图能否加载并比较输出。板端记录的 `OpUpdateKernelArgs failed` 发生在模型初始化，日志给出的名字是 `SubGraph_0`，具体算子仍待验证。

`CPRO_PATCHED_SOURCE` 指向个人可编辑目录。首次可在 AIR 生成环境执行 `bash targets/cpro/scripts/decoder_compare.sh source --patched-source /实际路径/cpro-edit`。将同一路径写入 AIR 环境的配置，之后可以修改该目录的 Python 文件直接运行。目录应位于会被覆盖的 case 之外。使用原有兼容源码目录时配置相应的 `TARGET_CONFIG`；自动生成的补丁和 `source.diff` 可用于 review。自定义 taps 对照会匹配现有深度卷积表达式，手改过这段表达式时会提示对应文件和行号。

## 输出与重跑

默认 case 父目录为 `$AI_RUN_DIR/target/tests`。同名 case 由 `golden` 重新创建，`air` 保留 golden 并重新导出，`compile` 重新生成 `board/`，`execute` 重新运行板端对比。普通运行覆盖同名结果。逐层模式可以给执行脚本加 `--mode layerwise --recovery` 接着跑，它会核对模型、板卡和输入输出。需要保留两个版本做 A/B 时，先归档前一份结果，或为两次选择不同的 `--run-root`。

- `board/matrices.csv` 包含三端的 3×3 矩阵，行是 actual，列是 reference。`board/comparison.json` 包含 prompt、logits 位置、预测 token 和未达阈值的张量。逐层模式还有 `board/Lxx/matrices.csv`。
- 线性层比较完整 conv/recurrent 状态，full 层比较有效 K/V。逐层模式另外检查写入区以外的 KV。单层和四层的 logits 取所测最后一层输出并接原模型 head。
- 整网 T64 沿用原 CPro wrapper 和量化 embedding。其它组合从实测入口 hidden 和状态开始。W4 scale、RoPE、MoE、GDN、FP16 SiluPatch 和 SoftplusPatch 的整网检查见[原 CPro 整网 prefill](target-prefill40-single.md)。
- 上板日志标出传输、执行、回收和清理步骤。一次 `hdc file send` 发送模型、外置权重与输入，一次 `hdc file recv` 拉回输出目录。收到并验证各文件后保存 `result.json`，再清理本次板端临时目录。执行或回收失败时，从回执的 `remote` 字段找到保留的现场。
