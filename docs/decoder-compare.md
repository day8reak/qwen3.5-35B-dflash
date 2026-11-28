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

### 按单层已通过的配置跑四层 T64 prefill

两台服务器先同步到同一版代码。78 采集本次 prompt 的四层 CPU、310P golden 并导出一个 L0–L3 AIR。

```bash
bash targets/cpro/scripts/decoder_prepare.sh --mode 4 --phase prefill --gdn-conv conv2d --rmsnorm-patch \
  --run-root "${AI_RUN_DIR:?}/target/tests/qlinear-scale-fp16"
```

78 完成后，在 35 接收该包，编译一个 OMC 并上板。

```bash
bash targets/cpro/scripts/decoder_run.sh --mode 4 --phase prefill --gdn-conv conv2d --rmsnorm-patch \
  --run-root "${AI_RUN_DIR:?}/target/tests/qlinear-scale-fp16" \
  --remote-run "${AI_RUN_DIR:?}/target/tests/qlinear-scale-fp16/4-prefill-t64" \
  --replace-weights --replace-inputs
```

三端共用 310P wrapper 实际收到的 INT8 embedding 编码和逐 token FP32 scale，先相乘再舍入到 FP16。78 检查 CPU 重建和送板操作数重建与 310P 实测 L0 入口全部 64 行逐元素相同，并检查 CPro AIR 中从两个外部输入到首个 RMSNorm 的路径。四层导出检查 36 个 W4 matmul 的分组权重 scale、4 个共享门控以及 3 个 GDR。35 在 `4-prefill-t64/board/comparison.json` 和 `matrices.csv` 输出四层 logits、L0–L2 的六个状态和 L3 的 K/V 三端对比。

### 用相同的算子配置跑 40 层单块 prefill

78 端按当前 prompt 准备 40 层 CPU、310P golden，导出一张 L0–L39 的 T64 AIR。三端走 wrapper 的 token ID 查表分支。QLinear 的 W4 权重 scale 和共享专家门控沿用已验证的 FP16 设置，GDN 的 `out_proj` 直接送入残差加法。

Golden 独立保存在 78 的 `$AI_RUN_DIR/target/golden/40-prefill-t64/<版本>/`。其中 `export/reference/` 是 310P 的输入、输出和 token 对照表，`export/L00/cpu/` 至 `export/L39/cpu/` 是逐层 CPU 结果。`binding.json` 记录 prompt、NZ 配置、参考实现、运行库版本及 checkpoint 文件信息。首次生成后，同一配置再次运行下方的准备命令会打印 `[golden] 复用`，直接把已保存的参考文件交给 AIR 导出。仅修改 CPro 源码或更换实验的 `--run-root` 都能复用。更换 prompt 或 checkpoint 会保存另一版。

如果在更新代码前已经跑完 40 层 golden，在重新运行准备命令前，先在 78 归档当前 case 一次。旧清单记录的是权重路径，使用这条命令时应确认原权重和参考计算环境仍与那次运行一致。脚本检查旧采集、40 份 CPU 清单及其与 310P 输入的对应关系。

```bash
bash targets/cpro/scripts/decoder_golden.sh --mode 40 --phase prefill --cache-existing-golden \
  --run-root "${AI_RUN_DIR:?}/target/tests/qlinear-scale-fp16"
```

首次生成和以后复用均执行

```bash
bash targets/cpro/scripts/decoder_prepare.sh --mode 40 --phase prefill --gdn-conv conv2d --rmsnorm-patch \
  --run-root "${AI_RUN_DIR:?}/target/tests/qlinear-scale-fp16"
```

35 端接收整网 AIR，编译一个 OMC 并上板执行一次。编译盘需要同时容纳 AIR 包、OMC 外置权重和 OMG 临时文件。

```bash
bash targets/cpro/scripts/decoder_run.sh --mode 40 --phase prefill --gdn-conv conv2d --rmsnorm-patch \
  --run-root "${AI_RUN_DIR:?}/target/tests/qlinear-scale-fp16" \
  --remote-run "${AI_RUN_DIR:?}/target/tests/qlinear-scale-fp16/40-prefill-t64" --replace-inputs --replace-weights
```

查看 `40-prefill-t64/board/comparison.json`。其中 `output_pos` 是最后一个有效 token 的下标，`predictions` 是三端预测的首 token，`cpro_vs_310p.logits` 是该位置的 logits 对比。`failed` 列出超限张量，其中包括缓存。

这次从空缓存执行单个 T64 块。每个 GDN 层用旧缓存和本层 QKV 拼出的因果卷积窗口计算当前 hidden，同时把 QKV 物理末四行写入本层返回的 `conv_state`。当前 logits 的计算不回读这份新缓存；下一块 prefill 或 decode 才使用它。作为对照，已有四层 valid19 的 310P 采集逐元素确认 L0–L2 的缓存取行 60–63。对应 CPro dump 记录在四层 case 的 `board/L00-L03/operator-dump/attempt-*/L00-L03/replay/gdn-padding.json`。

### 用原始源码跑四层 T64 prefill

在 AIR 生成服务器的个人配置（`air.env` 或共用的 `decoder.env`）中，将 `CPRO_SOURCE` 设为自己的完整 CPro 工程根目录，该目录下应能看到 `export_glm_model_qwen3_5_moe.py`。模型实现可直接改 `$CPRO_SOURCE/models/modeling_qwen3_5_moe_hiai_cpro.py`，修改算子注册或 GE 转换时再改根目录的导出脚本。目录放在会被重建的 case 之外。手改源码也可以使用 `--no-patch`，该参数会运行你指定的源码现状。已有四层 golden 时，两台服务器各运行一条命令。

在 CPro AIR 生成服务器运行

```bash
bash targets/cpro/scripts/decoder_air.sh --mode 4 --phase prefill --no-patch
```

在 CPro OMC 生成和板端执行共用的服务器运行

```bash
bash targets/cpro/scripts/decoder_run.sh --mode 4 --phase prefill --no-patch
```

`--no-patch` 直接执行 `$CPRO_SOURCE/export_glm_model_qwen3_5_moe.py`。导出日志打印实际入口以及 AIR 中 `SiluPatch`、`SoftplusPatch` 和 `QuantBatchMatmulV4444` 的数量。使用未改动的原始模型时，前两项应为零。脚本创建四层 HF 配置副本，继续传入 `quant_mode=enable quant_bits=4 seq_len=64`。这次实验沿用 `--gdn-conv original` 的输入、静态 T64 OMG 参数和 V2 上板参数。同名 `4-prefill-t64` 目录保存本次产物，成功执行后在 `board/matrices.csv` 查看三端精度。首次生成四层 golden 时，第一条命令使用 `decoder_prepare.sh`。

### 只撤销四层的自定义激活替换

已有四层 prefill golden 时，在 AIR 生成服务器执行

```bash
bash targets/cpro/scripts/decoder_air.sh --mode 4 --phase prefill --gdn-conv conv2d --native-activations
```

在 OMC 生成和板端执行服务器执行

```bash
bash targets/cpro/scripts/decoder_run.sh --mode 4 --phase prefill --gdn-conv conv2d --native-activations
```

此选项以对齐后的 CPro 源码为基础，在本次导出的副本上恢复原生 SiLU 和 Softplus。W4 分组权重 scale 使用 FP16，RoPE、GDR、MoE 的对齐修改继续生效。导出时打印并检查 `SiluPatch=0` 和 `SoftplusPatch=0`，同时检查 W4 和 GDR 端口。两种四层原生卷积对照都从相同的 golden、embedding 权重和 YAML 配置生成输入。原 CPro wrapper 接收 INT8 embedding 及逐 token FP32 scale。普通分层诊断图和 T1 decoder 图接收 FP16 hidden。原生卷积还接收 128 个 INT32 的 `conv_state_pos`，默认首项为有效长度、其余项为零。可用 `DECODER_CONV_CONTROL` 指定原板端 bin。

同一运行目录中保留了成功的 `--no-patch` 记录时，板端执行服务器会先保存其输入槽位，再逐项对照本次的角色、shape、dtype 和已有 bin 内容标识。该对照使用服务器上的清单与执行记录。

上面两条命令在原 ConvState 更新之后显式调用分组 `F.conv2d`。L0、L1、L2 各导出一个 Conv2D，卷积窗口为 `[1,8192,1,67]`，卷积核为 `[8192,1,1,4]`。三种激活对照都从 `CPRO_ORIGINAL_PATCHED_SOURCE`（或自动生成的原 ConvState 源码）导出，使用相同的四层配置、OMG 参数和 V2 输入接口。省略 `--gdn-conv` 会沿用此前的 `original` 卷积配置。

### 四层只启用 SiluPatch

以同样指定 `--gdn-conv conv2d` 的 `--native-activations` 四层 T64 结果为基线。复用同一份 golden、CPro 源码、权重和 ConvState 控制设置。在 AIR 生成服务器执行

```bash
bash targets/cpro/scripts/decoder_air.sh --mode 4 --phase prefill --gdn-conv conv2d --silu-only
```

在 OMC 生成和板端执行服务器执行

```bash
bash targets/cpro/scripts/decoder_run.sh --mode 4 --phase prefill --gdn-conv conv2d --silu-only
```

模型使用 FP16 `SiluPatch` 和原生 `F.softplus`。导出检查 14 个 `SiluPatch` 和 0 个 `SoftplusPatch`，包括 L0–L2 的 6 个卷积与门控激活，以及四层 MoE 的 8 个激活。在同一用例目录保留了基线的成功执行记录时，运行器还会对照两次运行的输入槽位。

四个调用位置均直接使用 `torch.ops.npu.silu_patch`，按上游 dtype 传入连续张量，算子返回同形状的 FP16 张量。GDN 卷积的激活以 FP16 送入 Q/K/V 分支。RMS 门控、共享专家和路由专家直接使用 SiLU 的 FP16 输出参与乘法。GDN 门控保留原模型末尾的 `return out.to(input_dtype)`。`export/package/L00-L03/source.diff` 可查看 `--silu-only` 或 `--physical-convstate` 实际采用的修改。

V2 执行完毕后，终端以 `PASSED` 或 `FAILED_ACCURACY` 打印 `comparison.json` 的路径。精度未达标时会列出未通过项并以状态码 1 退出。加载或执行阶段的异常仍可从该阶段的报错和 `replay/result.json` 定位。

### 四层只启用 SoftplusPatch

在 AIR 生成服务器执行

```bash
bash targets/cpro/scripts/decoder_air.sh --mode 4 --phase prefill --gdn-conv conv2d --softplus-only
```

在 OMC 生成和板端执行服务器执行

```bash
bash targets/cpro/scripts/decoder_run.sh --mode 4 --phase prefill --gdn-conv conv2d --softplus-only
```

SiLU 使用原生实现。导出检查 0 个 `SiluPatch`、3 个 `SoftplusPatch`，L0–L2 的 GDN 衰减分支各一处。`a.float() + dt_bias` 的结果直接进入 `SoftplusPatch`，其 FP16 输出直接与 `-exp(A_log)` 相乘。三处激活的输入和输出形状均为 `[1,64,32]`。

对照三次运行时固定源码、Conv2D、权重、输入和编译环境。若原生激活和 `--softplus-only` 均能加载执行，只有 `--silu-only` 在 `UpdateArgsForAicore` 处失败，就优先排查引入 SiluPatch 后的编译任务及其参数。加载阶段还没有产生可比较的推理输出。

### 四层 SoftplusPatch 配合物理尾部 ConvState 写回

在已有四层 golden 的 AIR 生成服务器执行

```bash
bash targets/cpro/scripts/decoder_air.sh --mode 4 --phase prefill --gdn-conv conv2d --softplus-only --physical-convstate
```

在 OMC 生成和板端执行服务器执行

```bash
bash targets/cpro/scripts/decoder_run.sh --mode 4 --phase prefill --gdn-conv conv2d --softplus-only --physical-convstate
```

这组命令从相同的对齐源码导出一张四层图。与上一组 softplus-only 相比，新增的 `--physical-convstate` 将 L0–L2 的 `linear_attention_convstate_update` 调用替换为 `torch.cat` 和切片。拼接后的 68 行中，末 4 行写回状态，末 67 行送 Conv2D；四层 AIR 接口随之去掉 `conv_state_pos`。有效长度仍传给 GDR。导出日志列出原更新算子数（预期 0）和显式拼接的缓存数（预期 3）。

要在这个已执行的配置上单独观察启用自定义 SiLU 对模型加载的影响，两端只去掉 `--softplus-only`。默认的两种自定义激活都会启用，预期 AIR 含 14 个 `SiluPatch` 和 3 个 `SoftplusPatch`。两次都用相同的四层 golden、Conv2D 和物理尾部 ConvState。若板端生成了矩阵，即已完成加载与推理；`FAILED_ACCURACY` 单独记录精度结果。

默认对齐源码对原 CPro 的其他修改如下。

| 位置 | 修改 |
|---|---|
| L3 full attention 的 RoPE | 用 INT32 `position_ids` 取 cos/sin 表中的行，Q/K 旋转在 FP32 计算，结果转回 hidden 的 dtype。 |
| A8W4 投影 | 导出器使用 CPro 的 `QLinear`，激活为 INT8，W4 权重打包在 INT32 中。权重 scale 使用原加载器的 dtype，逐 token scale 走原 TorchAir converter。单层的 `case.json` 记录每个 W4 matmul 在 AIR 中实际收到的 dtype。 |
| GDN 和激活 | 默认对齐导出使用 Conv2D、SiluPatch 和 SoftplusPatch。自定义激活直接接收上游张量。Softplus 的偏置加法使用 FP32，下游乘法由两个操作数的 dtype 决定。激活 kernel 的注册类型为 FP16。GDR 使用 T64 chunk、FP16 的 Q/K/V、g、beta 和状态，`effective_length` 为 INT16。 |
| MoE | 专家分支使用实际量化后的 INT8 张量进行 reshape，已量化输入直接使用。路由和共享专家门控按对应权重及输出 dtype 运算。 |

导出日志打印实际命令和源码对照目录。导出完成后可在 `export/package/L00-L03/source.diff` 查看导出器、模型、wrapper 和 `utils.py` 相对配置目录的逐行差异。若配置目录本身已手改，差异以该目录当前的源码为准。

### 四层整图只在 GDN 使用 SiluPatch

要定位四层整图的加载问题，可以给两端都传 `--gdn-silu-only`。L0–L2 的 GDN 在卷积后和 RMS 门控处各调用一次 `torch.ops.npu.silu_patch`。L0–L3 的 MoE 共享专家和路由专家都使用原 CPro 的 `self.act_fn`。这组命令沿用物理尾部 ConvState、Conv2D、GDN 衰减分支的 SoftplusPatch 和已有的四层 golden。

在 AIR 生成服务器执行

```bash
bash targets/cpro/scripts/decoder_air.sh --mode 4 --phase prefill --gdn-conv conv2d --physical-convstate --gdn-silu-only
```

导出器按形状检查 6 个 `SiluPatch`（卷积后 `[1,8192,64]` 和 RMS 门控 `[2048,128]` 各 3 个）以及 3 个 `SoftplusPatch`（`[1,64,32]`）。两种自定义算子的输入输出都是 FP16。

在 OMC 生成和板端执行服务器执行

```bash
bash targets/cpro/scripts/decoder_run.sh --mode 4 --phase prefill --gdn-conv conv2d --physical-convstate --gdn-silu-only
```

生成三端矩阵说明这张 OMC 完成了加载和推理。若只有启用 MoE 自定义 SiLU 的整图加载失败，可进一步分别隔离共享专家和路由专家。若这张图仍在 `UpdateArgsForAicore` 失败，则保留 GDN 的调用就能复现该加载现象。

要区分 GDN 内的两个调用位置，两台服务器都把上述命令中的 `--gdn-silu-only` 换成同一个选项。仍使用已有 golden 和一张四层 OMC。

| 选项 | 使用 SiluPatch 的位置与 AIR 预期 | 另一处的计算 |
|---|---|---|
| `--gdn-silu-site conv` | L0–L2 卷积后，各 1 个 `[1,8192,64]` | RMS 门控恢复 `F.silu(gate.to(torch.float32))` |
| `--gdn-silu-site gate` | L0–L2 RMS 门控，各 1 个 `[2048,128]` | 卷积后使用原生 SiLU，Q/K/V 分支接收 FP16 |

两个选项均保留 3 个 `SoftplusPatch` 和 MoE 的原生 SiLU。上板日志打印本次编译的 `conv` 或 `gate` 选择，AIR 导出时按形状核对调用位置。若只有一组加载失败，该位置及其相邻图值得优先排查。若两组分别能加载，同时启用两处才失败，则需要对照组合图的编译任务。

### 四层启用全部 SiluPatch，去掉 GDN 门控输出的显式 FP32 转换

在上面两条命令中都把 `--gdn-silu-only` 换成 `--all-silu-no-gate-cast`。导出器核对 14 个 FP16 `SiluPatch`，其中 GDN 卷积后和 RMS 门控各 3 个，MoE 共享专家和路由专家各 4 个，同时核对 3 个 FP16 `SoftplusPatch`。GDN 门控默认就使用 `torch.ops.npu.silu_patch(gate.contiguous())` 的返回值。乘法两侧 dtype 不同时，导出器仍可能按 PyTorch 的类型提升规则生成 Cast。导出日志的三行 `[GDN 门控 AIR]` 显示每个门控输出在 AIR 中的直接下游节点，`export/package/L00-L03/case.json` 的 `silu_patch.nodes[].consumers` 也记录该信息。

### 从四层 AIR 定位 RMS 门控加载失败

在能访问已有四层 AIR、OMG 和板卡的服务器上运行：

```bash
bash targets/cpro/scripts/decoder_compare.sh gate-probe
```

命令从当前 `4-prefill-t64` 的 AIR 中取第一个形状为 `[2048,128]` 的 RMS 门控，复制实际节点及其属性，沿用四层原生 OMG 参数和 HIAI V2。四张小 OMC 分别为 `cast-mul`（从 SiLU 输出到 Mul，包含源图在两者之间的 Cast 或视图）、`silu`、`silu-cast` 和 `silu-cast-mul`。切口使用固定的合成浮点输入。这个实验测加载和执行，不衡量模型精度。

`board/L00-L03/gate-probes/result.json` 汇总每项状态。定位加载问题时查看 `LOAD_MODEL_FAILED` 项的 HiLog，确认是否复现同一条 `OpUpdateKernelArgs failed`。如果四项都为 `EXECUTED`，说明该门控与这两个下游节点单独组合可以运行，后续应检查它们在整图中的上游和编译任务。四层模型及其 replay 记录保留在原目录。

四项均执行成功后，在保留这些产物的 OMC 生成服务器运行以下命令。它从已有文件打印四层的各条门控输入链、乘法另一侧的来源，以及成功小图的相应连线。报告按节点名列出 OMG 的 `RuntimeConfig` 记录，按门控形状对照 `BuildSingleOp` 的端口字段，并列出本次编译目录里保留的同名 kernel JSON 参数。整图中多个节点可能共用同一形状，具体失败的板端任务仍需有任务标识的服务端日志确认。

```bash
bash targets/cpro/scripts/decoder_compare.sh gate-report
```

报告还用 `omg --mode=1` 分别解析两份已经上板的 OMC。在 `OMC 连线` 段，每个可见子图按自己的节点名解析输出 0，列出引用它的消费者输入槽、图输出以及生产者的 `dst_name` 记录，并打印原 Cast 和 Mul 同名节点的实际输入。若只在子图的 `NetOutput` 看到引用，表示该值到达这个子图的出口。若 OMG 只暴露外层 `GraphOp`，内部连线的状态仍需从编译阶段图定位。

`[AIR shape]` 段分别读取卷积 SiLU 和 RMS 门控 SiLU 的输入、输出及直接消费端。每个端口的 `ge_dims` 来自 `TensorDescriptor.shape.dim`，`fx_dims` 来自 `_meta` 字符串，`fx_tensor` 是 FX 张量名。对照卷积端口、下游转置的输入输出和同次 OMG 的 `BuildSingleOp` 记录，可以定位 `[1,8192,64]` 与 `[1,64,8192]` 首次出现差异的位置。单独的导出器 shape trace 只记录了自定义算子的输入。

若 `silu-cast-mul` 在板上执行成功，可以在同一服务器接回第一处 RMS 门控的原始上游，复用该结果作为两侧均从 bin 输入的基线。

```bash
bash targets/cpro/scripts/decoder_compare.sh gate-upstream
```

三张 OMC 都以源 AIR 中的同一个门控乘积为输出。`z-branch` 保留原 `in_proj_z` 的 DynamicQuant、W4 投影、Reshape 和权重，乘法的 RMS 侧直接读取 bin。`rms-branch` 保留原 RMSNorm 和 gamma，SiLU 输入直接读取 bin。`both-branches` 同时保留两侧。切口使用固定合成输入验证加载和执行。结果在 `board/L00-L03/gate-upstream/result.json`。比较四层故障时，核对失败阶段和该次 HiLog 的 `OpUpdateKernelArgs failed`。

### 在四层整图按层选择 RMS 门控 SiLU

`gate-upstream` 的三项均为 `EXECUTED` 时，可以在原四层 T64 导出和上板流程中使用 `--rms-silu-layers`。先用 `none` 让三个 RMS 门控对相同的连续 FP16 输入调用 `F.silu`。此时仍有 11 个 `SiluPatch`（3 个卷积和 8 个 MoE）和 3 个 `SoftplusPatch`。保留了 Conv2D、物理尾部 ConvState 和门控乘法的类型提升。

在 CPro AIR 生成服务器复用已生成的 golden 并重新导出四层 AIR。

```bash
bash targets/cpro/scripts/decoder_air.sh --mode 4 --phase prefill --gdn-conv conv2d --physical-convstate --rms-silu-layers none
```

在合并部署的 OMC 和板端执行服务器接收、编译、上板。两端使用相同的层选择。此命令会覆盖当前四层运行目录中的旧板端结果。

```bash
bash targets/cpro/scripts/decoder_run.sh --mode 4 --phase prefill --gdn-conv conv2d --physical-convstate --rms-silu-layers none
```

如果 `none` 能执行，在两条命令中都改用 `0`，仅将 L0 的 RMS 门控换回 `SiluPatch`。也可选 `1`、`2` 或升序逗号列表来定位单层或组合触发条件。导出时逐层打印门控选择，并核对 AIR 中 RMS 门控形状的自定义节点数。出现精度矩阵或 `FAILED_ACCURACY` 表示模型已执行。复现原加载问题需在该次运行看到 `load_model` 和 `OpUpdateKernelArgs failed`。

### 四层 T64 按原门控形状调用自定义激活

L0、L1、L2 的 RMS 门控直接将 `[2048,128]` 传给 `SiluPatch`，衰减支路将 `[1,64,32]` 传给 `SoftplusPatch`。旧的 `--rms-silu-token-view` 用于选择已经导出过的 `[64,4096]` 四层产物。

门控改动写入导出用源码副本的 `models/modeling_qwen3_5_moe_hiai_cpro.py`，位置是 `Qwen3_5MoeRMSNormGated.forward`。原 CPro 导出脚本通过 TorchAir 将该调用映射为 `SiluPatch`，生成的 AIR 直接交给 OMG。可在本次运行的 `export/package/L00-L03/source.diff` 中查看实际 PyTorch 源码差异。

CPro AIR 生成服务器复用当前四层 golden：

```bash
bash targets/cpro/scripts/decoder_air.sh --mode 4 --phase prefill --gdn-conv conv2d --physical-convstate
```

CPro OMC 生成和板端执行共用的服务器：

```bash
bash targets/cpro/scripts/decoder_run.sh --mode 4 --phase prefill --gdn-conv conv2d --physical-convstate
```

四层包的 `case.json` 记录转换器观测到的激活形状。板端执行后，在 `4-prefill-t64/board/comparison.json` 查看三端精度。

### 导出完整 L0 linear attention

在保留当前四层 T64 编译输入包的 CPro OMC 生成服务器执行以下命令，即可从这份 AIR 直接切出 `layers[0].linear_attn`。

```bash
bash targets/cpro/scripts/decoder_compare.sh gdn-export
```

它沿用源图的 RMS 门控选择、Conv2D 和物理尾部 ConvState。入口为归一化后的 hidden、旧卷积历史、旧递推状态和有效长度。出口是第一条残差加法之前的 out_proj 输出、新卷积历史和新递推状态。切片检查 5 个 W4 投影、1 个 chunk GDR 和 1 个 SoftplusPatch。`--rms-silu-layers 0` 的源图应切出 2 个 SiluPatch，`none` 应为 1 个。

终端打印单层 `export.air` 的完整路径。该文件和它引用的权重保存在 `4-prefill-t64/board/L00-L03/gdn/linear-attention/` 下，移动产物时保留该目录结构。`case.json` 记录实际的输入、输出、权重和源 AIR。本命令只导出，复用当前四层包，无需重新采集 golden。

要在 OMC 和板端执行共用的服务器测试加载和执行，可直接运行：

```bash
bash targets/cpro/scripts/decoder_compare.sh gdn-whole
```

这一条命令包含切图、编译一个 OMC 和 V2 上板。使用固定合成 hidden、零初始缓存和当前四层 fixture 的有效长度。执行状态保存在 `board/L00-L03/gdn/result.json`，适合排查加载故障。早先基于三端真实输入的 GDN 精度对照另见 [GDN 层对比](guides/gdn-layer-t64.md)。

### 从已复现加载错误的 L0 GDN 分段定位

在保留本次 `gdn-whole` 产物或对应四层 T64 物理 ConvState 包的 CPro OMC 生成和板端执行服务器，运行一次：

```bash
bash targets/cpro/scripts/decoder_compare.sh gdn-bisect
```

脚本优先复用 `board/L00-L03/gdn/linear-attention` 的单层 AIR 和权重。如果目前只有四层包，会先在 `board/L00-L03/gdn-bisect-source` 生成完整 L0 对照，并在本批上板。第一层分别测到 GDR 输出为止的前段，以及从 core 和 hidden 到 out_proj 的后段。第二层分开 QKV 投影、卷积到 Q/K/V、衰减、beta、GDR 状态、门控两支和 out_proj。另有跨切口的图保留原来的生产者和消费者。在 L0 RMS 门控使用 SiluPatch 时，`cut-before-silu` 和 `cut-after-silu` 分别把该节点的输入或输出作为 FP16 外部输入，保留其余 GDN 路径及三个原始输出。切片使用合成输入测试 OMC 加载和执行。

结果写入 `board/L00-L03/gdn-bisect/localization.json`。对每张图保留源节点、切口、退出码和诊断路径。`minimal_reproductions` 列出本批按源节点集合包含关系最小、且当前客户端的 HiLog 能关联到 `OpUpdateKernelArgs failed` 的加载失败图。`passing_halves` 列出同一父图复现但两个子图分别执行成功的组合。如果只有完整 GDN 复现，仍需检查在切口两侧一起编译时生成的任务和参数。单项切图、编译或执行异常单独记录，其他有效切片继续上板。

### 用原 CPro 的板端命令重跑已有 OMC

在板端执行服务器上，可以保留现有四层 `taps` OMC 和输入，只改变传给 `model_run_tool_internal` 的参数。脚本按当前 OMC 的槽位绑定输入，并传入 `--model`、`--input`、`--times=1`、`--modelmanager_itf=V2`、外置权重的 `--weight_dir`、`--enable_item=1`、`--input_shape`、`--output_shape` 和 `--output_dir`。加载方式和输入输出类型使用工具默认值。实际命令保存在 `board/L00-L03/replay/run.sh`，退出状态在同目录的 `result.json`。

```bash
bash targets/cpro/scripts/decoder_board.sh --mode 4 --phase prefill --gdn-conv taps --original-board-args
```

### 从原 CPro 源码生成四层或四十层 prefill

`--gdn-conv original` 保留原 CPro wrapper、`LinearAttentionConvstateUpdate` 的 `linear_update_tensor` 输入和模型源码中的 `F.conv1d`。模型继续使用 CPro QLinear 和 FP16 分组权重 scale、RoPE、MoE、GDR 接口以及 FP16 输入输出的 `SiluPatch` 和 `SoftplusPatch`。导出单张图，返回 logits 和每层两个状态。四层有 9 个输出，四十层有 81 个输出。板端使用上一节的 V2 参数。

已有对应 CPU 和 310P golden 时，在 AIR 生成服务器运行第一条，在 OMC 生成和板端执行共用的服务器运行第二条。首次运行或更换输入 prompt 时，第一条改用 `decoder_prepare.sh`。两条命令均可将 `--mode 4` 改为 `--mode 40`。

```bash
bash targets/cpro/scripts/decoder_air.sh --mode 4 --phase prefill --gdn-conv original
bash targets/cpro/scripts/decoder_run.sh --mode 4 --phase prefill --gdn-conv original
```

导出器调用原 `export_glm_model_qwen3_5_moe.py`，传入 `mode=export quant_mode=enable num_hidden_layers=4 seq_len=64 topK=16 config=… quant_bits=4`。构造模型之前，脚本在本次导出目录内复制 HF 配置并选取对应层数，权重链接到配置指定的 checkpoint。可在 AIR 环境设置 `CPRO_ORIGINAL_PATCHED_SOURCE` 生成并使用个人源码副本，或指定已修改的源码目录。生成的副本包含 `source.diff`。原生 ConvState 需要 `INT32[128]`，有对应 prompt 的实测文件时设置 `DECODER_CONV_CONTROL`。默认首元素为有效长度，其余为零；这个规则在 valid19 的语义仍待验证。

需要用私有工程的 bash 入口时，可在 AIR 环境执行下面这条命令生成源码。在私有工程的干净副本中可以用 `git apply /实际路径/cpro-native-edit/source.diff` 应用同一组修改。手动运行原 exporter 需要将此仓库的 `model/` 绝对路径加入 `PYTHONPATH`，并使该次实验私有 HF 配置中的层数与 `num_hidden_layers` 一致。自动入口会处理这两步。

```bash
bash targets/cpro/scripts/decoder_compare.sh source --mode 4 --phase prefill --gdn-conv original --patched-source /实际路径/cpro-native-edit
```

编译使用所选 DDK 的 `tools/tools_ascendc/set_ascendc_env.sh` 与 `tools/tools_omg/omg`，按新 AIR 的实际输入名生成静态 T64 shape，使用外置权重和 `--limit_partition_size=false`。历史四十层命令针对 `40block_55_quant` 使用 64/1 双档。之前的双档试验中，T1 分支的 Reshape 输入为 `[1,1,2048]`，目标为 `[64,2048]`。本次对照只检验 T64 prefill。

### 其他卷积实现

默认 `--gdn-conv conv2d`。要在整图绕开 Conv2D，两个组合脚本都加 `--mode 40 --phase prefill --gdn-conv taps`，或在各角色配置 `DECODER_GDN_CONV=taps`。CPro 用原权重顺序对四个窗口切片逐通道乘加，FP32 累加后转回卷积权重 dtype，再进入 FP16 SiluPatch。40 层 AIR 导出时检查 30 处卷积 SiLU 各自有四个乘积输入，并检查 AIR 中卷积节点数为零。源文件在 `export/source-taps/models/` 下可供 review。对照的目标是验证整图能否加载并比较输出。板端记录的 `OpUpdateKernelArgs failed` 发生在模型初始化，日志给出的名字是 `SubGraph_0`，具体算子仍待验证。

`CPRO_PATCHED_SOURCE` 指向个人可编辑目录。首次可在 AIR 生成环境执行 `bash targets/cpro/scripts/decoder_compare.sh source --patched-source /实际路径/cpro-edit`。将同一路径写入 AIR 环境的配置，之后可以修改该目录的 Python 文件直接运行。目录应位于会被覆盖的 case 之外。使用原有兼容源码目录时配置相应的 `TARGET_CONFIG`；自动生成的补丁和 `source.diff` 可用于 review。自定义 taps 对照会匹配现有深度卷积表达式，手改过这段表达式时会提示对应文件和行号。

## 输出与重跑

默认 case 父目录为 `$AI_RUN_DIR/target/tests`。同名 case 由 `golden` 重新创建，`air` 保留 golden 并重新导出，`compile` 重新生成 `board/`，`execute` 重新运行板端对比。普通运行覆盖同名结果。逐层模式可以给执行脚本加 `--mode layerwise --recovery` 接着跑，它会核对模型、板卡和输入输出。需要保留两个版本做 A/B 时，先归档前一份结果，或为两次选择不同的 `--run-root`。

- `board/matrices.csv` 包含三端的 3×3 矩阵，行是 actual，列是 reference。`board/comparison.json` 包含 prompt、logits 位置、预测 token 和未达阈值的张量。逐层模式还有 `board/Lxx/matrices.csv`。
- 线性层比较完整 conv/recurrent 状态，full 层比较有效 K/V。逐层模式另外检查写入区以外的 KV。单层和四层的 logits 取所测最后一层输出并接原模型 head。
- 四层 T64 的 310P golden 通过 wrapper 的 `if` 分支将 INT8 embedding 行乘以逐 token FP32 scale。两个操作数按实际调用保存到 `export/fixture/embedding_inputs/`，包括 padding 行。CPU 用这两份数据重建 L0 输入，四层 CPro OMC 直接接收这两份数据。四层图返回 logits、前三层的 conv/recurrent state 和 L3 的 K/V，共 9 个输出。
- 40 层 T64 整网推理默认走 wrapper 的 `else` 分支。CPU、310P 和 CPro 都用同一组 token ID 查 checkpoint 的 `embed_tokens`。生成 40 层 golden 时，CPU 查表结果与 310P 实测 L0 入口的 64 行按 FP16 位模式核对。CPro 接收小端 INT32 的 `input_ids.bin`，形状 `[1,64]`，共 256 字节。AIR 导出时沿词表 Gather 的索引输入确定 token ID 节点，记录在 `export/air/manifest.json` 的 `boundary.token_ids_input`。上板时按这个节点名区分同形状的 `position_ids`。已有 40 层 golden 的实验可以在 78 运行 `decoder_air.sh` 重新导出，再在 35 运行 `decoder_run.sh` 编译上板。独立 decoder block 和 T1 对照接收已采集的入口 hidden 与状态。其余算子配置见[原 CPro 整网 prefill](target-prefill40-single.md)。

### 共享专家门控的 L0 整层回归

导出完成后，在用例的 `export/package/air-source/` 查看这次送给 TorchAir 的源码。L0 的 Softplus 调用在 `cpro/models/modeling_qwen3_5_moe_hiai_cpro.py`，算子注册在 `cpro/export_glm_model_qwen3_5_moe.py`，单层的 `DecoderGraph.forward` 和 `dynamo_export` 调用在 `decoder/decoder_case_air.py`。源码作为普通文件随 AIR 包传到 OMC 编译机。40 层原生 prefill 的对应目录是 `export/air/air-source/`。

单层 T64 prefill 将偏置加法的 `[1,64,32]` 结果直接传给 `SoftplusPatch`，输出直接与 `-exp(A_log)` 相乘。可在 `export/package/L00/activations.txt` 查看导出器记录的形状。

如 L0 OMC 编译成功、上板在 `OpUpdateKernelArgs` 停止，可以在 OMC 服务器用 `bash targets/cpro/scripts/decoder_compare.sh l0-decay-report` 读取现有 AIR 与 OMC。它从单层 `ChunkGatedDeltaRule` 的命名输入 `g` 回溯 `SoftplusPatch` 和衰减乘法，再显示 OMC 可见图中的生产者、消费者与图输出。OMG 元数据可能只暴露 `GraphOp` 边界。

在板端执行服务器用 `bash targets/cpro/scripts/decoder_board.sh --mode linear --phase prefill --gdn-conv conv2d --original-board-args` 对同一 OMC 改用原 V2 加载参数。如果仍重现相同加载错误，在合设的 OMC 与板端执行服务器运行 `bash targets/cpro/scripts/decoder_compare.sh l0-original-omg`。它以同一 AIR 调用原 CPro OMG 入口，再用相同的原 V2 参数对照。新 OMC、回执和三端比较写入 `board/original-omg/`，第一次编译的 OMC 留在 `board/L00/omc/`。

已有 L0 golden 时，在 AIR 生成环境执行 `bash targets/cpro/scripts/decoder_air.sh --mode linear --phase prefill --gdn-conv conv2d` 即可复用同一份输入和参考输出。单层导出在加载 checkpoint 并将权重转到 FP16 后，为普通 RMSNorm 预先计算 `gamma = weight + 1.0`，将其作为 buffer 传入原 `npu_rms_norm`。如已有 OMG 的 `Add_3 compileInfo is null` 日志，可在覆盖产物前，在 OMC 生成环境执行 `bash targets/cpro/scripts/decoder_compare.sh diagnose --mode linear --phase prefill` 查看同名 AIR 节点的操作数与消费端。普通 RMSNorm 的 gamma 对应 `RmsNorm` 的第二个输入。

在合设的 CPU/310P golden 与 CPro AIR 生成环境执行：

```bash
bash targets/cpro/scripts/decoder_prepare.sh --mode linear --phase prefill --gdn-conv conv2d
```

在合设的 CPro OMC 生成与板端执行环境执行：

```bash
bash targets/cpro/scripts/decoder_run.sh --mode linear --phase prefill --gdn-conv conv2d --replace-inputs --replace-weights
```

该用例执行 L0 的 GDN、残差和 MoE。CPU 和 CPro 的单层入口来自同一组 embedding 编码和 scale。GDN 门控 SiLU 的原始输入形状为 `[2048,128]`。共享专家的标量门控将 FP16 点积结果直接传给 sigmoid。导出时打印 AIR 中的门控数量和端口 dtype。查看 `$AI_RUN_DIR/target/tests/1-linear-prefill-t64/board/matrices.csv` 中 `hidden_0` 的有效 token，以及同目录的 `comparison.json`。

要用板端 dump 观察完整 L0 的自定义 RMSNorm，在 78（310P 与 AIR 环境）执行

```bash
bash targets/cpro/scripts/decoder_prepare.sh --mode linear --phase prefill --gdn-conv conv2d --rmsnorm-patch --dump
```

在 35（OMG 与 HDC 环境）执行

```bash
bash targets/cpro/scripts/decoder_run.sh --mode linear --phase prefill --gdn-conv conv2d --rmsnorm-patch --dump
```

`--rmsnorm-patch` 在普通和 GDN 门控 RMSNorm 的 PyTorch 调用处使用 `torch.ops.npu.rmsnorm_patch`。模型按现有 dtype 传入 x 和 gamma，直接使用算子返回的 z。kernel 注册的 `x, gamma → z` 均为 FP16 ND，gamma 长度等于 x 的最后一维。单层普通 RMSNorm 传入预计算的有效 gamma，门控 RMSNorm 使用原权重。OpDef 没有 epsilon 参数，精度对照需核实 kernel 的固定 epsilon。源 AIR 各端口的实际 dtype 见 `export/package/L00/case.json` 的 `rmsnorm_nodes[].air_dtypes`。

L0、四层和四十层的常规 T64 prefill 默认在 PyTorch 中使用 FP16 的 QLinear 分组权重 scale，共享专家的 FP16 MatMul 结果直接进入 sigmoid。导出器记录每个 W4 matmul 的 `scale`、`pertoken_scale` 端口和每层门控的实际 dtype。逐 token 激活 scale 延用原模型的计算路径。需要复用同一深度的参考数据时，78 的 `decoder_prepare.sh --reuse-golden-from <基线 case 目录>` 会先接收已有 golden，再导出 AIR。

`--dump` 使用原 decoder 用例目录。78 保存同一批 token 和 L0 入口的 CPU/310P 中间值。35 只编译一个完整 L0 OMC，使用 V2 和 `--enable_item=2` 执行一次，再运行 `data_proc_tool_internal`，拉回解析器输出。从用例目录可查看

- `board/L00/replay/dump/` 和 `dump-inventory.json`（同级）。完整文件树、文件大小、板端推理时间窗口和解析器命令。`parser.log` 保存解析器输出。
- `board/package/fixture/l0-steps-reference/values.npz` 和同级的 `tensors.json`。CPU、310P 参考张量及其名称、dtype 和形状。
- `board/comparison.json` 和 `board/matrices.csv`。同一次推理的整层输出三端比较。中间值与 Python 变量的对应关系待根据本机解析器的实际 dump 格式确认，状态为 `PENDING_PORT_MAPPING`。

已有完整 L0 dump 时，在 35 运行 `python3 targets/cpro/tools/l0_dump_report.py "$case_dir"`，其中 `case_dir` 指向本次的 `1-linear-prefill-t64` 目录。它对比准备送板的 hidden 与 CPU、310P 的 L0 入口，再用本次 AIR 的输入名和首个 RMSNorm 的完整节点名查找 dump。对逻辑维度匹配的候选，脚本按字节数列出连续 FP16、FP32 或 FP16 前缀等可能读法，与送板输入及两端归一化后的 `hidden` 比较。长度多出 32 字节时分别列出偏移 0 和 32 的结果。报告给出有效 token 的误差和全 64 行是否与输入相等。文件名中的序号、比较分数和相同内容分组都不自动决定端口归属。

要一次生成整层逐步三端矩阵并定位首个大偏差，在 35 运行 `python3 targets/cpro/tools/compare_operator_dump.py --case "$case_dir"`。结果在 `board/L00/dump-comparison/`。其他层或模型也可使用此离线工具，端口映射和读表方法见[通用算子 dump 三端对比](guides/operator-dump-comparison.md)。

确认第一个 RMSNorm 的 x、z 各为完整的 FP16 ND 原始文件后，可在原命令后加 `--rms-io <x文件> <z文件>`，路径相对 `board/L00/replay/dump/`。它会对比 x 与送板输入、z 与两端的 `hidden`，范围是有效 token。带分片或有格式头的文件需先确认解析器的格式。

- 板端的 `model_run_tool_internal` 使用 V2，显式传入 `--input_shape` 和 `--output_shape`。每项按端口顺序写作 `名称:物理维度`，同一方向的多个端口用分号连接。旧编译记录缺少输出名称时，同机可用的 OMG 可以从现有 OMC 补读元数据；仍缺名称时，执行端通过 `runtime.json` 配套的描述查询工具读取板上模型。完整调用保存在该次 `replay/run.sh` 和 `replay/result.json` 的 `argv` 字段。
- 板端固定使用 `/data/local/tmp/qwen35_case_<用例>/work`，例如 `qwen35_case_40-prefill-t64/work`。逐层模式在用例名后追加层号，例如 `40x1-prefill-t64-L00`。
- 每次上传当前 OMC。脚本按清单里的文件标识和板上的字节数复用输入、外置权重，缺失或变化的文件随 OMC 一次上传。成功后一次拉回输出目录并按槽位绑定，删除板端 OMC 和输出，留下当前输入和权重。失败现场保留到同用例下次启动，路径见 `result.json` 的 `remote` 字段。

在板端执行服务器，已有编译产物时可以直接重跑。例如，手动改了输入 bin 后用下面的命令强制重传所有输入：

```bash
bash targets/cpro/scripts/decoder_board.sh --mode 40 --phase prefill --gdn-conv taps --replace-inputs
```

用 `--replace-weights` 强制重传所有外置权重，两项可以一起使用。嵌入 OMC 的权重会随 OMC 更新。这两个参数同样适用于组合入口 `decoder_run.sh`。手动覆盖同名同大小文件且没有重新生成清单时，使用对应参数。逐层模式显式替换时会重新执行各层，即使同时给了 `--recovery`。

四层用例在板端工具退出失败后会自动读取该次运行的 HiLog。已有失败现场可在板端执行服务器单独取日志：

```bash
bash targets/cpro/scripts/decoder_compare.sh diagnose --mode 4 --phase prefill
```

先取日志再重跑，同名板端工作目录会在下次运行时覆盖。诊断使用本次板端时间和工具线程号关联客户端 PID，再查找 hiaiserver 同一请求的加载错误。运行输出会打印报告、完整时间窗口和匹配的客户端日志路径。

四层 V2 返回成功而精度失败时，同一条 `diagnose` 命令离线读取保留的运行包，输出有效行的 embedding 重建差异和卷积控制量。保留了原生卷积 AIR 时还会打印 L0 QKV matmul 的输入与 scale 端口。

### 四层 T64 实际输入下的 L0 逐步精度

各组的变量、直接上游、逻辑形状和比较范围见 [L0 逐步三端对比变量表](guides/l0-steps-variables.md)。

要定位进入 L0 前的差异，在 CPU/310P golden 生成服务器运行一条命令。它使用原 `inference_segment.py`、原 YAML 和现有四层输入包，加载原 segment 1 到 NPU，在 L0 forward 前停止。

```bash
bash targets/cpro/scripts/decoder_compare.sh l0-embedding
```

默认读同一用例的 `export/fixture` 和 `export/package/L00-L03`。AIR 生成环境分开部署时，`DECODER_EMBEDDING_CASE` 可指向 golden 服务器可读的本次 `L00-L03` 目录，只需其中的 `case.json` 以及 `inputs/inputs_embeds.bin`、`inputs/embed_scale.bin`。

`l0-embedding/result.json` 记录实际执行的查表或逐行乘法、数据类型、源码位置、与 L0 实测 FP16 hidden 相符的观测点，以及各有效 token 的编码和 scale 对照。只观察到浮点查表时，仅能确定这次 forward 用到的查表类型。若观测点都不能还原入口，状态为 `ENTRY_PATH_UNRESOLVED`。这项诊断取数在 input RMSNorm 之前，后续投影的激活编码见 `qkv_a8`。

保留已运行的 `4-prefill-t64`。这组诊断使用其中的四层 AIR、物理尾部 ConvState、Conv2D 和自定义激活，支持 RMS 门控的 `[64,4096]` 视图。两台机器使用同一版仓库。

1. 在 CPU、310P golden 生成环境执行 `bash targets/cpro/scripts/decoder_compare.sh l0-steps-reference`。脚本用原 YAML 的 prompt 再取一次 L0 的实际张量，对齐四层 fixture 的 token、输入 hidden 和两个返回缓存，并在 CPU 上用原 W4 权重独立计算。
2. 在合设的 CPro OMC 生成及板端执行环境执行 `bash targets/cpro/scripts/decoder_compare.sh l0-steps`。它从现有 AIR 切出六组观测，从原始 embedding、scale、初始状态等输入算到观测点。参考文件沿用已有 SSH 配置或 `DECODER_GOLDEN_LOCAL_ROOT` 挂载。单独重跑已编译的诊断 OMC 可用 `decoder_compare.sh execute --gate-probe l0-steps --mode 4 --phase prefill`。

已采集的 T64、19 个有效 token 的一次运行中，310P 在 `embed_tokens(input_ids)` 查 FP16 表，输出与 L0 入口逐元素相同。CPro 的 INT8 embedding 乘 scale 后，主机重建与该入口的相对 L2 为 2.472%，板端与主机重建的相对 L2 为 0.0263%。这两项测量都在 L0 投影的 DynamicQuant 之前。

要单独观察 L0 计算造成的偏差，在合设的 OMC 生成及板端执行环境运行 `bash targets/cpro/scripts/decoder_compare.sh l0-steps-same-input`。它复用四层 AIR 和已有的 CPU/310P 参考，把 CPro 的 embedding 乘法输出替换成同一份 310P 层入口。原图端口为 FP32 时，对 FP16 输入做精确扩宽。对照结果写在 `board/L00-L03/l0-steps-same-input/`。先看“**三端 L0 入口全部 64 行逐元素相同**”，再按有效 token 比较 `out`、`post_norm` 和 `shared_gate_logit`。原四层运行使用量化 embedding，因此其返回缓存仅在原输入诊断中作为切图对照。

结果在 `$AI_RUN_DIR/target/tests/4-prefill-t64/board/L00-L03/l0-steps/`。每个变量的说明和三端矩阵在 `cosine.txt`、`error_ratio.txt`，相对 L2 在 `matrices.csv`。先看 `comparison.json` 的 `four_layer_cache_control`，其中 `exact` 表示诊断切图与四层板端返回的同一个 L0 缓存逐元素相等。随后看 `first_cos_or_l2_crossing_by_group` 以及对应的上游张量；筛选条件为有效行余弦 ≤ 0.99 或相对 L2 ≥ 0.05。观测值包含上游累积误差，额外导出中间量也可能改变 OMG 优化。缺失、编译失败或无法对齐的项目列在 `missing`。

`validN` 取 YAML prompt 的有效 token。`physical60_63` 单列最后四行，供对照按物理尾部更新的 ConvState。`expert_ids` 按每个 token 的专家集合计算，`route_weights` 按专家编号展开。专家内部的两次矩阵乘和 SwiGLU 只比较三端共同选中的同一 token 和 expert，范围名给出共同路由数与总数。310P 的融合 SwiGLU 只直接暴露乘积。GDR 取融合算子的实际入参、core 和最终状态。
