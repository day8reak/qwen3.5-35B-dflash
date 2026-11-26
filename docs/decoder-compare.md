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

此选项以对齐后的 CPro 源码为基础，在本次导出的副本上恢复原生 SiLU 和 Softplus。W4 scale、RoPE、GDR、MoE 等修复继续生效。导出时打印并检查 `SiluPatch=0` 和 `SoftplusPatch=0`，同时检查 W4 和 GDR 端口。两种四层原生卷积对照都从相同的 golden、embedding 权重和 YAML 配置生成输入。原 CPro wrapper 接收 INT8 embedding 及逐 token FP32 scale。普通分层诊断图和 T1 decoder 图接收 FP16 hidden。原生卷积还接收 128 个 INT32 的 `conv_state_pos`，默认首项为有效长度、其余项为零。可用 `DECODER_CONV_CONTROL` 指定原板端 bin。

同一运行目录中保留了成功的 `--no-patch` 记录时，板端执行服务器会先保存其输入槽位，再逐项对照本次的角色、shape、dtype 和已有 bin 内容标识。该对照使用服务器上的清单与执行记录。

上面两条命令在原 ConvState 更新之后显式调用分组 `F.conv2d`。L0、L1、L2 各导出一个 Conv2D，卷积窗口为 `[1,8192,1,67]`，卷积核为 `[8192,1,1,4]`。两种激活对照都从 `CPRO_ORIGINAL_PATCHED_SOURCE`（或自动生成的原 ConvState 源码）导出，使用相同的四层配置、OMG 参数和 V2 输入接口。在这两种激活对照中，省略 `--gdn-conv` 会沿用此前的 `original` 卷积配置。

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

V2 执行完毕后，终端以 `PASSED` 或 `FAILED_ACCURACY` 打印 `comparison.json` 的路径。精度未达标时会列出未通过项并以状态码 1 退出。加载或执行阶段的异常仍可从该阶段的报错和 `replay/result.json` 定位。

### 用原 CPro 的板端命令重跑已有 OMC

在板端执行服务器上，可以保留现有四层 `taps` OMC 和输入，只改变传给 `model_run_tool_internal` 的参数。脚本按当前 OMC 的槽位绑定输入，并传入 `--model`、`--input`、`--times=1`、`--modelmanager_itf=V2`、外置权重的 `--weight_dir`、`--enable_item=1`、`--input_shape`、`--output_shape` 和 `--output_dir`。加载方式和输入输出类型使用工具默认值。实际命令保存在 `board/L00-L03/replay/run.sh`，退出状态在同目录的 `result.json`。

```bash
bash targets/cpro/scripts/decoder_board.sh --mode 4 --phase prefill --gdn-conv taps --original-board-args
```

### 从原 CPro 源码生成四层或四十层 prefill

`--gdn-conv original` 保留原 CPro wrapper、`LinearAttentionConvstateUpdate` 的 `linear_update_tensor` 输入和模型源码中的 `F.conv1d`。模型继续使用修正的 CPro QLinear 与 FP16 尺度、RoPE、MoE、GDR 接口以及 FP16 输入输出的 `SiluPatch` 和 `SoftplusPatch`。导出单张图，返回 logits 和每层两个状态。四层有 9 个输出，四十层有 81 个输出。板端使用上一节的 V2 参数。

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
- 四层和整网 T64 均使用原 CPro 导出入口、量化 embedding 与逐 token 的 embedding scale。四层设置 `num_hidden_layers=4`，将 L0–L3 连同最终 norm/head 导成一张 AIR。原生接口返回 logits、前三层各自的 conv/recurrent state，以及 L3 的 K/V，共 9 个输出。其余组合从实测入口 hidden 和状态开始。W4 scale、RoPE、MoE、GDN、FP16 SiluPatch 和 SoftplusPatch 的整网检查见[原 CPro 整网 prefill](target-prefill40-single.md)。
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
