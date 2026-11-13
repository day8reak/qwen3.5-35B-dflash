# Target W4：310P golden → CPro 小子图精度定位

当前执行命令集中在[Attention 定位与整层 GDN 测试](target-batch-rerun.md)。旧 target-layer-accuracy 产物和 golden 已删除，先重新采集四层参考，再只导出、编译 Attention Prefill／Decode 两张图。新产物统一放在 target/tests；下面旧路径仅是历史方法示例，不能作为当前仍存在的输入。

2026-11-10：当前主线已调整为[四层整图 dump 与 OMC／精度状态表](target-accuracy-status.md)。本页保留独立边界捕获、重放和编译排查的方法，用于整图首次偏差已经缩小后的复现；不再要求先逐个编译通过所有边界图。

首批仅覆盖原模型第 0～3 层的 prefill 和第一个 decode。**导出阶段不构造整模型，不构造整层再裁图。** 每个进程只导一个 CPro 边界的一个输出，退出后再处理下一个。MoE 专家大子图不在本批导出范围内。

原 `inference.py` 和私有模型文件保留原样。用户确认这里的可信参考是 `inference.py` 四层运行，整网分段运行使用另一个 `inference_segment.py`。当前直接步骤入口固定四层；旧 probe 入口默认 `--model-layers 4`。两者都在原 `load_npu_model` 构造时传入 `layer_num=4`，缓存分配也使用四层，不会先构造 40 层再裁剪。decode 使用这四层加原 norm/lm_head 的输出，不声称是完整 40 层模型生成。

构造 device 为 CPU，完成原 W4 量化替换后才搬到 NPU。当前直接步骤入口保留四层量化模型；旧 probe 逐个搬入 decoder layer、norm 和 lm_head，在执行后同步搬回 CPU。原 loader 仍会读取完整 HF/量化 checkpoint，主机内存需要足够。搬运不传 dtype、不做 `.half()`，CPU 不重算 golden，诊断计时不作为模型性能。入口要求原外部量化 embedding 路径，不支持 `--use_origin_model` 或 pseudo_quant。

新增工具代码可公开，原始源码、权重、配置和实际 dump 留在各自工作目录，不提交仓库。

本地已验证采集与重放的主机逻辑；实际 TorchAir/AIR、选定 DDK 的 OMC 和 X90 数值结果需要在用户服务器上执行。单独子图的编译失败、接口差异、运行失败、数值误差分别保留。

## 对齐边界

| CPro 边界 | golden 的来源 |
|---|---|
| Input/Post-attention RMSNorm、Q/K Norm | 原模块实际输入输出 |
| QLinear / 原有普通投影 | 原投影输入输出，以及已经转换完成的 W_q、scale 或 weight/bias |
| LinearAttentionConvstateUpdate | 原实际更新的 conv_state；原 Conv1D 输入中最后 T+K−1 个位置 |
| Conv1D | 上述有效输入；原卷积输出中实际被使用的最后 T 个位置 |
| SiLU | 原卷积后的实际激活调用 |
| decay / beta | 原 GDN 调用的 g、beta；原 a、b 投影输出和 A_log、dt_bias |
| repeat Q/K | 原 GDN 的重复 head 输入；取每组首 head 作为重放输入 |
| ChunkGatedDeltaRule | 原 NPU 调用的 core、raw final state，分别导出 |
| recurrent state 回写转换 | 原 raw state 和实际写回缓存的 state |
| RoPE | 原函数实际 Q/K 输入输出 |
| CacheUpdate K/V | 调用前 cache、updates、块号、偏移；实际更新后的完整 cache |
| FusedAttention | 原 NZ 布局输入和实际输出，保留 mask、页表、长度 |
| attention 布局恢复与 gate | 原 attention 输出、Q projection 中的 gate、实际 o_proj 输入 |
| residual add | 原分支输出、残差和实际加法结果 |

CPro GDN 接收调用前缓存的 dtype；310P 传入内核前的 FP32 cast 不会被误当成 CPro 的输入合同。量化投影沿原 exporter 的 replacement globals 选择实际 QLinear，不因 CPro 文件中有同名类就替换。重新导出通过 `--target-config` 读取原 HF／W4 权重：普通 Norm 按 CPro 参数 dtype 加载，W4 由原工具重建并与 capture 核对，golden 不改。当前选择范围与完整命令见[GDN 单模块测试](target-batch-rerun.md)。

Convstate 的辅助 `linear_update_tensor` 使用原 CPro 导出声明的零 INT32[128]，并在夹具记录该策略。`effective_length` 来自实际 310P 调用，不使用导出示例里的零长度。若原程序走外部卷积实现，未观察到有效 Conv1D 边界，采集标记 INCOMPLETE；不会编造 golden。

## 1. 在已验证的 310P 环境采集

以下变量由实际环境设置：`REPO_DIR` 指向本仓库，`TARGET_SOURCE` 指向原项目（含 inference.py、utils.py、models/），`AI_RUN_DIR` 指向仓库外的输出目录。激活原来的 PyTorch/torch_npu 环境，保留原来工作的配置和推理参数。

用户已确认切换环境后新旧入口均可运行。本轮继续使用已跑通的 `inference_target_probe.py` 和其采集结果；已有 CAPTURED 批次可直接进入导出/编译，无须换入口重新生成。

### 备选入口：直接调用原 inference 的步骤

用户要求直接提取推理步骤，新增 `inference_target_steps.py`。导入原 `inference.py` 的函数定义，**不执行其 main，不使用 runpy**。流程为原 wrapper 构造（工厂仅创建前四层、CPU）→ 原 `quant_model_inplace(...quant_bits=4)` → 仅搬运量化后的四层模型 → 原 `prefill()` → 其实际 logits 的 argmax 和返回 cache → 原 `decoder()`。不改 forward、量化算法、dtype、group_sizes、ACLNN 版本或 GE converter，不安装 V4/V5 桥接库。

```bash
python "$REPO_DIR/targets/cpro/tools/inference_target_steps.py" \
  --source-root "$TARGET_SOURCE" \
  --config ./config/qwen3_5_moe.yaml \
  --output "$AI_RUN_DIR/target-layer-accuracy/direct-001" \
  --max_token 1
```

默认执行第一个配置 prompt 的 prefill + 一次 decode，dump 第 0～3 层。注意用户提供的 main 最后覆盖了 chat template，实际使用 **`prompt * 32`**；新入口默认 `--prompt-repeat 32`，不偷偷换输入或添加 padding。已有原 golden 时，可以添加 `--input-ids /实际路径/input_ids.bin`，直接复用其中的 INT32、batch=1 token（此路径要求没有 padding，真实长度等于文件 token 数）。`--prompt-index` 选择配置中的 prompt；更改输入策略须与原 golden 对齐。

输出布局：

- `steps.json`：原源码哈希、运行环境、逐步状态、实际 token ID、执行结果；成功执行仍不代表已与旧 golden 做过数值验证。
- `original/prefill/` 和 `original/decode/`：原函数自行保存的 bin，分开目录避免 decode 覆盖 prefill；原 decoder 多步循环内部的同名文件仍按原逻辑覆盖。
- `original/prefill-result.pt`、`original/generated-ids.pt`：实际返回的 prefill logits/首 token、decode token；没有第二次重算。
- `capture/manifest.json`：CPro 粒度的中间输入/输出和权重。后续 list/export 的 `--capture` 指向这个 `capture/` 子目录，批量导出和比较脚本保持不变。

若要直接验证没有采集 hook 的原函数路径，增加 `--baseline-only` 并使用新输出目录；它只保留原函数结果，不生成 CPro 中间采集。两个模式都只构造四层，不建立四十层模型，不导整图。原 loader 仍读取完整 checkpoint，因此主机内存需求未变。旧 `inference_target_probe.py` 保留用于复现已有运行，其逐层搬运策略与此入口的四层量化后常驻策略分别记录。

2026-11-09 的环境报告确认：已加载的 custom QuantMatmul 库导出 V4 pair，没有 V5 pair；当前失败调用为 V5，属主规则对照通过。用户补充的 `QuantBatchMatmulV4444` / `GroupedMatmulV555` 打印来自 **AIR 导出**，不能当成 310P eager 成功执行记录。这些是 GE op type，不能直接作为 ACLNN API 版本。直接步骤入口保留原调用；若同样报 V5 错误，仍需恢复原成功运行的源代码/环境组合，不能把错误归为 CPro 数值偏差。

### 本轮沿用的入口：执行原主程序并逐层搬运

```bash
cd "$TARGET_SOURCE"
python "$REPO_DIR/targets/cpro/tools/inference_target_probe.py" \
  --source-root "$TARGET_SOURCE" \
  --output "$AI_RUN_DIR/target-layer-accuracy/golden" \
  --model-layers 4 --layers 0,1,2,3 --calls 2 -- \
  --config "$TARGET_CONFIG" --quant_mode enable --quant_bits 4 --max_token 2
```

启动应打印 `model on CPU; original quantization runs before layer streaming`，第一次 forward 应打印 `streaming all 4 decoder layers, norm and head; one unit on device at a time`。manifest 的 `placement` 记录 checkpoint 层数、实际构造层数和各单元调用次数。旧版 096d207 固定完整模型计算的策略已被四层参考替代。重试使用新的 output 目录，保留失败目录；后续 list/export 的 `--capture` 对应改成新目录。

`--` 后是原 inference.py 的参数，可以保留原来的其他参数。不要启用 pseudo_quant。输入 prompt/长度由原程序决定，不在采集入口中偷偷补齐或截断。原程序需完成 prefill 和至少一次 decode。`--calls 3` 可额外采第二个 decode，但 CPro 端仍是独立 golden 输入重放，不等于连续反馈验证。

先确认原程序仍生成正确内容，检查 `golden/manifest.json` 的 status 为 CAPTURED。`wrapper_outputs/0/output0.bin` 是原 wrapper 的第一个实际输出，可与原 golden 比对；它不是额外导出的整模型图。manifest 包含原命令、源文件身份、覆盖情况及逐 tensor 的 shape/dtype/SHA256。

```bash
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" list \
  --capture "$AI_RUN_DIR/target-layer-accuracy/golden"
```

### 首个 QLinear 报 groupSize 错误时

原 `QLinear` 未传 `group_sizes`。[Ascend op-plugin 7.1.0 实现](https://github.com/Ascend/op-plugin/blob/7.1.0/op_plugin/ops/opapi/QuantMatmulKernelNpuOpApi.cpp) 的 `check_and_get_groups` 在未传参数时返回 0，A8W4 分支将其直接交给 `aclnnQuantMatmulV5`。这与当前日志要求 groupSize=256 不一致；仅凭这一条错误不能确认已加载的 vendor 内核是否同时匹配 I4ZN 布局和 FP32 scale。

需要显式 group256 时，在采集命令的 `--` **前**加 `--qlinear-group-size 256`。只在真实 `x_quant=[M,K]` INT8、`W_q=[K,N/8]` INT32、`scale=[K/256,N]`、`pertoken_scale=[M,1]` 且连续时补入 `group_sizes=[0,0,256]`；不改权重、scale dtype、布局或量化分组。默认不改原调用，不自动失败重试或转伪量化。选项启用后的 reference 标记为 `310P_EXPLICIT_QMATMUL_GROUP256_BASELINE_UNVERIFIED`，需与已工作的四层输出对齐，不能直接声称原 golden 精度已通过。

新版本每次采集保存 `qlinear-audit.json`，包含算子真实 shape/dtype/stride、由 scale 推导的组大小、原始/实际参数、Python/torch_npu 路径与已加载库。若 matmul 失败，保存该次的 `qlinear-failure.pt`，下一轮只加载这个算子的输入和权重即可复现，不加载模型：

```bash
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" replay-qlinear-failure \
  --capture "$AI_RUN_DIR/target-layer-accuracy/golden-group256-001" \
  --output "$AI_RUN_DIR/target-layer-accuracy/qlinear-replay-001" \
  --group-size 256
```

省略 `--group-size` 则复现保存的原始调用参数。每次只调用一个 matmul，结果为 EXECUTED/FAILED；EXECUTED 不代表数值正确。旧版本失败目录未保存该夹具，不能用于这个命令。若后续出现 scale dtype 或布局错误，保留 audit 和 report，先核对已成功环境的 torch_npu/CANN/自定义库，不盲目转换 FP32 scale 或重打包权重。

### 显式 group256 后报 TransData 找不到 binary

当前反馈已从 groupSize 参数校验推进到 `aclnnQuantMatmulV5` 内部 `TransData` 的 GetWorkspace 失败（561103）。这仍是算子执行前的后端问题，没有输出可用于精度判断。报错中的 `c0:4294967261` 仅作为原始日志保留；不能据此推断需要哪一种格式、改 scale dtype 或重打包已经 I4ZN 排列的权重。

旧报告存在两处盲点：库名过滤遗漏了 `libcust_opapi.so`，单算子重放只取了调用前的 maps。更新后直接复用原 `qlinear-failure.pt`，不重新加载模型。设置 `QMATMUL_CAPTURE` 为刚才使用的 `--capture` 目录（含 qlinear-audit.json 和 qlinear-failure.pt），在同一环境运行：

```bash
ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_SLOG_PRINT_TO_STDOUT=1 \
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" replay-qlinear-failure \
  --capture "$QMATMUL_CAPTURE" \
  --output "$AI_RUN_DIR/target-layer-accuracy/qlinear-backend-001" \
  --group-size 256 \
  > "$AI_RUN_DIR/target-layer-accuracy/qlinear-backend-001.log" 2>&1
```

这是进程内的 [INFO 日志开关](https://github.com/Ascend/pytorch/blob/master/env.sh)，不改变输入、算子参数、量化方式或库搜索路径。再次运行请换新的输出目录和日志名。

`report.json` 新增以下证据，采集失败时也保留：

- `stage`：区分输入准备、op dispatch、输出回读及保存失败。
- `runtime_at_end` / `libraries_added`：调用后实际映射的 custom、stock、feature API 库；hook 包装期间仍能读取原 `_schemas`。
- `backend_inventory`：按 `ASCEND_CUSTOM_OPP_PATH` 声明顺序、OPP 默认 vendor 配置和实际 maps，读取对应库的路径、哈希、属主 UID、QuantMatmul/TransData 导出符号。`readelf` 缺失或失败写 UNAVAILABLE，不误判成“没有 V5”。不执行 dlopen，不加载候选库，不改环境/权限。
- `npu_formats`：本次重放张量的 NPU format 描述符。手工 I4ZN 字节排列与描述符是两回事；CPU 夹具不保存设备格式标签，旧采集缺失的标签标记 `NOT_RECORDED_BY_OLDER_CAPTURE`，不伪造一致性。

[官方 op-plugin 7.1.0 加载器](https://github.com/Ascend/op-plugin/blob/7.1.0/op_plugin/utils/op_api_common.h) 逐库查找符号，再做 [属主检查](https://github.com/Ascend/op-plugin/blob/7.1.0/op_plugin/utils/op_api_common.cpp)。报告的 `owner_compatible_with_op_plugin_7_1` 只是与这份公开源码规则的对照，当前安装包是否使用同一逻辑仍要结合日志确认。环境中的路径存在、库被映射、库导出 V5，都不能单独证明实际 dispatcher 选择了它。

回传 `report.json` 和同次 `.log`。重点核对 `aclnnQuantMatmulV5GetWorkspaceSize` / `aclnnQuantMatmulV5` 的 `is found in ...`、`not in custom lib`、属主跳过、dlopen/dlsym 错误和 TransData 失败上下文。没有命中日志不自动证明走 stock。若明确跳过 vendor，再处理该加载原因；若明确选中 vendor，再查该 vendor 的内部 TransData 依赖与当前 CANN/310P 支持。原始运行环境和成功 golden 仍是对照，不根据假设切换到伪量化。

## 2. 有 TorchAir 的环境逐个导出小 AIR

设置 `CPRO_EXPORT_SCRIPT` 为原来用于 CPro AIR 的导出脚本。仅导入其中的 converter 定义，不调用其 main/export_air。

```bash
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" export \
  --capture "$AI_RUN_DIR/target-layer-accuracy/golden" \
  --source-root "$TARGET_SOURCE" --converter-script "$CPRO_EXPORT_SCRIPT" \
  --target-config "$TARGET_CONFIG" \
  --output "$AI_RUN_DIR/target-layer-accuracy/air-batch"
```

每次只读取该小子图所需的 CPU 权重和输入。导出使用静态真实 shape；每个输出单独成为 AIR 唯一输出，不使用整层多输出图。batch 自带运行所需的输入和 golden，迁移到 DDK 服务器时保留完整目录结构，包括 `weight/` sidecar。

若先查指定边界，可增加 `--points convstate_update conv1d conv_silu decay_beta repeat_qk gdn recurrent_writeback`，或者使用 `--kinds norm norm_gated qlinear`。筛选只决定本次导出哪些图；采集数据保持完整。失败日志为 batch 根目录各 case 的 `.log`，不自动改 converter 或尝试其他精度。

## 3. DDK 服务器批量编译

在该服务器设置 `BATCH` 为迁移后的 air-batch 目录。`OMC_BATCH` 必须是该服务器 `AI_RUN_DIR` 下的新目录，与 AIR batch 和 DDK 目录互不包含。使用已选定的 X90 DDK，不换芯片或安装目录。

```bash
OMC_BATCH="$AI_RUN_DIR/target-layer-accuracy/omc-batch-001"
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" compile \
  --batch "$BATCH" --output "$OMC_BATCH" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --tbe-diagnosis "$TBE_DIAGNOSIS"
```

`TBE_DIAGNOSIS` 使用此前已经完成的兼容诊断目录；无须该兼容选项的环境可省略该参数。保持原编译使用的 format 选项；需要时显式传 `--use-origin-format`。不设置 weight_data_type，保持量化载体。小图要求权重内嵌 OMC。每个编译结果和原始日志分别保留在 `$OMC_BATCH/models/<case>/omc/`，总报告是 `$OMC_BATCH/compile.json`。

旧版若全部报 `compilation output must be outside the source and DDK directories`，是脚本把输出放进 AIR 源目录导致的编译前失败。更新代码后直接用上述新 `--output` 重试，无须重新采集或导出 AIR，也无须删除原 `compile.json`。其他编译失败重试同样换用 `omc-batch-002` 等新目录。底层 OMG 仍使用 `--output=model`，由脚本设置独立工作目录。

OMC batch 保存 AIR manifest 的快照，并以相对路径引用原 AIR batch 的输入和 golden；上板准备前会校验两份 manifest 及夹具的哈希。保留原 AIR batch，迁移时将两个目录一起移动并保持相对位置。

### 读取失败子图的 AIR / IR 描述

`CalledProcessError` 只说明 OMG 退出，实际失败点在各 case 的 `omc/compile.log`。以下入口读取已有文件，汇总源 AIR、编译阶段、日志尾部和失败 IR；不会运行 OMG、导图或修改权重。用导出时保存的 `air-schema.pb` 解读源图，用显式选择的 DDK schema 解读编译器 IR，不能混用两者的 dtype 枚举。

```bash
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" inspect-compile \
  --batch "$OMC_BATCH" --ddk-root "$CPRO_DDK_ROOT" \
  --cases l00_c00_input_layernorm_o0 l00_c00_linear_attn_in_proj_qkv_o0 \
          l00_c00_gdn_o0 l00_c00_linear_attn_norm_o0 \
          l00_c00_convstate_update_o0 l00_c00_conv1d_o0 \
  --output "$AI_RUN_DIR/target/tests/accuracy-inspection-001"
```

`report.json` 保留各节点端口、shape、schema 对应 dtype、属性和连线；不包含权重 Tensor payload。空 shape 标记未解析，不补写预期值。优先从 `omc/diagnostics/manifest.json` 读取已保存的 IR 并校验 SHA256；旧批次兼容 `omc/diagnostics/*.irpb`、`omc/compile-work/generated/dump_graph/*.irpb`，以及无扩展名的普通文件 `omc/diagnostics/dump_graph`、`omc/compile-work/generated/dump_graph`。每图最多解读 12 份 IR，优先失败推形和分区后的快照；无法解码的 IR 保留路径、哈希和错误，不按另一种 schema 猜测。缺失、超出保存预算、文件变化和缺少 DDK schema 均使检查状态为 `PARTIAL`，`ir_inventory` 记录查找范围、文件类型及保存状态。此入口依赖已迁移的 AIR batch 及其 schema 文件，不要求导出服务器上的 torch_npu 环境。

旧检查器仅搜索临时 `dump_graph/`，遗漏了 `diagnostics/`。更新后可以直接检查现有失败批次，无须为收集已保存的 IR 再次编译。若仍为空，结合 `compiler_report.compiler_diagnostics` 和 `ir_inventory` 判断是否实际保存，不能仅凭日志中的 “dump file ... create” 认定文件存在。

2026-11-10 后续回传已确认三个 case 的 `dump_graph` 都是无扩展名普通文件，检查状态为 `FOUND` / `INSPECTED`。原保存器和检查器只按目录处理的遗漏已修正。这三份旧文件标记为 `UNBOUND_LEGACY_FILE`，其中间 shape 仍为空，节点与源图基本一致；不能当作已完成推形或分区的快照。QLinear 的 `[256,1024]` 仍以该次编译日志为证据，不能从这些空 descriptor 推导最终存储类型。

### 与已成功整网编译的条件对照

用户补充的成功命令先 source `tools/tools_ascendc/set_ascendc_env.sh`，再调用 `tools/tools_omg/omg`，使用 `--limit_partition_size=false`、外置权重和动态档位 64/1。它为对照基线提供了执行条件，但尚未证明本批小图复现了这些条件。

| 条件 | 用户报告成功的整网 | 当前小图 |
| --- | --- | --- |
| 入口 | 外层脚本最终调用同一个 `master/omg`，不追加 platform/target 参数 | 直接调用 `master/omg`，显式 `--platform=kirinx90 --target=omc` |
| 环境 | source 原 AscendC 环境脚本 | 继承调用进程后，过滤已知外部 CANN 路径、重排库和 Python 搜索路径，并设置 SOC/配置路径；若启用 TBE 兼容选项，还使用局部 wheel |
| shape | `--dynamic_dims=64,64,64,64,64;1,1,1,1,1` | 本次 prefill 为静态 256；decode 为静态 1 |
| 状态与连接 | 80 个 `RefData`，保留完整消费关系 | 独立 `Data` 输入，每个图仅保留一个输出 |
| 分区与权重 | `limit_partition_size=false`，外置权重 | 未显式指定分区限制，权重内嵌 |

同名算子在独立图和完整图中可能经过不同融合、布局转换与实现选择。ConvState 的第二输出在整网连接后续卷积，单输出探针可能失去该融合条件；这是待原图/日志验证的假设。整网成功不证明静态 256 的单输出版本可独立编译，小图失败也不能推出 DDK 全局缺少算子。外置权重本身没有证据可解释 V4444 的八倍宽度差异。

已回传的两个入口脚本确认：外层入口把 `LD_LIBRARY_PATH` 替换为 `master/lib64`、`ddk/tbe/lib64`、`ddk/ascendcpp/lib64`，设置 `CCE_LIB_DIR` 和 `KIRIN_CONF_JSON_PATH`；已导出的 `HIAI_VERSION` 改为 `master`。只有显式传入 `--platform` 才追加平台库与 Python 路径并设置 `SOC_VERSION`。原成功命令没有该参数，因此外层脚本保留继承的 `SOC_VERSION` / `PYTHONPATH`，并不证明它们为空。source 脚本还调用了未回传内容的 `set_tbe_env_ddk.sh`，不能仅凭两个入口文本还原历史成功环境。外层脚本不改写 AIR，也没有 INT4 解包逻辑。

已提供的 `sourced-wrapper-default` 在原成功命令的已 source shell 中，对同一份保留 AIR 做入口对照。这个显式诊断 profile 复现已检查的 Linux/master/无 platform 参数分支，不执行厂商脚本中的 chmod、mkdir 或 loader 链接写入。两个脚本必须匹配回传 SHA256，且当前 `DDK_PATH` 必须指向所选 DDK；不匹配时在创建编译批次前报错。它保留当前 shell 的注册与 Python 环境，直接执行同一个 master ELF，省略 target/platform 参数，并选择 `limit_partition_size=false`、外置权重；不能同时传 `--tbe-diagnosis`。

这是一组入口、环境、参数条件的对照，尚不是单因素实验。输入 shape 保持原静态 256，状态仍是 Data、单输出；不套用整网 89 个输入名或 64/1 动态档位。保留原有运行目录内的 IPC 临时目录和 multiprocessing overlay、禁止 Python 字节码写入，并在报告中列明这些与原命令的差异。对照成功后还需拆分变量确认原因。

以下命令已由用户执行，保留用于复现，无须重复运行。用户回传 `accuracy-entry-compare-004/compile.json`：QLinear、ConvState 仍退出 1，GDN 仍 SIGSEGV（-11）。摘要不足以确认失败位置和之前完全相同；本轮入口条件组合未使三图编译通过。

```bash
AIR_SOURCE=$(python - "$OMC_BATCH" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
report = json.loads((root / 'compile.json').read_text())
print((root / report.get('source_batch', '.')).resolve())
PY
)
ENTRY_COMPARE="$AI_RUN_DIR/target/tests/accuracy-entry-compare-004"
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" compile \
  --batch "$AIR_SOURCE" --output "$ENTRY_COMPARE" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --compiler-profile sourced-wrapper-default \
  --cases l00_c00_linear_attn_in_proj_qkv_o0 \
          l00_c00_convstate_update_o0 l00_c00_gdn_o0
```

`compile.json` 保留完整源 manifest 哈希和 `source_batch`，列出 `selected_cases`；未选项标为 `NOT_SELECTED`，顶层编译状态仅针对所选项。每个 case 的 `omc/report.json`、`compile-command.json`、`omg-environment.json` 和 `compile.log` 保留原始证据。平台标为 `UNSPECIFIED_COMPILER_DEFAULT`，不得据此宣称 X90 已编译通过；现有板端 fixture 入口拒绝该诊断批次。默认 `isolated-kirinx90` 路径不变。

优先比较 QLinear 是否仍推为 1024、ConvState 是否仍缺少 format config、GDN 是否仍返回 -11。若一致，再分别核对算子合同、整图消费者/RefData 及 GDN 调用栈；若不同，再逐项拆分入口条件。整个流程复用现有 AIR，不重新导出或编译 40 层模型。

### GDN 已成功编译的历史基线

应先复用历史已定位结论，避免把同名算子不同接口的成功混为一谈：

- **2026-10-27 的七输入融合探针也曾编译成功。** 历史代码含 INT16 `effective_length`，g 为 FP16，状态绑定经安装源码校验改为 FP16，两个输出均保留。成功的用户回传见[历史记录](history/validation.md#融合gdn最新真实编译回传)，两例 TBE 审计均为 `APPLIED`。因此不能认定七输入本身不可编译。
- 当时的直接编译失败为 `platform_info.te_update_version` 的 `RuntimeError`。12 组用户实测确认：先初始化 `kirinx90` 后 SOC 返回 `Ascend310B1`，底层更新再传 `kirinx90` 返回 `error`，传已初始化 SOC 名返回 `success`。`848bea7` 增加哈希绑定的运行目录内 TE wheel 适配，仅修正底层更新参数；OMG 仍选 `kirinx90`。`0acfeea` 的末态绑定依据安装 C++ 的两个 `GlobalTensor<half>` 缓冲区，不是凭日志猜 dtype。
- **2026-11-09 的六输入诊断又出现漏传 `--tbe-diagnosis`。** `334cb6e` 补齐 CLI、自动入口和编译器之间的传递，随后用户回传 T64/T1 编译和设备执行完成，worker 审计 `APPLIED`。该诊断 q/k/v/g/beta/初态、双输出均 FP16，不带 `effective_length`，不证明 master 七输入语义等价。
- 两条成功路径均有 `--use_origin_format=true`；早期融合探针还有 `--weight_data_type=FP16`。最新六输入诊断不强制权重 dtype。保留这些已验证条件，不把其中任意一个参数独立宣称为 SIGSEGV 的修复。

当前失败边界与上述基线有实质差异：TorchAir 七输入、静态 T256、g 为 FP32、末态声明 FP32、只接出 core，原失败命令未带 `use_origin_format`。原失败批次的 TBE 记录是 `PASSTHROUGH`（请求本来就是 `Ascend310B1`），且 CastT prebuild 已成功；不能把 `NOT_OBSERVED` 当成旧名称错误再次发生，也不能凭这一条记录确认所有 GDN worker 均无问题。入口对照失败后，优先核对这些已知图合同和编译配置差异；保持当前 AIR、golden 和七输入语义，不直接删除有效长度或将 FP32 数据改成 FP16 来冒充修复。

后续 `accuracy-gdn-origin-005` 已实测 `--use_origin_format=true`：相同 AIR 仍返回 SIGSEGV，CastT prebuild 成功，末条完整主进程日志进入 GDN OpPreBuild。这排除了“单加此开关即可修复本例”，没有定位崩溃函数；停止重复同项尝试，优先执行四层整图 dump。

2026-11-10 用户回传的 AIR 和 DDK `100.520.020.010` 日志支持以下判断。普通 Norm／Conv1d 的导出修正仍待新 OMG 结果；这些编译证据不代表板端数值通过。

| 边界 | 已观察的失败点 | 当前处理 |
| --- | --- | --- |
| 普通 Norm | FP16 `weight + 1` 的 Add 常量计算报类型不支持，CPUCL 常量子图报输入类型不一致，最终 Add tiling 报 `inputTensorPtr is NULL` | 导出前按原权重 dtype 计算一次 gamma，保留原 `npu_rms_norm` 调用及 epsilon；新 AIR 必须直接使用常量 gamma 且无 Add |
| Conv1d | 导出的 Conv2D 输入是 `[1,8192,259]`，filter 已为 `[8192,1,1,4]`，验证报两者 rank 不同 | 用显式 `unsqueeze(2) → conv2d → squeeze(2)` 表达；保留原 FP32 输入、权重、groups 和 bias，新 AIR 检查两端四维且高度为 1 |
| QLinear | x2 为 INT32 `[2048,1024]`、scale 为 FP32 `[8,8192]`；V4444 推导输出 `[256,1024]`，下游要求 `[1,256,8192]` | 已定位逻辑宽度与打包载体宽度冲突。还需所选 DDK 的 `InferShapeForQuantBatchMatmulV4444`、注册和 tiling 合同；不修改 golden 宽度或猜测解包规则 |
| ConvState | 内置推形成功，但本次入口下各 store 均不支持，强制分到 NPU 后找不到 implementation/format config | 先比较入口环境与注册选择，不能推出整个 DDK 全局缺少算子。Python fake 的第二输出为三维，DDK 推为 `[1,8192,1,259]`，也是待对齐的接口差异 |
| GDN | OMG SIGSEGV；此前 FP32→FP16 CastT prebuild 成功 | 崩溃函数仍需 IR / 调用栈定位。保持七个输入及原属性，不凭日志末行改 dtype 或更换 `kirinx90` |
| Gated Norm | OMC 已成功 | 保留原直接 gamma 路径；它的成功不能证明普通 Norm 相同 dtype/rank 组合可用 |

所有 case 的 TBE 适配审计均未证明本次适配实际应用；`NOT_OBSERVED` 不能写成已修复。普通 Norm 的 gamma 折叠仅接受已检查的 CPro forward AST，若原实现改变会停止导出。`export.json.export_adjustments` 保存原 weight 与折叠 gamma 的 dtype、shape、字节数和哈希；`export_graph_check` 保存新 AIR 的结构检查结果。

### 仅重导、重编普通 Norm 和 Conv1d

复用已有捕获目录，在导出服务器执行；不需要重新运行完整模型采集。`--kinds norm conv` 仅选择这两类小图，Gated Norm 不在其中。

```bash
AIR_RETRY="$AI_RUN_DIR/target-layer-accuracy/air-norm-conv-002"
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" export \
  --capture "$AI_RUN_DIR/target-layer-accuracy/golden" \
  --source-root "$TARGET_SOURCE" --converter-script "$CPRO_EXPORT_SCRIPT" \
  --kinds norm conv --output "$AIR_RETRY"
```

将该目录完整迁移到 DDK 服务器，设置 `AIR_RETRY` 为迁移后的目录，沿用原 DDK 与格式选项，写入新的 OMC 目录：

```bash
OMC_RETRY="$AI_RUN_DIR/target-layer-accuracy/omc-norm-conv-002"
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" compile \
  --batch "$AIR_RETRY" --output "$OMC_RETRY" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --tbe-diagnosis "$TBE_DIAGNOSIS"
```

主机回归覆盖全部有限 FP16 权重的 gamma 舍入、不同 Norm 输入 rank、Conv 的 prefill/decode、bias 和 FP16/FP32，且检查 PyTorch 图与源 AIR 结构。主机通过只证明这次导出表达改动；下一步以新 `compile.json` 确认 OMG 是否越过原失败点，再安排板端比较。

## 4. 批量上板与回收

编译批次为 `PARTIAL` 时也可以执行：本入口自动选择 `compile.json` 中的 `COMPILED` 项，无须删除失败项或重新编译。各子图使用独立捕获的 310P 输入，不依赖其他子图上板的输出。未编译项保留在 `uncompiled_cases`，即使已测子图全部一致，覆盖状态仍为 `PARTIAL`。

需要仅测其中几项时，可加 `--cases l00_c00_conv_silu_o0 l00_c00_linear_attn_norm_o0`；名称必须对应本批已编译项。执行、回收和解包均按绑定的 fixture 名单检查。旧版回收器只接受固定 Draft case 名单，更新代码后可直接回收已经启动的 Target 任务，无须再次 `--apply`。

复用已经工作的 `runtime/runtime.json` 和 HDC 环境。runner 新增 INT16 输入以及每次调用前恢复全部输入，需更新一次 native runner（不是重编完整模型）：

```bash
python "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$AI_RUN_DIR/runtime/native" --replace
```

使用原来构建 runner 的资源配置/环境变量。若原 native 目录不在上述位置，以 runtime.json 中的 `native_directory` 为准。其他任务结束后再更新该共享 runner。

```bash
python "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --compiled "$OMC_BATCH" --config "$AI_RUN_DIR/runtime/runtime.json" \
  --output "$AI_RUN_DIR/target-layer-accuracy/execution" --apply
```

此命令一次部署批量小图，板端逐图运行。导出时记录实际输入 Tensor 身份到源 AIR Data 名称的映射；执行前再按编译后的 GraphDef.input 顺序绑定输入；输出只有一个，不靠 shape 猜 K/V。runtime 检查缓冲区，输入 dtype 不匹配则停止。编译后输出 dtype 改变时按实际类型读取，并单独报告，绝不把 FP32 字节当 FP16。

每图 3 次预热、10 次采样，所有调用都恢复同一份输入。原先 runner 的 ReuseInput 模式不会用于此处的 state 重放。下列命令回收已启动任务，不重新执行或重编：

```bash
python "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --recover "$AI_RUN_DIR/target-layer-accuracy/execution" \
  --output "$AI_RUN_DIR/target-layer-accuracy/recovery-001"
```

若报告 PENDING，稍后用新的 recovery-002 目录再次回收。板端失败会停止后续图；已完成输出仍保留，缺失图不会算通过。

## 5. 输出批量比较 CSV / JSON

```bash
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" compare \
  --fixture "$AI_RUN_DIR/target-layer-accuracy/execution/fixture" \
  --native "$AI_RUN_DIR/target-layer-accuracy/recovery-001/native" \
  --output "$AI_RUN_DIR/target-layer-accuracy/comparison-001"
```

`summary.csv` 按调用、层号和 CPro 依赖顺序排列；`report.json` 包含每次调用的最大/平均绝对误差、相对 L2、首个差异坐标、最大误差位置、非有限值、输出 dtype 和重复稳定性。CacheUpdate 要求整个 cache 位一致，并区分原 golden 中保持不变的位置与被更新的位置。

浮点默认按[五项标准](target-accuracy-status.md)全部达标判 PASSED：cosine >0.99、relative-L2 <0.05、KL <0.005、RMSE <1.0、MeanRelativeError <1.0。缺项／不可计算不通过；显式 `--atol ... --rtol ...` 是额外 AND 条件。Cache 保留整缓存逐位合同，整数精确比较，每次调用均须通过且重复稳定。报告记录五项数值和失败项；已有 fixture/native 只需重跑 compare 到新的输出目录，无需重编／上板。`first_failed_boundary` 是这一批中按依赖顺序最早失败的边界，不自动认定它是完整图的唯一根因；独立分支没有唯一先后因果。

首批正确后再扩大层号或对失败附近做相邻小子图组合。前四层独立重放不能证明后 36 层、完整图融合或 CPro 连续缓存反馈正确。

### Attention 输出范围检查

旧 Attention 产物及 golden 已删除，原路径的只读检查命令已撤下。请从[当前测试页](target-batch-rerun.md)第 1.1 节重新采集，依次导出两张 Attention 图、编译和回收，再执行第 5.1 节的输出检查。新流程不依赖本页的历史目录。

检查器仍支持完整保留的 AIR／OMC、fixture、native：它读取实际源图的长度 Const，并按 head／16 行 Q 块统计误差，不裁剪或补零，也不改变五指标验收。`INSPECTED` 表示检查完成，不能证明精度通过、实际 tiling 任务数或设备布局。缺少原始文件时，历史 CSV 不足以运行这项检查。

输入身份观察点沿用 [TorchAir 官方 export_utils 实现](https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/python/torchair/_utils/export_utils.py) 的权重转换入口：只记录映射，不改变权重保存决策。已安装版本接口或 Tensor 身份无法确认时导出失败，不按参数编号、shape 或数值猜测 K/V。
