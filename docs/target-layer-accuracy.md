# Target W4：310P golden → CPro 小子图精度定位

首批仅覆盖原模型第 0～3 层的 prefill 和第一个 decode。**导出阶段不构造整模型，不构造整层再裁图。** 每个进程只导一个 CPro 边界的一个输出，退出后再处理下一个。MoE 专家大子图不在本批导出范围内。

原 `inference.py` 和私有模型文件保留原样。新入口在模型已经加载、量化完成后的第一次 forward 安装采集点，继续运行原推理。golden 来自原 310P 实际调用；CPU 上的重算不作为 golden。新增工具代码可公开，原始源码、权重、配置和实际 dump 留在各自工作目录，不提交仓库。

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

CPro GDN 接收调用前缓存的 dtype；310P 传入内核前的 FP32 cast 不会被误当成 CPro 的输入合同。原导出 `utils.QLinear` 指定的替换类用于量化投影，不因 CPro 模型文件中有同名类就自动替换。INT32 打包 INT4、INT64 编码 scale 保留容器与字节，不进行 `.half()` 或重新量化。

Convstate 的辅助 `linear_update_tensor` 使用原 CPro 导出声明的零 INT32[128]，并在夹具记录该策略。`effective_length` 来自实际 310P 调用，不使用导出示例里的零长度。若原程序走外部卷积实现，未观察到有效 Conv1D 边界，采集标记 INCOMPLETE；不会编造 golden。

## 1. 在已验证的 310P 环境采集

以下变量由实际环境设置：`REPO_DIR` 指向本仓库，`TARGET_SOURCE` 指向原项目（含 inference.py、utils.py、models/），`AI_RUN_DIR` 指向仓库外的输出目录。激活原来的 PyTorch/torch_npu 环境，保留原来工作的配置和推理参数。

```bash
cd "$TARGET_SOURCE"
python "$REPO_DIR/targets/cpro/tools/inference_target_probe.py" \
  --source-root "$TARGET_SOURCE" \
  --output "$AI_RUN_DIR/target-layer-accuracy/golden" \
  --layers 0,1,2,3 --calls 2 -- \
  --config "$TARGET_CONFIG" --quant_mode enable --quant_bits 4 --max_token 2
```

`--` 后是原 inference.py 的参数，可以保留原来的其他参数。不要启用 pseudo_quant。输入 prompt/长度由原程序决定，不在采集入口中偷偷补齐或截断。原程序需完成 prefill 和至少一次 decode。`--calls 3` 可额外采第二个 decode，但 CPro 端仍是独立 golden 输入重放，不等于连续反馈验证。

先确认原程序仍生成正确内容，检查 `golden/manifest.json` 的 status 为 CAPTURED。`wrapper_outputs/0/output0.bin` 是原 wrapper 的第一个实际输出，可与原 golden 比对；它不是额外导出的整模型图。manifest 包含原命令、源文件身份、覆盖情况及逐 tensor 的 shape/dtype/SHA256。

```bash
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" list \
  --capture "$AI_RUN_DIR/target-layer-accuracy/golden"
```

## 2. 有 TorchAir 的环境逐个导出小 AIR

设置 `CPRO_EXPORT_SCRIPT` 为原来用于 CPro AIR 的导出脚本。仅导入其中的 converter 定义，不调用其 main/export_air。

```bash
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" export \
  --capture "$AI_RUN_DIR/target-layer-accuracy/golden" \
  --source-root "$TARGET_SOURCE" --converter-script "$CPRO_EXPORT_SCRIPT" \
  --output "$AI_RUN_DIR/target-layer-accuracy/air-batch"
```

每次只读取该小子图所需的 CPU 权重和输入。导出使用静态真实 shape；每个输出单独成为 AIR 唯一输出，不使用整层多输出图。batch 自带运行所需的输入和 golden，迁移到 DDK 服务器时保留完整目录结构，包括 `weight/` sidecar。

若先查指定边界，可增加 `--points convstate_update conv1d conv_silu decay_beta repeat_qk gdn recurrent_writeback`，或者使用 `--kinds norm norm_gated qlinear`。筛选只决定本次导出哪些图；采集数据保持完整。失败日志为 batch 根目录各 case 的 `.log`，不自动改 converter 或尝试其他精度。

## 3. DDK 服务器批量编译

在该服务器设置 `BATCH` 为迁移后的 air-batch 目录，并保证它位于该服务器的 `AI_RUN_DIR` 下。使用已选定的 X90 DDK，不换芯片或安装目录。

```bash
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" compile \
  --batch "$BATCH" --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --tbe-diagnosis "$TBE_DIAGNOSIS"
```

`TBE_DIAGNOSIS` 使用此前已经完成的兼容诊断目录；无须该兼容选项的环境可省略该参数。保持原编译使用的 format 选项；需要时显式传 `--use-origin-format`。不设置 weight_data_type，保持量化载体。小图要求权重内嵌 OMC。每个编译结果和原始日志分别保留在 `models/<case>/omc/`，不覆盖失败任务重试。

## 4. 批量上板与回收

复用已经工作的 `runtime/runtime.json` 和 HDC 环境。runner 新增 INT16 输入以及每次调用前恢复全部输入，需更新一次 native runner（不是重编完整模型）：

```bash
python "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$AI_RUN_DIR/runtime/native" --replace
```

使用原来构建 runner 的资源配置/环境变量。若原 native 目录不在上述位置，以 runtime.json 中的 `native_directory` 为准。其他任务结束后再更新该共享 runner。

```bash
python "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --compiled "$BATCH" --config "$AI_RUN_DIR/runtime/runtime.json" \
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

浮点容差尚未校准，默认 METRICS_RECORDED，不假称精度通过。若已有冻结的容差，可在 compare 显式传 `--atol ... --rtol ...`；不要看完结果后放宽阈值。`first_failed_boundary` 是这一批中按依赖顺序最早失败的边界，不自动认定它是完整图的唯一根因；独立分支没有唯一先后因果。

首批正确后再扩大层号或对失败附近做相邻小子图组合。前四层独立重放不能证明后 36 层、完整图融合或 CPro 连续缓存反馈正确。

输入身份观察点沿用 [TorchAir 官方 export_utils 实现](https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/python/torchair/_utils/export_utils.py) 的权重转换入口：只记录映射，不改变权重保存决策。已安装版本接口或 Tensor 身份无法确认时导出失败，不按参数编号、shape 或数值猜测 K/V。
