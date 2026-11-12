# Target 精度状态

更新：2026-11-12。FP16 scale／Cache 索引候选批次共 46 图，上一轮 OMC 为 **42 COMPILED／4 FAILED**，42 个成功项已回传数值。38 个 QLinear 有 **36 项 relative-L2 <1%**，仅 L0 out_proj 的 Prefill／Decode 仍为 **1.3766%／1.7625%**。Cache 浮点控制组 Prefill K/V 零误差，Decode K/V 仍约 99.8%／99.7%；整数四图没有数值。

[交互结构图](assets/target-architecture-interactive.html)默认显示[候选 42 项 CSV](../specs/target-qlinear-cache-candidate-metrics-20261112.csv)，可切换[修改前 146 项基线](../specs/target-boundary-metrics-20261112.csv)。两批独立显示，未测节点保持灰色；导出的 SVG 标明批次。[算子族风险 CSV](../specs/target-operator-accuracy-risk.csv)逐行注明证据范围。relative_l2 是比值，乘 100 才是百分比；[旧静态 SVG](assets/target-architecture-accuracy.svg)为历史快照。

编译统计与数值来自用户回传，实际执行文件绑定尚未在本地主机核验。42 项状态为 38 METRICS_RECORDED、2 PASSED、2 FAILED，dtype_preserved／stable 均为 TRUE。浮点项没有冻结验收容差，不改写为 PASSED；重复稳定不等于正确。

## 与此前结论的区别

| 本轮范围 | 与基线相比 | 当前 relative-L2 |
|---|---|---:|
| 36 个 QLinear | 从巨大偏差降至 <1% | 0.054541%～0.7739864% |
| L0 linear_attn_out_proj，Prefill | max_abs 降至 0.006103516 | 1.3765734% |
| L0 linear_attn_out_proj，Decode | max_abs 降至 0.004150391 | 1.7624918% |
| Cache 浮点索引 Prefill K/V | 仍为整缓存逐位 PASSED | 0 |
| Cache 浮点索引 Decode K/V | 与旧结果在回传精度下一致，仍 FAILED | 99.8086223%／99.6896764% |

候选 Prefill／Decode 各 21 项，超过图示 1% 阈值分别为 1／3 项。修改前基线保留 146 项、51 项 >1%（Prefill 28／Decode 23），其中 GDN、RoPE、FusedAttention 及 Norm 等未被本轮覆盖。两批不能相加或写成一次整图验收。

### 高危算子重点定位表

下表汇总定位优先级：**QLinear／Cache 浮点项来自新候选，其余数值来自修改前基线**，不代表一次联合运行。数值行数量为“问题输出数／已测输出数”，C0 为“构建阻塞数／选中 case 数”，都不是整图内核数量。relative-L2 **统一为百分比**。

| 优先级 | 算子／定位区域 | 异常范围与数量 | relative-L2 | 首先检查什么 | 对照与边界 |
|---|---|---|---|---|---|
| **P1** | **QuantBatchMatmulV4444 所在 QLinear 链路，含 DynamicQuant** | 仅 L0 linear_attn_out_proj；Prefill **1/19**、Decode **1/19** | **1.3766%／1.7625%** | 检查 golden／误差范数、逐通道误差，再隔离 DynamicQuant、scale 舍入与累加差异 | 36/38 <1%；支持 scale 适配方向，两路 scale 同改不能单独定责 |
| **P0** | **CacheUpdate** | L3 Decode，K/V **2/2** | **99.69%～99.81%** | 分别比较当前写入区和未写历史区；核对 block/offset、RefData/alias、缓存初始化与输出回读 | Prefill **2/2 PASSED**；问题集中在本批 Decode，不能推广为全部调用失败 |
| **P1** | **RoPE 组合子图** | L3 Q/K；Prefill **2/2**、Decode **2/2** | **17.56%～19.62%** | 核对 position、cos/sin、partial rotary 维度、rotate_half、head/序列布局和实际输入绑定 | 两阶段全部已测输出异常；Slice/Concat/Mul/Add 的具体责任待拆分 |
| **P1** | **ChunkGatedDeltaRule：core** | L0–L2 Prefill，**3/3** | **1.56%～7.12%** | 核对 Q/K 归一化、g/beta、初态、chunk/effective_length，逐 chunk 比较首个偏差 | Decode core **0.115%～0.175%**；当前严重偏差集中在 Prefill |
| **P1** | **ChunkGatedDeltaRule：raw state** | L0–L2 Prefill，**3/3**，dtype 全不保持 | **12.10%～44.60%** | 比较写回前 raw state；核对实际输出 dtype、state 布局和累加精度，分开检查后续 FP16 转换 | 不用最终 writeback 零误差替代 raw state 验证，也不把全部误差归因于 dtype |
| **P1** | **AdnFusedInferAttention** | L3 Prefill，**1/1** | **99.67%** | 核对 QKV/NZ 布局、mask/scale、有效长度、padding、block table，按 query 检查 | Decode **0.251%**；Prefill 目前只有一个已测输出，需控制变量复现 |
| **P2** | **ChunkGatedDeltaRule：raw state dtype** | L0–L2 Decode，dtype **3/3** 不保持 | **0.065%～0.429%** | 核对参考与 OMC 实际输出 dtype、Cast 位置，以及多步状态反馈的误差变化 | 数值未超过 1%；与 Prefill 合计 6/6 raw state dtype 不保持，本轮候选未重测 |
| **C0** | **Cache 整数索引子图** | Prefill／Decode K/V，**4/4** AIR 检查失败 | **未执行** | 最新图为 INT32 Mul 输入、FP32 输出描述、Cast 回 INT32；已加入该乘法的局部 GE 元数据适配 | 保留 Gather／FloorMod／浮点路径检查；新 AIR、OMC 与精度均待验证 |
| **C0** | **LinearAttentionConvstateUpdate** | L0–L2、两阶段、两个输出：**12 项无数值；此前小图编译失败** | **未执行** | 对照成功整图的 RefData、双输出/卷积消费者及 hybrid 路径，与小图 Data、单输出及 single 路径的差异 | 原 T64 整图编译成功；当前是小图构建阻塞，没有精度结论 |

所有数值项使用各自捕获的 golden 输入，不能统一归因于上游 QKV 误差传播。表内是**重点定位边界，不是已经定责的内核 bug**；原始 `METRICS_RECORDED`／`FAILED` 状态保持不变。MoE 专家 `GroupedMatmulV555`、Router、共享专家和 LM head 尚无本批精度证据，继续列为待覆盖，不纳入已观察异常数量。

AdnFusedInferAttention 的“27 个有效 token／64 个物理 token”目前只是来源未确认的例子，不能据此撤销 Prefill 99.67% 的记录。当前单模块工具要求编译输出与 golden 的 shape 一致，并检查回读字节数；不会按较短长度自动对齐。相同 numel 仍不能证明 NZ 排列一致，padding 是否影响全量误差也待验证。先用[只读 Attention 检查](target-layer-accuracy.md#attention-输出范围检查)核对产生指标的真实产物，再确定有效 query 区域；原全量指标保留。

后续测试按用户确认仅保留全部 Cache 和明确 FAILED 项；当前候选批次因此缩为 [8 个 Cache 图](target-batch-rerun.md)。L0 out_proj 两项仍为 METRICS_RECORDED，不因图示 1% 阈值改判 FAILED，也不继续复跑。其他模块的旧指标仅作历史记录。

## QLinear 输入与权重检查

上一轮 46 图候选的 38 个 QLinear 已全部编译并回传数值。候选实际转换 weight scale 常量为 FP16，前端保留 FP32 `[M,1]` per-token，只在 GE converter 边界 Reshape＋Cast 到 FP16 `[M]`。packed W4 排布保持原样。结果支持这项接口适配，仍不能只凭两个 scale 同改后的汇总定位单一根因。

Cache 浮点控制组的 Prefill／Decode 四图也已编译。整数版先遇到 INT64 Gather，再遇到 INT32 FloorMod；此前 Decode V 日志已通过取行前 Cast、Gather、FloorDiv。乘减求余后，最新 AIR 为 **42 EXPORTED／4 FAILED**，失败集中在整数四图的浮点路径检查。用户回传的 Mul 两输入均为 INT32，输出缺少 `_meta` 且描述为 FP32，后接 INT32 Cast；这证明导出描述不一致，不证明设备的实际计算类型。

[官方 mul.Tensor 转换器](https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/python/torchair/_ge_concrete_graph/ge_converter/aten/mul.py)会在新建 Mul 后执行输出 dtype promotion，而[图规范化逻辑](https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/python/torchair/_ge_concrete_graph/fx2ge_converter.py)会把未定描述写成 FP32。这条路径与回传图一致；所选服务器安装版本仍以导出记录的源码哈希为准。`cache_index_export.py` 现在只对本候选的 INT32 向量乘标量 32 生成 typed Const／Mul，并在消费者建立之前设置来自 FX 操作数的元数据；其他签名拒绝，退出恢复注册。既不事后改 AIR 字段，也不放宽浮点路径检查。

### Target CPro 源码落点

根据保留的原源码与本轮候选，当前应区分以下修改：

| 位置 | 应用内容 | 验证边界 |
|---|---|---|
| `utils.py:replace_linear_to_QLinear` 的实际 QLinear 工厂 | 在 CPro W4 导出分支，把 `[K/256,N]` scale 数值转换为 FP16 contiguous buffer；forward 不再把它转回 FP32。保留 INT32 packed W4 | 38 项小图已测；L0 out_proj 尚有 1% 以上误差 |
| 导出器选择的 `npu_quant_matmul` GE converter | 保留 PyTorch 前端 FP32 per-token；在 GE 转换边界 Reshape `[M]`＋Cast FP16，再交给原 converter | 直接在 forward 把 per-token 转 FP16 会触发已见的 FakeTensor 错误 |
| `Qwen3_5MoeAttention.update` 的整数候选与 GE 转换 | `position.to(INT32)` 放在取第 0 行之前；整数商与乘减求余；页表保持 INT32；Mul 必须在转换时获得 INT32 元数据 | 非负位置且在 INT32／页表容量范围内；当前仅改小图候选，待 AIR、OMG 与整缓存数值验证 |
| `Qwen3_5MoeAttention.update` 的缓存对象 | 检查 `past_key_value.to(new_k.device)` 是否产生新对象；若发生拷贝，必须返回实际被写的缓存，并由调用者继续持有 | 原代码返回旧对象有条件性风险；本轮小图使用 clone 并返回同一 cache，不能把这行当成已确认的 Decode 根因 |

**不能只改 CPro 文件中同名的 QLinear。** 保留的 `utils.py` 在文件头导入 NZ 模块的 `QLinear`，`quant_model_inplace → replace_linear_to_QLinear → QLinear` 才是实际实例化链；NZ forward 还显式执行 `self.scale.to(torch.float32)`。小图工具通过实际函数 globals 追踪工厂并仅修改被测实例。生产合入应限定 CPro 导出，避免同时改变 310P golden 路径；不能批量修改共用 NZ 类或把 MoE／LM head 当成本批已验证。

整数表达式应为（当前 B=1、block_size=32，位置范围在构建输入时验证）：

```python
position = cache_position.to(torch.int32)
block_idx = position[0] // self.block_size
offset_in_block = position[0] - block_idx * self.block_size
target_blocks = self.block_table.to(torch.int32)[:, block_idx].reshape(-1)
```

对于已验证非负 INT32 位置，`block_idx * block_size <= position`，乘法不溢出且余数与 `%` 相等。该表达式规避 FloorMod，但仅主机／FX 为 INT32 不足以证明导出正确，还必须配合上述 GE 适配与 AIR 检查。整数版尚无数值，不能据此承诺修复 Decode Cache。

Decode Cache 下一步比较 **更新行** 与 **未写历史区**，同时核对输出绑定、输入复制和 CacheUpdate functional converter 是否完整保留历史。Prefill 通过而 Decode 整体 L2 接近 1，使历史区丢失值得优先检验，但汇总值尚不能证明它。源 `update()` 的 `.to()` 风险与小图 clone／functional 路径也须分开。

L0 out_proj 先查 `golden_l2`、`error_l2` 和逐通道分布，再隔离 DynamicQuant、scale 舍入及累加。相同输出投影类别的 L1／L2 已低于 1%，没有依据统一更改权重排布、group size 或补乘系数。GDN／RoPE／Attention 保留既有定位任务，不依据这批 QLinear／Cache 数据修改其公式。

以下只读命令检查已有 AIR、OMC、编译日志。静态代码推导自洽不等于实际加载的输入契约已验证；可用它核对当前候选的 scale 与 Cast。

在 35 服务器现有 Python 环境执行，`OMC_BATCH` 指向**产生本次数值的实际 OMC 批次**；以下默认值沿用候选批次。检查目录须为空或不存在，保留既有输入和编译产物。

```bash
export REPO_DIR=/home/w00949577/qwen3.5-35B-dflash
export AI_RUN_DIR=/home/w00949577/qwen3.5-35B-dflash-run
export CPRO_DDK_ROOT=/home/l00886453/ddk_0806/image/hiai_ddk/ddk_internal
export OMC_BATCH="${OMC_BATCH:-$AI_RUN_DIR/target/tests/qlinear-cache-candidates/omc}"
export QLINEAR_CHECK="$AI_RUN_DIR/target/tests/qlinear-contracts"

python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" inspect-qlinear \
  --batch "$OMC_BATCH" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$QLINEAR_CHECK"

cat "$QLINEAR_CHECK/summary.txt"
```

如已知本轮上板准备目录，在同一次命令中加 `--fixture "$EXECUTION/fixture"`，同时核对该 fixture 的模型、槽位和输入字节是否属于当前批次。未提供时明确保持未检查，不用相同文件名或尺寸推断同一组数据。

| 产物 | 用途 |
|---|---|
| `ports.csv` | 按 case／阶段列出 AIR、保留 IR、OMC 中可见端口的 dtype、shape、format、Cast／TransData／Transpose／Reshape 链；编译日志完整参数另列 |
| `report.json` | 常量完整字节哈希与导出时原权重哈希对照、权重对应端口、512 字节前缀的按类型解码、编译器身份和解析限制 |
| `summary.txt` | 全部 QLinear 的输入类型、权重绑定结果及未解析项，便于回传 |

重点看：

1. `weight_port_bindings`：x2 是否接到匹配 `W_q` 的常量、scale 是否接到匹配 `scale` 的常量，以及中间是否有转换。`EXACT_SOURCE_WEIGHT_BYTES` 只说明转换前源权重逐字节相同；没有匹配不能自行归因，应查看常量是否可见、被折叠或发生了 Cast。
2. `SOURCE_AIR` 对照 `COMPILER_BUILD_ARGUMENTS`：实际构建参数是否把两路 scale 变成了 FP16；保留完整参数序号及 AIR 端口名，不能把 optional 输入压缩后的序号误当成同一端口。日志交错损坏会标 UNPARSED，不猜测缺失字段。
3. scale 方向是否与模型维度吻合，packed 权重是否沿途经过 TransData／Transpose。检查依据是已观察的描述与转换链，I4ZN 字节前缀仅作对照，不能靠“看起来像 int4”证明逻辑 K/N 排列正确。
4. 如 OMC 封装内部节点，`OPAQUE_OR_FUSED` 表示不可见；编译参数也不是设备 GM 内容。若这些检查无差异，下一步采集 DynamicQuant 的真实 INT8／per-token 输出，并以相同输入、权重与 scale 比较 MatMul，继续验证实际数值。

命令不改源码、权重、scale 或 dtype。输入数据只有捕获的外部输入及源常量可读；**没有把这次检查冒充真实 MatMul 四输入 dump，也不宣称已经找到根因**。局部文件缺失会给出 PARTIAL 并继续其余 case。

## 覆盖边界

各小图使用独立捕获的 310P golden 输入，不能把 GDN／Cache／RoPE 等异常统一归因于上游 QKV。writeback／residual 零误差不证明其上游正确，也不证明 RefData 别名或多步状态反馈正确。MoE 专家 `GroupedMatmulV555`、Router、共享专家、LM head 和完整 40 层仍待覆盖。

[全部 AIR／OMC／上板／比较命令](target-batch-rerun.md) · [ConvState 编译对照](target-compile-triage.md) · [整图 dump 对比](target-dump-compare.md)
