# Target 精度状态

阶段与标准更新：2026-11-13；GDN 相邻独立边界最新回传 48 项五指标，**43 PASSED / 5 FAILED**，其余模块保留 2026-11-12 最近结果。当前准备整层 GDN FP32 state 和 Attention Prefill／Decode 共 8 图、20 输出，尚无该组设备数值。RoPE 四图已数值对齐，QuantBatchMatmul scale 适配与 RoPE FP32 旋转改动均已记录。Cache 最新结果仍为 Prefill **4/4 PASSED**、Decode **4/4 FAILED**，Decode 历史保留问题暂后处理。

**当前浮点标准（CPro / CHS 共用）**：按用户 2026-11-13 指定的下表，五项全部满足才通过；严格比较未四舍五入的值。此标准替换此前 cosine ≥0.9999 和量化 out_proj 单独 L2<2% 的判定。Cache 继续要求整缓存逐位一致，整数／索引／控制量继续精确比较。

| 误差对比算法 | 精度正常的参考标准 | JSON / CSV 字段 |
| --- | --- | --- |
| CosineSimilarity | > 0.99 | cosine |
| RelativeEuclideanDistance | < 0.05 | relative_l2 |
| KullbackLeiblerDivergence | < 0.005 | kl_divergence |
| RootMeanSquareError | < 1.0 | rmse |
| MeanRelativeError | < 1.0 | mean_relative_error |

`relative_l2 = norm(actual-golden)/norm(golden)`，0.05 即 5%；RMSE 为均方误差的平方根。KL 按 [MindStudio 定义](https://github.com/Ascend/msprobe/blob/4729e4e9b31a6f997e6f24e64dddd3fd9ec248ab/python/msprobe/msaccucmp/algorithm_manager/builtin_algorithm/alg_KullbackLeiblerDivergence.py)：两张量分别 min-max 归一化（常量跳过），截至至少 float32 epsilon，再各自除以和，计算 `D_KL(actual || golden)`。这是整张量分布比较，不对原始负数取 log，也不添加 softmax。

MeanRelativeError 为 `mean(abs(actual/golden-1))`；按 [MindStudio 定义](https://github.com/Ascend/msprobe/blob/4729e4e9b31a6f997e6f24e64dddd3fd9ec248ab/python/msprobe/msaccucmp/algorithm_manager/builtin_algorithm/alg_MeanRelativeError.py)，`abs(golden) <= float64 epsilon` 的项贡献 0，仍计入元素总数，报告另列此类元素数。全部使用 float64 归约。非空全零对照约定 cosine=1、relative_l2=0；仅一边为零、空张量、NaN/Inf 或必需指标不可用均不能通过。没有修改张量布局、裁剪有效区或改变 golden。

比较报告记录 `numerical_policy=DEVICE_FIVE_METRICS_V1`、阈值、各次调用的 `metric_checks` 和 `failed_metrics`。显式 `--atol/--rtol` 作为额外 AND 条件；Cache 逐位和重复稳定性检查也不会被五项门槛放宽。CSV 数值来自第一次调用，失败项合并全部调用，最终状态要求每次都通过。

历史 8 项 out_proj 满足旧 L2<2%，RoPE 四项数值零误差、Prefill Cache 四项逐位一致的观测保留；旧 CSV 缺少其余指标，不据此批量改成“五项通过”。有原始 fixture/native 时仅重新执行 compare 到新的输出目录即可，不需要导出、编译或上板。当前仓库只有 CPro 执行入口，公共指标实现不依赖后端；CHS 使用同一技能标准，不声称已有 CHS 实机验证。

Cache 工作恢复后只复测 4 个失败的 Decode 图（K/V × 浮点/整数）；已有通过观测的 Prefill Cache 和 QLinear 不再导出、编译或上板。

[交互 HTML](assets/target-architecture-interactive.html)统一展示[各模块最新指标 CSV](../specs/target-latest-boundary-metrics.csv)中的 150 项：38 项 QLinear 使用 scale 适配后结果、8 项 Cache 使用最新浮点／整数对照、4 项 RoPE 使用 FP32 旋转后的零误差结果，其余 100 项保留最近一次实测值。新结果按层、调用、阶段、边界与输出替换旧值，不再切换结果批次。Prefill／Decode 各 75 项；这是模块进度汇总，不代表一次完整图执行。relative_l2 是比值，乘 100 才是百分比。

数据来自用户回传表及报告片段，未在本地核验实际执行文件绑定。新表的原始精度按原样保存；V 明细的更多小数仅在下文记录，不填充到未附 case ID 的两行中。dtype_preserved／stable 全 TRUE 表示类型保持和重复稳定，不代表计算正确。

## GDN 当前独立测试

L0–L2 × Prefill／Decode 的 48 个独立边界已回传完整五指标：decay_beta 的 g/beta 12 项、repeat_qk 的 Q/K 12 项、gdn 的 core/raw state 12 项、recurrent_writeback 6 项、linear_attn_norm 6 项。43 项通过；5 项失败均为 Prefill GDN，全部重复稳定。最新 CSV 和 HTML 已按模块替换旧结果。

| 层 | Prefill core L2 | Prefill raw state L2 | Decode core L2 | Decode raw state L2 |
| --- | ---: | ---: | ---: | ---: |
| L0 | 5.772% · FAILED | 44.5964% · FAILED | 0.1346% · PASSED | 0.0683% · PASSED |
| L1 | 7.1232% · FAILED | 26.745% · FAILED | 0.1748% · PASSED | 0.4292% · PASSED |
| L2 | 1.5584% · PASSED | 12.0998% · FAILED | 0.1152% · PASSED | 0.0648% · PASSED |

表中状态取自最新五项判定，并非仅按 L2。L0/L1 Prefill raw state 同时未通过 cosine；其余三项失败只有 relative_l2 超限。六个 raw state 的 dtype_preserved 仍为 FALSE，Decode 数值通过不等于状态接口类型一致。

定位顺序为门控 g/beta → Q/K 重复 → GDN core 和 raw state → 状态转换、gated norm。每个图输入都是该边界的原捕获，所以上游小图误差不会自动传入下游小图。历史 repeat_qk 和 writeback 为零误差，只说明各自独立转换对齐，不证明 CPro GDN 末态或连续 Prefill→Decode 反馈正确。

按用户要求，下一组改测整个 GatedDeltaNet 层：五个投影、ConvState、卷积、SiLU、门控、Q/K 重复、GDN、gated norm 和 out_proj 在同一张图真实串联。输入捕获 state 数值转 FP32，initial_state、last_recurrent_state、层内 writeback 为 FP32，三个输出分别比较 hidden、new_conv_state 与原始 FP32 raw-state golden。仅改变输出类型不能证明内核累加精度；新流程检查源 AIR 和编译后可见 state 边界，拒绝用末端 FP16→FP32 Cast 掩盖半精度 producer。DDK 内核是否支持需实际编译验证。

保留捕获 T256/T1、effective_length、chunk64/1、权重和 golden；六图 GDN 加两图 Attention，共 8 图、20 输出，具体见[当前测试命令](target-batch-rerun.md)。本地 CPU 检查不替代 AIR、OMC 或设备验收；此组尚无新设备结果，也不包含连续 Prefill→Decode 状态反馈。

## QuantBatchMatmul：保留的改动

有效改动在 [candidates.py](../model/qwen35_moe_dflash/target/layer_probe/candidates.py) 的 `adapt_qlinear()` 和 `qlinear_ge_scales()`，通过 `--qlinear-scales fp16` 应用于测试导出：

| 位置 | 改动与原因 |
|---|---|
| weight scale 缓冲区 | 导出前执行真实 FP32 → FP16 数值转换并 contiguous，forward 使用 FP16 scale；不能只更改描述符而保留 FP32 字节 |
| PyTorch A8W4 前端 | per-token scale 保留 FP32，满足 torch_npu FakeTensor/meta 契约；提前 `.half()` 曾直接导致 AIR 导出失败 |
| GE converter → V4444 | 将 per-token scale Reshape 为 `[M]`，再显式 Cast 到 FP16，交给原 converter 生成 V4444 |
| 源 AIR 检查 | 核对 x1 INT8、packed x2 INT32、scale FP16、pertoken FP16，且保留可执行的 per-token FP32 → FP16 Cast |

INT32 载体内的 packed W4 字节及排布不变，V4444 原属性和可选输入交给原 converter。此修复对齐实际数据类型与当前算子契约，没有修改算子注册、tiling 或 kernel，也不代表仅切换一个 scaleType 就完成了字节转换。前端与 GE 输入的 dtype 要分层处理。

38 项 QLinear 已有设备回传，原数百至上万的 relative-L2 异常消失；本批 relative-L2 范围为 0.054541%～1.762492%。其中 8 项量化 out_proj 按旧 `<2%` 门槛通过，其余 30 项保持原始 METRICS_RECORDED，全部退出当前复跑范围。原始记录见 [QLinear 数值](../specs/target-qlinear-cache-candidate-metrics-20261112.csv)。这支持 scale 适配有效，尚不单独定位是哪一路 scale 导致旧错误。

合入完整 Target 时，须沿实际 exporter 的 `quant_model_inplace → replace_linear_to_QLinear → QLinear` 工厂及所选 converter 应用。当前源码的实际工厂可能来自 utils.py 导入的 NZ 类，只改 CPro 文件里的同名类可能无效；共用路径应保持 310P golden 不变。独立导出适配不等于生产整图已合入；MoE 和 LM head 不能继承这 38 项结论。

## RoPE：保留的改动与最新结果

本地私有 CPro/NZ 源码快照中，`Qwen3_5MoeRotaryEmbedding._init_rope_cache()` 已用 FP32 计算角度、cos 和 sin，`forward()` 查表后才转换输出类型。此处已符合此前 Draft 修复所要求的计算顺序；快照不证明服务器当前代码完全相同。

现有捕获点是 `apply_rotary_pos_emb(q,k,cos,sin)`，不含三角函数计算。新增 `--rope-precision fp32` 显式将四个输入提升为 FP32 后调用原 CPro helper，最后恢复各自捕获输出 dtype；原始输入和 golden 不变。它是旋转乘加候选，不能标记为 sin/cos 生成修复。源 AIR 检查乘加边两端类型；这也不证明 OMC 编译后的计算精度。

有效实现为 [rope.py](../model/qwen35_moe_dflash/target/layer_probe/rope.py) 的 `apply_fp32()`，经 [replay.py](../model/qwen35_moe_dflash/target/layer_probe/replay.py) 的 opt-in 分支导出。Q、K、cos、sin 先 `.float()`，调用原 helper；每个结果转换回其原捕获输出 dtype。partial rotary 的前 64 维做旋转，后 192 维保持原数值。实际四图输出均为 FP16；通用实现不假设输出 dtype 一定等于 Q/K 输入 dtype。

用户回传 `target/tests/rope-candidate/comparison/report.json`／`summary.csv`：

| case | 输出 shape | 旋转区数值相等元素 | 未旋转区数值相等元素 | 整体 max_abs / relative-L2 |
|---|---|---:|---:|---|
| l03_c00_rope_o0 · Prefill Q | [1,16,256,256] | 262144 / 262144 | 786432 / 786432 | 0 / 0 |
| l03_c00_rope_o1 · Prefill K | [1,2,256,256] | 32768 / 32768 | 98304 / 98304 | 0 / 0 |
| l03_c01_rope_o0 · Decode Q | [1,16,1,256] | 1024 / 1024 | 3072 / 3072 | 0 / 0 |
| l03_c01_rope_o1 · Decode K | [1,2,1,256] | 128 / 128 | 384 / 384 | 0 / 0 |

四图 dtype_preserved／stable 均为 TRUE，两个分区均无 NaN／Inf、无数值差异坐标。**已测四项数值完全对齐，不再上板复跑**；这是数值相等，不额外声称输出字节哈希与 golden 相同。原始状态仍为 METRICS_RECORDED／passed=null，因为当时比较器没有配置 RoPE 自动验收阈值；现有比较器按上面的五项标准判定，历史缺项不补造。

此前 17.56%～19.62% 的独立边界误差在此候选下消失，支持保留显式 FP32 旋转路径；未重新计算 sin/cos，不能归因于三角函数修复，也尚未确定旧图的具体错误指令。本地未重新执行服务器任务或核验该批完整文件绑定。合入完整 Prefill 后仍需检查串联输出和状态。

## AdnFusedInferAttention 修好后，能否跑 Prefill

**可以推进全注意力分支串联和四层 Prefill 联调，但不能据此宣布完整 Prefill 精度通过。** L3 的 Q/K/V 投影、RoPE、初始 Prefill Cache 已有独立结果；当前 Attention Prefill relative-L2 仍为 99.6669611%，Decode 为 0.2509511%。修好 Prefill Attention 后，应将已经验证的 QuantBatchMatmul／RoPE 改动接入同一整图，复查 L3 输出和 Prefill 末态。

| 范围 | 尚需处理或验证 |
|---|---|
| L0–L2 GDN Prefill | core relative-L2 分别为 5.7719723%、7.1231514%、1.5583993%；raw state 为 44.5964183%、26.744998%、12.0997546%，且 raw state dtype 不保持；没有新结果撤销这些偏差 |
| ConvState | 12 项独立小图编译失败、无数值；旧四层 T64 整图曾编译成功，因此小图失败不等于整图一定不能编译，整图状态更新仍须核验 |
| MoE、Router、共享专家、LM head | 不在这批独立数值验证覆盖内；独立 residual_mlp 使用 golden 分支输入，也不能证明 MoE 内部正确 |
| 整图与状态 | 验证修复确实进入生产图、最终 hidden／logits、KV 和 GDN 末态；T64 图需要对应 T64 golden，不能直接裁切 T256 捕获代替 |

旧 target-layer-accuracy 产物及 golden 已删除，当前先重新采集四层参考，只重建 Attention Prefill／Decode 两张测试图；整层 GDN 之后复用同一 capture。历史指标保留，但不冒充仍可读取的实际 dump。按用户最新要求保持 Attention 算子不变，先检查调用侧：全图与单模块 Q 都采用 packed `[B,heads×D/16,S,16]`，Q 长度在轴 2，actual_seq_lengths_q 传静态列表 `[S]`。导出检查 INT64 常量字节与 captured Q 长度一致，保留 allQLen 的独立 KV 遍历语义及原 mask 跨度。用户确认源码 DEFAULT_QUERY_SEQ_STEP=16，所贴实现的 T64/T256 Prefill 均应在 pa.Decode 内走 mActual=16，T1 Decode 为 1；当前无需为切分支再改 Q 调用。实际 tiling 的任务总数仍待核对，T64/T256 应为 64/256（B1、16 heads）；源码 Const 不等于已经观察到 host GetData 非空。[完整调用检查](target-batch-rerun.md#6-attention-prefill-调用合同)。

合入有效改动后还需四层 Prefill 的 hidden／logits 和末态验证。初始空缓存 Prefill 可以与 Decode 分开推进；Decode Cache 历史保留错误仍阻塞后续生成，多段／续接 Prefill 同样需要历史保留验证。已对齐模块不重复编译，独立小图结果不替代整图验收。

## 最新 Cache 结果

| 阶段 | 输出 | 浮点索引 | 整数索引 | max_abs（两者相同） | relative-L2（两者相同） |
|---|---|---|---|---:|---:|
| Prefill | K | PASSED | PASSED | 0 | 0 |
| Prefill | V | PASSED | PASSED | 0 | 0 |
| Decode | K | FAILED | FAILED | 8.6875 | 99.8086% |
| Decode | V | FAILED | FAILED | 3.494141 | 99.6897% |

“相同”仅指用户表中精度；尚未拿到两个版本的配对输出哈希，不能宣称逐字节相同。四个整数图已有数值，之前 INT64 Gather、INT32 FloorMod 和 Mul 元数据问题不再列为本轮活动阻塞；这不等于确认了服务器所用源码及 OMC 哈希。

## Decode Cache 原因分析

Prefill 与 Decode 调用同一个 CacheUpdate，契约都是更新指定位置并返回完整缓存；区别主要在新增 token 数和旧历史。Prefill 写入多个 token，首次调用的旧缓存通常为零；Decode 通常只更新一个 token，并保留此前大量非零历史。续接 Prefill 同样需要保留历史，不能把“Prefill 通过”理解为算子已经满足所有历史保留要求。

新增片段与 Decode V 的数值相符；它没有 case ID／输出哈希，不能分别绑定到浮点或整数版本。贴出的重复 invocation 数值一致：

| 字段 | 值 | 能说明什么 |
|---|---:|---|
| actual／golden shape、dtype | [64,32,32,16]，FP16 | 本次比较覆盖完整 1,048,576 元素 |
| exact_elements | 917,610 | 数值不一致共 130,966 个 |
| mismatched_preserved_elements | **130,966** | 所有数值不一致都落在 golden 相对输入未变化处 |
| mismatched_updated_elements | **0** | golden 相对输入变化处没有数值不一致 |
| max_error_coordinate | [0,0,0,6] | 最大误差在缓存前部 |
| actual_at_max／golden_at_max | **0／3.494140625** | 至少这个应保留的值被读成零 |
| golden_l2／error_l2 | 173.94256735162648／173.4027825929186 | 误差范数几乎达到完整 golden 范数 |
| relative_l2 | 0.9968967644497468 | 与表中四舍五入的 V 结果吻合 |

[比较器](../model/qwen35_moe_dflash/target/layer_probe/batch.py)按输入与 golden 的逐位相等位置划分 preserved，分区不一致计数使用数值比较。它区分的是 **golden 未变化／变化区**，并非直接按物理写入范围划分；一次写入若与旧值相同，也会落入未变化区。整缓存通过门槛仍是逐位一致。这里 130,966 = 1,048,576 − 917,610，因此所有已报告的数值不一致已被未变化区解释；不能额外声称整个历史区每个值都为零，或所有写入位置都已逐位通过。

这组证据说明当前首要问题是**输出没有完整保留旧缓存**，而不是继续调整浮点／整数索引公式。两个索引版本共用 cache、updates、functional converter 和执行输出路径；共同路径仍然会保留同一个错误。共有的输入打包或目标槽位错误也不能只凭汇总完全排除。

### 源码支持的机制与尚缺的证据

| 环节 | 已检查行为 | 对定位的含义 |
|---|---|---|
| [capture.py](../model/qwen35_moe_dflash/target/layer_probe/capture.py) | 调用前将输入复制到 CPU；调用后捕获被原位更新的 cache | 没看到因保存同一可变对象而让 before 自动变成 after 的代码问题；实际文件仍需绑定核验 |
| [replay.py](../model/qwen35_moe_dflash/target/layer_probe/replay.py) | clone 输入 cache，传入更新，再返回同一局部 cache | 不能用生产代码返回旧 .to() 对象的风险直接解释这个小图 |
| 原导出脚本的 functional converter | 以自定义 CacheUpdate 输出承接 functional 结果；此前 AIR 为 Identity(cache) → CacheUpdate → NetOutput | 输出名称叫 x 不等于运行时输入输出存储别名；clone 在 AIR 中成为 Identity，也不保证物理复制 |
| [tensor_probe_main.cpp](../targets/cpro/tools/tensor_probe_main.cpp) | 每次调用重新复制输入；默认输出初始化为 zero；保存 CopyOutput 的结果 | 每次输入应含捕获的旧历史；该工具读回的是输出缓冲区，不是更新后的输入 |
| [hiai_executor.cpp](../targets/cpro/runtime/src/hiai_executor.cpp) | 分别创建输入、输出张量；构造时清零输出；SetInput 仅复制输入 | 若内核只写更新位置、输出又未继承旧输入，则输出历史会保持零，符合已观察的失败形态 |

**首要机制假设：原位／局部写入的 CacheUpdate，被作为返回完整独立缓存的 functional 输出使用，历史复制或别名契约没有落实。** 这是由源代码与数值共同支持的优先方向，还没有目标内核、OpDef／RefData 映射及实际 buffer 地址来确定责任所在。

若 Prefill 的未写区域本来为零，默认清零的输出即使漏拷历史，也可能通过；Decode 引入非零历史后便暴露问题。这能解释阶段差异，但仍应以该批输入内容确认前提。该完整缓存比较有同 shape、同 dtype 和明确分区计数，不能用来源不明的 Attention “27／64 token”例子解释或撤销此次失败。

### 接下来修改哪里

1. **先补全已有四个 Decode case 的分区计数与配对输出哈希**：读取保留的 comparison/report.json 中 invocations 的 mismatched_preserved_elements、mismatched_updated_elements，以及 output_sha256。已有 V 片段不需要重复上板才能确认历史保留失败。
2. **核对 CacheUpdate 的真实输出契约**：若它声明返回独立完整缓存，输出须先保留全部输入历史，再应用更新；若仅支持原位写入，GE 注册、别名／RefData 和调用者必须遵循同一契约，并返回实际更新后的存储。不能只加一个会导出为 Identity 的 clone。
3. **若仍不能区分内核与绑定**，下一次有界诊断同时读取执行后的输入缓存与输出，并核对实际槽位。如果输入已完整更新、输出只有新 token，优先修复 functional／输出绑定；若输入保持原样、独立输出缺历史，检查独立输出实现是否遗漏全量复制。图内可能另有缓存副本，读回外部输入时需结合实际编译图判断。
4. 修复后只复测 4 个失败的 Decode K/V 图，两种索引均需整缓存逐位通过；Prefill 保留已通过记录，不再编译。连续状态反馈另待验证。仅预填输出、屏蔽历史区域或放宽 Cache 容差，不能作为精度修复。

本次新增导出阶段筛选 --phases decode，让后续清单只包含 4 个失败的 Decode Cache；报告、图示及命令同步更新。没有修改内核、输出初始化或 Cache 逐位比较门槛。对实际 CPro 源码的修复，应以核实的 CacheUpdate 契约为依据。

## 量化 out_proj 验收

按此前用户 **relative-L2 <0.02（严格小于）** 的记录，已有候选结果如下；这不是五项新标准的完整验收：

| 层／边界 | Prefill relative-L2 | Decode relative-L2 | 当前验收 |
|---|---:|---:|---|
| L0 linear_attn_out_proj | 1.3765734% | 1.7624918% | 两项通过 |
| L1 linear_attn_out_proj | 0.2138118% | 0.1442531% | 两项通过 |
| L2 linear_attn_out_proj | 0.7739864% | 0.7687055% | 两项通过 |
| L3 self_attn_o_proj | 0.2410854% | 0.2437747% | 两项通过 |

历史 42 项中原始状态仍是 38 METRICS_RECORDED、2 PASSED、2 FAILED。当前新增的验收结论仅覆盖 8 个量化 out_proj，其余 30 个 QLinear 低于 1% 但不借此新增未指定的验收门槛；QLinear 全部退出当前复跑范围。小图通过不代表完整模型或多步状态反馈已通过。

候选采用真实 FP16 weight scale，前端 per-token 保留 FP32，仅在 GE 边界 Reshape／Cast 为 FP16；原 packed W4 不变。生产合入须沿实际 QLinear 工厂：保留源码的 utils.py 导入 NZ QLinear，只改 CPro 文件里的同名类可能不生效；共用 NZ 路径不能同时改变原 310P golden。MoE／LM head 不继承本批结论。

## 高危算子重点定位表

当前先测 GDN 及相邻边界；RoPE 数值已对齐，Cache 恢复后仅复测 4 个失败的 Decode 图。下表保留各模块最近已知结果，其他模块不自动加入复跑；[风险 CSV](../specs/target-operator-accuracy-risk.csv)分别标注证据范围，量化 out_proj 按旧 2% 门槛、RoPE 按零误差观测移出高危项；新标准缺项不补造。

| 区域 | 已观察结果 | 当前处理 |
|---|---|---|
| CacheUpdate | 最新 Prefill 4/4 PASSED；Decode 4/4 FAILED，L2 99.69%～99.81% | 暂后：检查历史保留与 functional 输出契约 |
| LinearAttentionConvstateUpdate | 12 项小图编译失败，无数值；原 T64 整图已编译 | 保留失败定位；对照 RefData、双输出和消费者 |
| RoPE | FP32 旋转后四图零误差，旋转区／未旋转区均数值一致 | 独立边界已对齐；不再复跑，待整图集成验证 |
| ChunkGatedDeltaRule | 历史 Prefill core 1.56%～7.12%，raw state 12.10%～44.60%；6 个 raw state dtype 不保持 | 当前：GDN 及相邻边界共 48 图，建立五指标基线 |
| AdnFusedInferAttention | Prefill 99.67%、Decode 0.251% | 待处理 Prefill；27／64 token 例子来源未明，不能撤销指标 |

各小图使用独立捕获的 golden 输入，不统一归因于上游 QKV 误差传播。整缓存通过也不证明图内 alias 或连续状态反馈已经通过。MoE、Router、共享专家、LM head 与完整 40 层仍待覆盖。[旧静态 SVG](assets/target-architecture-accuracy.svg)为历史快照。

[单模块复测命令与结果](target-batch-rerun.md) · [ConvState 编译对照](target-compile-triage.md) · [Attention 输出范围检查](target-batch-rerun.md#51-attention-输出检查)
