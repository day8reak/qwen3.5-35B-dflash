# Target 精度状态

更新：2026-11-12。最新 **8 个 Cache 候选均有数值**：Prefill 的 K/V × 浮点/整数 **4/4 PASSED**；Decode **4/4 FAILED**。整数索引已越过此前导出／编译阻塞，但没有改善 Decode 数值。补充的 Decode V 明细已将该片段的失败定位到**历史缓存保留区**；具体是内核复制、原位契约还是输出绑定仍需核验。

按用户最新验收口径，量化 **out_proj 的 relative-L2 <2% 即通过**。已有 8 项全部满足，包括 L0 Prefill／Decode 的 1.3765734%／1.7624918%。原始 CSV 的 METRICS_RECORDED 保留，当前验收结论另列；其他算子不套用 2%，Cache 仍要求整缓存逐位一致。后续仅复测 4 个失败的 Decode Cache 图（K/V × 浮点/整数）；已通过的 Prefill Cache 和 QLinear 不再导出、编译或上板。

[交互 HTML](assets/target-architecture-interactive.html)默认显示[最新 Cache 8 项 CSV](../specs/target-cache-candidate-metrics-20261112.csv)，可切换[历史候选 42 项](../specs/target-qlinear-cache-candidate-metrics-20261112.csv)和[修改前基线 146 项](../specs/target-boundary-metrics-20261112.csv)。三批独立显示，未测节点灰色；SVG 导出包含批次。relative_l2 是比值，乘 100 才是百分比。

数据来自用户回传表及报告片段，未在本地核验实际执行文件绑定。新表的原始精度按原样保存；V 明细的更多小数仅在下文记录，不填充到未附 case ID 的两行中。dtype_preserved／stable 全 TRUE 表示类型保持和重复稳定，不代表计算正确。

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

1. **先补全已有四个 Decode case 的分区计数与配对输出哈希**，使用[只读报告命令](target-batch-rerun.md#先读已有报告无需重编或上板)。已有 V 片段不需要重复上板才能确认历史保留失败。
2. **核对 CacheUpdate 的真实输出契约**：若它声明返回独立完整缓存，输出须先保留全部输入历史，再应用更新；若仅支持原位写入，GE 注册、别名／RefData 和调用者必须遵循同一契约，并返回实际更新后的存储。不能只加一个会导出为 Identity 的 clone。
3. **若仍不能区分内核与绑定**，下一次有界诊断同时读取执行后的输入缓存与输出，并核对实际槽位。如果输入已完整更新、输出只有新 token，优先修复 functional／输出绑定；若输入保持原样、独立输出缺历史，检查独立输出实现是否遗漏全量复制。图内可能另有缓存副本，读回外部输入时需结合实际编译图判断。
4. 修复后只复测 4 个失败的 Decode K/V 图，两种索引均需整缓存逐位通过；Prefill 保留已通过记录，不再编译。连续状态反馈另待验证。仅预填输出、屏蔽历史区域或放宽 Cache 容差，不能作为精度修复。

本次新增导出阶段筛选 --phases decode，让后续清单只包含 4 个失败的 Decode Cache；报告、图示及命令同步更新。没有修改内核、输出初始化或 Cache 逐位比较门槛。对实际 CPro 源码的修复，应以核实的 CacheUpdate 契约为依据。

## 量化 out_proj 验收

按用户 **relative-L2 <0.02（严格小于）**，已有候选结果为：

| 层／边界 | Prefill relative-L2 | Decode relative-L2 | 当前验收 |
|---|---:|---:|---|
| L0 linear_attn_out_proj | 1.3765734% | 1.7624918% | 两项通过 |
| L1 linear_attn_out_proj | 0.2138118% | 0.1442531% | 两项通过 |
| L2 linear_attn_out_proj | 0.7739864% | 0.7687055% | 两项通过 |
| L3 self_attn_o_proj | 0.2410854% | 0.2437747% | 两项通过 |

历史 42 项中原始状态仍是 38 METRICS_RECORDED、2 PASSED、2 FAILED。当前新增的验收结论仅覆盖 8 个量化 out_proj，其余 30 个 QLinear 低于 1% 但不借此新增未指定的验收门槛；QLinear 全部退出当前复跑范围。小图通过不代表完整模型或多步状态反馈已通过。

候选采用真实 FP16 weight scale，前端 per-token 保留 FP32，仅在 GE 边界 Reshape／Cast 为 FP16；原 packed W4 不变。生产合入须沿实际 QLinear 工厂：保留源码的 utils.py 导入 NZ QLinear，只改 CPro 文件里的同名类可能不生效；共用 NZ 路径不能同时改变原 310P golden。MoE／LM head 不继承本批结论。

## 高危算子重点定位表

当前 Cache 复测仅覆盖 4 个失败的 Decode 图，已通过的 Prefill 不再编译。下表其他数值为修改前基线的历史定位记录，不自动加入复跑；[风险 CSV](../specs/target-operator-accuracy-risk.csv)分别标注证据范围，量化 out_proj 已按用户 2% 门槛移出高危项。

| 区域 | 已观察结果 | 当前处理 |
|---|---|---|
| CacheUpdate | 最新 Prefill 4/4 PASSED；Decode 4/4 FAILED，L2 99.69%～99.81% | P0：检查历史保留与 functional 输出契约 |
| LinearAttentionConvstateUpdate | 12 项小图编译失败，无数值；原 T64 整图已编译 | 保留失败定位；对照 RefData、双输出和消费者 |
| RoPE | 历史 Q/K 两阶段 L2 17.56%～19.62%，METRICS_RECORDED | 历史记录，本轮不复跑 |
| ChunkGatedDeltaRule | 历史 Prefill core 1.56%～7.12%，raw state 12.10%～44.60%；6 个 raw state dtype 不保持 | 历史记录，本轮 Cache 结果不能解释它 |
| AdnFusedInferAttention | 历史 Prefill 99.67%、Decode 0.251% | 历史记录；27／64 token 例子来源未明，不能撤销指标 |

各小图使用独立捕获的 golden 输入，不统一归因于上游 QKV 误差传播。整缓存通过也不证明图内 alias 或连续状态反馈已经通过。MoE、Router、共享专家、LM head 与完整 40 层仍待覆盖。[旧静态 SVG](assets/target-architecture-accuracy.svg)为历史快照。

[Cache 当前报告与复测命令](target-batch-rerun.md) · [ConvState 编译对照](target-compile-triage.md) · [Attention 输出范围检查](target-layer-accuracy.md#attention-输出范围检查)
