# Target OMC 与精度状态

更新：2026-11-12。本轮源码对齐后 **90/158 项编译成功、68 项失败**，尚无本轮数值回传；失败分组及下一步见[编译排障](target-compile-triage.md)。90 项可按[完整批量命令](target-batch-rerun.md#4-批量上板)继续上板。

下表和结构图的数值属于旧批：此前累计 **146/158 编译成功、12 个 ConvState 失败**；142 项有数值（2 PASSED、2 FAILED、138 METRICS_RECORDED），另 4 个已编译项缺指标。新旧批次分别计数，不合并覆盖。

数据来自用户回传的编译清单、成功确认与比较摘要，本地主机未重验原始 OMC／native 输出。计数是独立边界输出图，不能当作完整模型覆盖率或内核数量。没有冻结浮点容差的项目保持 METRICS_RECORDED。

小图源码复查已发现普通 Norm 使用了 NZ 的 FP16 权重，而原 CPro 构造 FP32 权重；已修正为读取原 HF 参数。QLinear 实际替换工厂确为 utils 导入的 NZ 类，没有发现仅因类选择导致错误的证据。全部 16 类的处理与重新生成命令见[全批重跑](target-batch-rerun.md#2-全部-air-重新导出)。下列指标及图表仍属于旧导出，修正后的精度尚未回传；recurrent_writeback 的零误差仅验证转换值，不证明状态别名和连续反馈正确。

## 算子风险 CSV

[交互结构图 HTML](assets/target-architecture-interactive.html)可按层展开、切换 Prefill／Decode，包含层内两次残差与层间连接；下载后用浏览器打开。[静态 SVG](assets/target-architecture-accuracy.svg)和[逐边界指标 CSV](../specs/target-boundary-metrics-20261110.csv)一并保留。

图中以 relative-L2 >1% 标红：Prefill 28 项、Decode 23 项，共 51 个输出；该值只用于排查筛选，不改变验收状态。[算子族风险 CSV](../specs/target-operator-accuracy-risk.csv)共 47 行：35 个边界分组＋12 个覆盖缺口区域，包含 MoE、模型 IO 与状态反馈。它不是逐 GE 节点清单；覆盖缺口的 case 数 0 表示未测。CSV 中 relative_l2 是比值，乘 100 才是百分比。

## 当前捕获边界批次

范围：原模型第 0～3 层，已捕获的 Prefill T256／首次 Decode T1。旧完整批次记录的 DDK 为 `100.520.020.010`，平台 `kirinx90`，profile 为 `isolated-kirinx90`，保留 `use_origin_format=true`、内嵌权重；其 compile.json 为 `PARTIAL`、`ALL_MANIFEST_CASES`，`diagnostic_only=false`。下表将新 26 项成功确认计入累计覆盖，新批次的实际配置以其报告为准。

| 模块（c00／c01） | case 数 | 编译成功 | 编译失败 | 本批设备／精度 |
|---|---:|---:|---:|---|
| 量化投影：Linear Attention qkv/z/b/a/out、Self Attention q/k/v/o | 38 | 38 | 0 | 已回传；全部输出显著异常 |
| GDN core／state 两个输出 | 12 | 12 | 0 | 已回传；Prefill 明显偏差，Decode 误差较小 |
| repeat Q/K 两个输出 | 12 | 12 | 0 | 已回传；数值零误差 |
| recurrent state 回写 | 6 | 6 | 0 | 已回传；数值零误差 |
| attention／MLP residual | 16 | 16 | 0 | 已回传；数值零误差 |
| Conv SiLU、decay/beta、门控 Norm | 24 | 24 | 0 | 已回传；误差较小，尚无浮点验收门槛 |
| RoPE Q/K 两个输出 | 4 | 4 | 0 | 已回传；全部明显偏差 |
| Cache K/V | 4 | 4 | 0 | Prefill 2 项 PASSED；Decode 2 项 FAILED |
| FusedAttention、attention layout/gate | 4 | 4 | 0 | Prefill Attention 明显偏差；另 3 项误差较小 |
| 普通 Norm：input/post-attention、Self Attention Q/K Norm | 20 | 20 | 0 | 18 项 METRICS_RECORDED，dtype 保持且稳定；另 2 项待回传 |
| ConvState 两个输出 | 12 | 0 | 12 | NOT_RUN |
| Conv1d | 6 | 6 | 0 | 4 项 METRICS_RECORDED，dtype 保持且稳定；另 2 项待回传 |
| **合计** | **158** | **146** | **12** | **142 项有数值回传，另 4 个已编译项待回传，不能记整批精度通过** |

### 高危算子重点定位表

清单整理于 2026-11-11，数值仍来自截至 2026-11-10 的[逐边界指标](../specs/target-boundary-metrics-20261110.csv)。P0／P1 合计 **51 个 relative-L2 >1% 的边界输出**；另外单列 3 个 Decode 状态 dtype 问题和 12 个编译阻塞。数值行的数量是当前范围内的“问题输出数／已测输出数”；C0 的数量是“编译失败数／选中 case 数”，都不是整图内核数量。下表 relative-L2 **统一为百分比**，量化投影的原始比值范围是 483.04～11278.20。

| 优先级 | 算子／定位区域 | 异常范围与数量 | relative-L2 | 首先检查什么 | 对照与边界 |
|---|---|---|---|---|---|
| **P0** | **QuantBatchMatmulV4444 所在 QLinear 链路，含 DynamicQuant** | L0–L2 qkv/z/b/a/out，L3 q/k/v/o；Prefill **19/19**、Decode **19/19** | **48,304%～1,127,820%** | 先比 DynamicQuant 的 INT8 输出与 per-token scale；再用相同量化输入、packed W4、权重 scale 隔离 MatMul；核对 group256、转置、布局与槽位 | 38/38 边界异常；DynamicQuant 尚无独立数值，不单独认定其内核错误 |
| **P0** | **CacheUpdate** | L3 Decode，K/V **2/2** | **99.69%～99.81%** | 分别比较当前写入区和未写历史区；核对 block/offset、RefData/alias、缓存初始化与输出回读 | Prefill **2/2 PASSED**；问题集中在本批 Decode，不能推广为全部调用失败 |
| **P1** | **RoPE 组合子图** | L3 Q/K；Prefill **2/2**、Decode **2/2** | **17.56%～19.62%** | 核对 position、cos/sin、partial rotary 维度、rotate_half、head/序列布局和实际输入绑定 | 两阶段全部已测输出异常；Slice/Concat/Mul/Add 的具体责任待拆分 |
| **P1** | **ChunkGatedDeltaRule：core** | L0–L2 Prefill，**3/3** | **1.56%～7.12%** | 核对 Q/K 归一化、g/beta、初态、chunk/effective_length，逐 chunk 比较首个偏差 | Decode core **0.115%～0.175%**；当前严重偏差集中在 Prefill |
| **P1** | **ChunkGatedDeltaRule：raw state** | L0–L2 Prefill，**3/3**，dtype 全不保持 | **12.10%～44.60%** | 比较写回前 raw state；核对实际输出 dtype、state 布局和累加精度，分开检查后续 FP16 转换 | 不用最终 writeback 零误差替代 raw state 验证，也不把全部误差归因于 dtype |
| **P1** | **AdnFusedInferAttention** | L3 Prefill，**1/1** | **99.67%** | 核对 QKV/NZ 布局、mask/scale、有效长度、padding、block table，按 query 检查 | Decode **0.251%**；Prefill 目前只有一个已测输出，需控制变量复现 |
| **P2** | **ChunkGatedDeltaRule：raw state dtype** | L0–L2 Decode，dtype **3/3** 不保持 | **0.065%～0.429%** | 核对参考与 OMC 实际输出 dtype、Cast 位置，以及多步状态反馈的误差变化 | 数值未超过 1%；与 Prefill 合计 6/6 raw state dtype 不保持，不计入上述 51 项 |
| **C0** | **LinearAttentionConvstateUpdate** | L0–L2、两阶段、两个输出：**12/12 小图编译失败** | **未执行** | 对照成功整图的 RefData、双输出/卷积消费者及 hybrid 路径，与小图 Data、单输出及 single 路径的差异 | 原 T64 整图编译成功；当前是小图构建阻塞，没有精度结论 |

所有数值项使用各自捕获的 golden 输入，不能统一归因于上游 QKV 误差传播。表内是**重点定位边界，不是已经定责的内核 bug**；原始 `METRICS_RECORDED`／`FAILED` 状态保持不变。MoE 专家 `GroupedMatmulV555`、Router、共享专家和 LM head 尚无本批精度证据，继续列为待覆盖，不纳入已观察异常数量。

精度工作先处理 P0，再处理 P1；C0 保留为编译定位主线。可先选以下代表 case，修正后再扩到同类全部 case：量化 `l00_c01_linear_attn_in_proj_b_o0`；Cache `l03_c01_cache_k_o0`／`cache_v_o0`；RoPE `l03_c00_rope_o0` 与 `l03_c01_rope_o0`；GDN `l00_c00_gdn_o0`／`gdn_o1`；Attention `l03_c00_fused_attention_o0`。阶段相反的已有小图作为对照，保持输入、权重和比较口径不变。

QLinear 的共同现象是同层 qkv/z/b/a 或 q/k/v 的 max-abs 接近，例如 L0 Prefill 约 21972～21980。它提示优先核对共享的量化／权重／scale／数据解释链路，但摘要不足以确定具体哪一步错误。这里没有据此改权重、scale、算子或比较方式。

GDN 的原始 relative-L2 如下，均稳定；core 的 dtype 保持，raw state 的 dtype 均发生变化。尚无证据将 Prefill 的大偏差全部归因于 dtype 变化。

| 层 | Prefill core | Prefill raw state | Decode core | Decode raw state |
|---|---:|---:|---:|---:|
| 0 | 0.057719723 | 0.445964183 | 0.001345661 | 0.000682967 |
| 1 | 0.071231514 | 0.267449980 | 0.001747658 | 0.004292200 |
| 2 | 0.015583993 | 0.120997546 | 0.001152179 | 0.000648462 |

### 数值一致与误差较小的边界

| 分组 | 范围 | case 数 | 本轮观察／原状态 |
|---|---|---:|---|
| repeat Q/K | L0–L2，Prefill＋Decode | 12 | max-abs=0、relative-L2=0；METRICS_RECORDED |
| recurrent writeback | L0–L2，Prefill＋Decode | 6 | max-abs=0、relative-L2=0；METRICS_RECORDED |
| residual attention／MLP | L0–L3，Prefill＋Decode | 16 | max-abs=0、relative-L2=0；METRICS_RECORDED |
| Cache K/V | L3，Prefill | 2 | 整缓存逐位一致，原报告 **PASSED** |
| Conv SiLU | L0–L2，Prefill＋Decode | 6 | relative-L2 约 0.123%～0.187%；METRICS_RECORDED |
| decay g／beta | L0–L2，Prefill＋Decode，两个输出 | 12 | 约 0.040%～0.267%；METRICS_RECORDED |
| linear attention gated Norm | L0–L2，Prefill＋Decode | 6 | 约 0.164%～0.255%；METRICS_RECORDED |
| GDN core／raw state | L0–L2，Decode，两个输出 | 6 | 约 0.065%～0.429%；raw state dtype 不同；METRICS_RECORDED |
| FusedAttention | L3，Decode | 1 | 约 0.251%；METRICS_RECORDED |
| attention layout/gate | L3，Prefill＋Decode | 2 | 约 0.034%～0.053%；METRICS_RECORDED |

合计：**36 项数值零误差（含 2 项 Cache PASSED）、33 项误差较小、51 项输出明显异常，共 120 项**。“误差较小”是本轮观察分组，没有新增统一阈值；只有 Cache 的两项保留原报告 PASSED。零数值误差也不自动等于已逐位验证，浮点正负零等仍应区分。

所有小图均使用各自独立捕获的 310P 输入。writeback／residual 一致不证明其上游 GDN／投影正确；Prefill Cache 通过不证明 Decode Cache 保留旧状态正确。`stable=TRUE` 表示重复输出一致，不等于计算正确。原比较器只把显式门槛失败写为 FAILED，因此其“首个 FAILED”不能代表首个数值异常。

### 新增 26 项编译成功，剩余 12 项

本次用户回传的 [22 行原始指标 CSV](../specs/target-norm-conv-metrics-20261110.csv) 保留全部字段与精度。18 项普通 Norm、4 项 Conv1d 的 status 均为 `METRICS_RECORDED`，dtype 保持、重复稳定，error 均为空。以下百分数是原始 relative-L2 乘 100：

| 本次回传分组 | 数量 | max-abs 范围 | relative-L2 |
|---|---:|---:|---:|
| 普通 Norm，Prefill | 8 | 0.01953125～0.125 | 0.108389%～0.145771% |
| 普通 Norm，Decode | 10 | 0.00390625～0.03125 | 0.039795%～0.189404% |
| Conv1d，Prefill | 2 | 0.007720947～0.008674622 | 0.026080%～0.027210% |
| Conv1d，Decode | 2 | 0.003479003～0.004500389 | 0.021476%～0.026757% |

这批结果没有新增明显异常边界。仍缺数值回传的 4 项是 `l00_c00_input_layernorm_o0`、`l03_c00_self_attn_q_norm_o0`、`l00_c00_conv1d_o0`、`l00_c01_conv1d_o0`；它们已编译成功，不将缺少 CSV 解释为运行失败。缺项范围与此前 4 项代表批次一致，但没有其数值数据时不代填结论。

## 后续定位

先独立检查表中的四类自定义算子及 RoPE；ConvState 编译对照见[剩余编译问题](target-compile-triage.md)。量化链路中的 DynamicQuant、权重／scale、布局与输入绑定尚未拆开验证，不能仅凭边界输出定责 QuantBatchMatmulV4444 内核。

若要观察真实误差传播，使用同一输入、状态和长度的四层整图 dump，按[离线比较入口](target-dump-compare.md)绑定节点。已有成功整图为 T64，不能直接用本批 T256 golden 截断替代；MoE、LM head 和连续状态反馈仍待覆盖。

旧环境失败与历史单算子记录见 [Git 历史](https://github.com/day8reak/qwen3.5-35B-dflash/blob/459060e/docs/target-accuracy-status.md)。
