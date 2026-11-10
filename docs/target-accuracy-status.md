# Target OMC 与精度状态

用户已回传[修正环境后的批量复测](target-batch-rerun.md)，并进一步确认重导的 **Norm 20 项＋Conv1d 6 项全部编译成功**。累计 **146/158 项已编译，剩余 ConvState 12 项**。原 `environment-recheck-001` 的 120 项已有设备数值摘要，均 `stable=TRUE`，状态为 **2 PASSED、2 FAILED、116 METRICS_RECORDED**；新 26 项精度尚未回传。

更新：2026-11-10。新 26 项的执行入口在主机 prepare 触发旧 16 MiB I/O 限额，设备尚未运行。主机与 native runner 的单模型限额已同步改为 32 MiB，并通过主机边界检查；请按[上板文档](target-norm-conv-validation.md)重建一次 runner 后复用 OMC 重跑。此工具修复不改变下方数值结论；四层整图 dump 保留为后续方案。

主机侧已提供[批量 dump 比较脚本](target-dump-compare.md)，复用原四层入口的捕获清单。当前支持显式映射和离线比较；CPro dump 尚无样例，自动格式适配与节点映射待首批元数据。

本页整理用户提供的旧 compile.json、120 行比较摘要及新增 26 项编译成功的确认。新增项的逐 case 编译报告和 OMC 哈希尚未回传，本地主机未连接服务器或板子重验原始文件。累计计数覆盖同一批 158 个边界输出图的最新结果，不修改旧批次记录，也不代表整模型或 MoE 覆盖率。编译、执行、数值验收和重复稳定性分别记录；没有预先确定的浮点容差时，只记录误差。

## 1. 当前捕获边界批次

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
| 普通 Norm：input/post-attention、Self Attention Q/K Norm | 20 | 20 | 0 | 用户确认编译成功；设备／精度结果待回传 |
| ConvState 两个输出 | 12 | 0 | 12 | NOT_RUN |
| Conv1d | 6 | 6 | 0 | 用户确认编译成功；设备／精度结果待回传 |
| **合计** | **158** | **146** | **12** | **120 项有数值回传，另 26 项待验证，不能记整批精度通过** |

旧完整批次 manifest SHA256 为 `9f433d0092c93f5c156a5c6ab59d6d603dcef338b8132dc28b1e0f77fcf1b1e2`，source_batch 为 `../../../air-batch`。其 38 项检查摘要确认：Norm 仍含 Add 并在 Add 计算／tiling 失败，Conv 在输入／filter rank 检查失败，ConvState 在 format config 获取失败。用户随后确认重导 Norm／Conv 的 26 项已编译成功；上述旧 manifest 哈希不用于标识新 AIR。下一步只定位 ConvState，并保留所有成功模型。

代表产物：`l00_c00_gdn_o0` 70306 字节，SHA256 `9b4a621df37bdb583c85e212efc07e5e4fc28b898468b83170d2923a3a1d2352`；`l00_c00_linear_attn_in_proj_qkv_o0` 8698908 字节，SHA256 `45d90205590770405d2f08e756f256ddddf2b34cd106157c824ab9ba3ebb96d8`。哈希来自此前编译回传；本次比较摘要按相同层／调用／point／输出对应登记，尚未重新核验原始文件链。

此前 prepare 的 `arg6_1` 报错已在 `ff23e33` 修复检查器遗漏的 `INT16/DT_INT16` 映射。后续回传已覆盖 120 个编译成功项的数值结果，该准备阶段阻塞不再是当前进度。

### 本轮输出明显异常的边界

以下共 **51 个 case**，是根据误差量级及已有 Cache 精确门槛整理的排查清单，**不是已证明有 bug 的 51 个内核**。原始 `METRICS_RECORDED` 状态不改写为自动验收失败。除量化投影明确标注比值外，表中百分数为原始 relative-L2 乘 100。

| 边界 | 层／阶段／输出 | case 数 | relative-L2 | 当前结论 |
|---|---|---:|---|---|
| QLinear 量化投影 | L0–L2 qkv/z/b/a/out、L3 q/k/v/o；Prefill＋Decode | 38 | **483.04～11278.20（比值）** | 全部显著异常；不能当作正常量化舍入误差 |
| RoPE | L3，Q/K；Prefill＋Decode | 4 | **17.56%～19.62%** | 两阶段、两个输出均明显偏差 |
| GDN core | L0–L2，Prefill | 3 | **1.56%～7.12%** | 明显高于本批 Decode core；优先单列 Prefill |
| GDN raw state | L0–L2，Prefill | 3 | **12.10%～44.60%** | 大偏差，且 `dtype_preserved=FALSE` |
| FusedAttention | L3，Prefill | 1 | **99.67%** | 输出明显异常，max-abs=1766.914063 |
| CacheUpdate K/V | L3，Decode | 2 | **99.69%～99.81%** | 原报告 **FAILED**；整缓存逐位门槛未通过 |

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

| 类别 | 具体边界 | 范围 | case 数 | 目前可下的结论 |
|---|---|---|---:|---|
| 普通 Norm | input_layernorm | L0–L3，c00／c01 | 8 | 重导后用户确认编译成功；精度待回传 |
| 普通 Norm | post_attention_layernorm | L0–L3，c00／c01 | 8 | 重导后用户确认编译成功；精度待回传 |
| 普通 Norm | self_attn_q_norm／self_attn_k_norm | L3，c00／c01 | 4 | 重导后用户确认编译成功；精度待回传 |
| ConvState | convstate_update，o0／o1 | L0–L2，c00／c01 | 12 | format config 获取失败；实际实现选择／支持条件待定位 |
| Conv1d | conv1d，o0 | L0–L2，c00／c01 | 6 | 重导后用户确认编译成功；精度待回传 |
| **合计** | **原 38 项** | | **38** | **新增 26 成功，剩余 12 项 ConvState 失败** |

后续编译定位只需补取 [ConvState 两份代表日志](target-compile-triage.md#convstate只取两份代表日志)，定位 format config 之前的选择过程。Norm／Conv 导出与编译步骤已经完成，不再重复；[新 26 项上板与比较命令](target-norm-conv-validation.md)已提供实际新批次的执行、回收和报告步骤，精度仍待回传。MoE 专家计算尚未纳入本批小图，四层整图和连续状态反馈也未由本表验证。

### 旧环境编译记录

下面保留 `accuracy-find`、入口对照 `accuracy-entry-compare-004` 和 `accuracy-gdn-origin-005` 的历史结果；其中 QLinear／GDN、Norm／Conv1d 的旧编译失败均已有后续成功结果，当前状态以上方累计表为准。

| 模块／case | OMC 编译 | 设备执行 | 精度结论 | 当前证据与处理 |
|---|---|---|---|---|
| 普通 Norm：input/post-attention、Q/K Norm | 旧导出失败；修正后待回传 | 失败项未运行 | 未验证 | gamma 的 FP16 Add 常量计算／tiling 失败；已实现 gamma 预计算，尚无新 DDK 成功证据 |
| QLinear：`l00_c00_linear_attn_in_proj_qkv_o0` | 失败；入口对照仍失败 | 未运行 | 未验证 | 原日志输出推为 `[256,1024]`，Reshape 要求 `[1,256,8192]`；逻辑宽度与 packed INT32 载体宽度需核对 |
| 其他量化投影、out_proj | 本页未取得完整逐 case 状态 | 未确认 | 未确认 | 004 对照中 z/b/a 为 `NOT_SELECTED`，不等于失败，也不等于其他批次未编译 |
| ConvState：`l00_c00_convstate_update_o0` | 失败；入口对照仍失败 | 未运行 | 未验证 | 原日志在 format/implementation 配置阶段失败；第二输出须单独核对 |
| Conv1d | 旧导出失败；修正后待回传 | 失败项未运行 | 未验证 | Conv2D 输入三维、filter 四维；已实现显式补高度维的等价导出 |
| GDN：`l00_c00_gdn_o0` | 失败，SIGSEGV；加 origin-format 仍失败 | 未运行 | 未验证 | 最新 005 记录见下文；与历史成功探针不是同一合同 |
| Gated Norm：`linear_attn_norm` | 已回传成功 | 数值回传摘要待逐 case 绑定 | 尚不能登记数值通过 | 使用直接 gamma 路径；不能由此宣布普通 Norm 通过 |
| SiLU、decay/beta、attention 布局恢复及 gate 等 | 未取得完整逐 case 状态 | 未确认完整覆盖 | 未确认完整覆盖 | 保留原批次；优先从四层 dump 补齐实际覆盖 |
| MoE 专家计算 | 未纳入本轮小图导出 | 未运行本轮小图 | 未验证 | 四层整图内实际存在的 MoE 节点纳入 dump 检查；尚不宣称已覆盖 |

以下是此前会话回传摘要中的数值线索。当前仓库没有对应完整逐 case 报告及 OMC 哈希绑定，单列为“摘要待绑定”，不计入已核验 PASS 数量，也不套用到上表的 004／005：

| 模块 | 摘要中的编译／执行情况 | 摘要中的数值情况 | 登记状态 |
|---|---|---|---|
| repeat Q/K | 已有可执行 OMC 的结果 | 与捕获参考完全一致 | 摘要报告一致；待绑定层、阶段、case 和原报告 |
| recurrent state 回写转换 | 已有可执行 OMC 的结果 | 与捕获参考完全一致 | 同上；不代表 GDN 计算或连续状态反馈通过 |
| residual add | 已有可执行 OMC 的结果 | 与捕获参考完全一致 | 同上；不代表整层通过 |
| 激活／门控 Norm | 已有数值结果摘要 | 小误差 | 只有误差线索，没有已确认验收结论 |
| Self Attention RoPE | 已有数值结果摘要 | relative-L2 约 0.18～0.20 | 明显偏差线索；具体 Q/K、层及阶段待原报告绑定 |
| FusedAttention | 已有数值结果摘要 | 大误差／不稳定 | 待原报告复核；不能记通过 |
| Decode CacheUpdate | 已有数值结果摘要 | 缓存逐位不一致 | 失败线索；与下表独立 CacheUpdate 诊断分开保留 |

**旧环境 GDN 005 的确定结论：**相同 6361 字节 AIR（SHA256 `4cbc27e1ac15809e25ceb48bb865cc12844d5a06d0b60fc2c98075c6b10abd12`），DDK `100.520.020.010`，`isolated-kirinx90` 加 `--use_origin_format=true` 后仍返回 `-11`。CastT prebuild 成功，最后完整主进程日志进入 `ChunkGatedDeltaRule OpPreBuild`；这不能确定崩溃函数。TBE 审计事件为 `PASSTHROUGH`，请求／原生 SOC 都是 Ascend310B1，不能套用此前平台名称错误的修复结论。空的 `*_bt.log` 没有调用栈证据。本轮停止无新证据的同参数重试。

小图以各自独立捕获的 310P 输入运行，不能把它们的误差自动解释为上游误差传播。整图 dump 才能观察同一次执行中的真实传播关系。

## 2. 已成功编译的历史 OMC

这些产物可复用，但其成功范围不转移给新的 AIR、dtype、shape 或状态接口。

| OMC／来源 | 编译 | 设备执行 | 数值验收 | 重复稳定性／范围 |
|---|---|---|---|---|
| 2026-11-09 GDN T64，六输入、FP16 双输出 | 成功 | 完成 | `METRICS_RECORDED`；未校准容差 | 零／非零初态各 10 次有限且逐字节稳定 |
| 2026-11-09 GDN T1，同上 | 成功 | 完成 | `METRICS_RECORDED`；未校准容差 | 零／非零初态各 10 次有限且逐字节稳定 |
| 2026-11-09 CacheUpdate T64 | 成功 | 完成 | 整缓存逐位比较失败 | 连续／置换块两例均 10 次有限且稳定；返回 cache 合同待与 RefData 对齐 |
| 2026-11-09 CacheUpdate T1 | 成功 | 完成 | 整缓存逐位比较失败 | 块尾／块首两例均 10 次有限且稳定；写入区正确不等于整缓存通过 |
| 2026-10-27 GDN prefill8／decode1，七输入融合探针 | 两张成功 | 后续已回传加载／执行成功 | 后续结果存在明显偏差，未通过 | 不能宣称七输入本身不可编译；与当前 T256、FP32 g 的图不同 |
| 用户此前 40 层多 shape 整网 OMC | 用户报告曾编译成功 | 本页没有对应绑定结果 | 本页未确认 | 原命令包含动态 T64／T1、80 个 RefData；不能证明 T256 小图必然可编译 |
| 用户本次提供的 4 层 T64 整图 | 已提供转换命令；未收到本轮成功产物回执 | 本轮待确认 | 本轮待比较 | 有已成功产物就直接复用；没有才执行下方一次转换 |

11-09 四张 OMC 和详细数值见[历史设备回传](current-status.md#本轮设备结果与下一步)。GDN 相对 L2（core／state）分别为 T64 零初态 0.36166%／0.06612%、T64 非零初态 0.61187%／0.08209%、T1 零初态 0.05123%／0.05066%、T1 非零初态 0.05012%／0.05029%。这组参考是独立诊断参考，并非本次四层 310P golden。

已回传的 GDN T64 OMC SHA256：`c74fde9447192d08c9c72d0220c975f9280e322bfe64ff814497093b984c5fa4`；T1：`b96815646df5937aeaee3d6cb7cd4ec09130dfacb2b9b2e2215c9ecfb19b69d8`。早期七输入成功和后续数值失败分别见[编译记录](history/validation.md#融合gdn最新真实编译回传)与[板端记录](history/validation.md#自定义算子调用覆盖与gdn板端回传)。

## 3. 主线：四层整图一次 dump，离线批量定位

整图保留实际上下游连接、融合和 RefData 状态关系，直接检验真实执行路径。独立单输出小图会改变这些条件：小图不编译，不足以判断整图的对应模块也无法执行；整图编译成功也不证明精度通过。

执行顺序：

1. **锁定同一个四层 T64 用例。** 310P 与 X90 使用相同 checkpoint、W4 权重、token、有效长度、position、mask、页表、初始 conv/recurrent/KV state。沿用已工作的 `run_model_prefill.sh` 输入顺序、dtype、加载方式和外置权重目录。现有 T256 golden 不能直接截断后充当 T64 golden；若已有对应 T64 golden 则复用，没有才采集该四层用例。
2. **复用或编译一次四层 OMC。** 成功后保留 OMC 及全部引用 sidecar；无需先修好所有独立小图。本轮不导出／编译 40 层。
3. **运行同一个 Prefill，用 `--enable_item=2` dump 算子输入输出。** 沿用工作脚本的真实调用，在副本中增加开关，并把输出指向本次独立目录。处理、拉回本轮 dump 后，后续分析直接读取这份数据。
4. **先查第 0～3 层输出，再沿首个异常层的数据依赖定位。** 同一次 dump 可反复离线比较 Norm、投影、ConvState、Conv、GDN、Attention、残差与 MoE 的可观测边界。使用图拓扑／依赖关系，不能按日志时间顺序判断首个错误。
5. **只重放已经缩小的异常区域。** 若输入已匹配而输出偏离，在保留必要 RefData、双输出、消费者和 dtype 的最小连续子图中定位；有 SIGSEGV 再抓该次真实调用栈。未缩小范围时不继续轮流尝试编译参数。

| dump 比较结果 | 下一步 |
|---|---|
| 输入匹配，输出在冻结容差内匹配 | 该次调用、该边界通过；不推广到其他长度或状态 |
| 所有相关输入匹配，输出超差 | 首个局部候选；检查权重／属性、实现选择、布局转换及内核 |
| 输入已经超差，输出也超差 | 回溯输入生产者；先保留误差，不直接认定当前算子有错 |
| 普通输入匹配，但状态／控制输入未核对 | 仍不能认定局部计算错误；核对 RefData、有效长度及页表 |
| 融合内部张量未导出 | 先比较融合区域的输入输出；必要时只增加该区域的观测 |
| 编译或运行未完成／dump 缺失 | 记录对应阶段失败／覆盖缺失，不记数值通过 |

整数索引、token、控制字段按精确值／字节核对。浮点张量记录 max-abs、RMSE、relative-L2、cosine 和非有限值数，并按已冻结的各类容差判断；“输入足够接近”需保留实际误差，不能冒充相同输入。没有容差的项目保持 `METRICS_RECORDED`。

映射表至少保留：层号、Python 边界、编译节点／融合区域、输入／输出端口、真实 dtype、物理布局、逻辑 shape、有效范围、dump 文件及哈希。Q/K/V 不按相同 shape 猜槽位；NZ／padding 按已验证规则解包；packed INT4 或编码 scale 不按容器 dtype 做数值强转。融合和量化中间值没有一一对应参考时，先验证区域边界，不强行逐文件对齐。

第一轮仅定位一次 Prefill。它不能证明 Decode、部分有效长度或连续状态反馈正确；Prefill 定位完成后复用方法检查一次 Decode 和状态回灌。dump 模式的耗时不作为性能结果。

## 4. 用户服务器执行入口

DDK／OMG 编译在用户指定的 **35 服务器**执行；310P 推理和 golden 仍在此前的**权重服务器**。`10.136.166.147` 是历史参考命令来源，不是本轮 DDK 执行机。35 的完整地址不在此猜测。

下方 `TARGET_PREFILL_DIR`、`CPRO_DDK_ROOT`、`FOUR_LAYER_AIR` 分别填写 **35 服务器**上的实际工作目录、DDK 根目录和四层 AIR 路径；相对 AIR 路径以工作目录解析，保留引用权重。历史示例的 `/home/l00886453/DDK` 与此前失败的 `/home/l00886453/ddk_0806` 不能未经核对就当作 35 上的安装路径。保留实际路径／哈希。原命令省略 target/platform，保留其已知工作条件做整图对照，最终目标和实现归属以实际编译／运行证据核对。

**已有可运行四层 OMC 时跳过转换。** 需要编译时，使用原 AIR 所在工作目录和全新输出目录：

```bash
(
set -e
cd "${TARGET_PREFILL_DIR:?}"
: "${CPRO_DDK_ROOT:?}" "${FOUR_LAYER_AIR:?}"
FOUR_RUN="${AI_RUN_DIR:?}/target/tests/four-layer-dump-001"
test ! -e "$FOUR_RUN"
mkdir -p "$FOUR_RUN"
df -h "$FOUR_RUN"

source "$CPRO_DDK_ROOT/tools/tools_ascendc/set_ascendc_env.sh"
readlink -f "$CPRO_DDK_ROOT" > "$FOUR_RUN/ddk-path.txt"
sha256sum \
  "$CPRO_DDK_ROOT/tools/tools_omg/omg" \
  "$CPRO_DDK_ROOT/tools/tools_omg/master/omg" \
  "$FOUR_LAYER_AIR" > "$FOUR_RUN/compile-input.sha256"

"$CPRO_DDK_ROOT/tools/tools_omg/omg" \
  --model "$FOUR_LAYER_AIR" \
  --output "$FOUR_RUN/qwen3_5_35B" \
  --framework 1 --limit_partition_size=false \
  --save_weights_as_external_data=true > "$FOUR_RUN/omg.log" 2>&1
)
```

这段仅执行转换并保留日志；还需核对新生成的 OMC、sidecar 和编译成功记录，不能只看 wrapper 退出码。此前提供的 wrapper 末尾是 `exit $result`，未明确接收 OMG 的退出码。输入 AIR 的外部权重引用继续按原目录解析，不能只拷贝 AIR 到空目录。

板端使用新目录，例如 `/data/local/tmp/qwen3_5_moe/prefill/four-layer-dump-001/`，完整发送模型包及外置权重；保留旧的可运行模型。以原 `run_model_prefill.sh` 的实际底层命令为准，下面展示需要的参数形态，尖括号必须替换为原脚本中的真实值，不能直接运行：

```text
/data/local/tmp/tools_sysdbg/model_run_tool_internal \
  --model=<本次四层OMC绝对路径> \
  --input=<原脚本已验证的有序输入文件列表> \
  --output_dir=<本次输出目录> --enable_item=2 \
  <保留原脚本中的其余必要参数>
```

原脚本如设置迭代次数或状态初始化，首轮固定为一次 Prefill，避免多次调用混入同一 dump。当前已知用法来自用户提供的工作命令；工具的具体输出命名／目录仍按所选版本实际结果处理。`--enable_item=1` 的性能采集不能替代这里的 IO dump。

随后按用户给出的处理入口，在设备上执行：

```bash
/data/local/tmp/tools_sysdbg/data_proc_tool_internal --result_path=/data/local/tmp/
```

保留运行前后的文件清单和日志，确认本轮实际生成的 dump 目录，再用 `hdc file recv` 拉回该目录。不能假定 `output_dir` 就是所有原始 dump 的位置，也不要把 `/data/local/tmp/` 中历史 dump 混为本轮数据。

一次回传所需的是：四层编译结果／已有 OMC 身份、原运行脚本中的真实命令和有序输入说明、dump 文件清单及张量描述、对应 T64 的 310P golden 清单。模型权重和大体积张量留在服务器；GitHub 只保存状态表、比较摘要和证据索引。目前尚未取得本工具版本的 dump 样例，因此还没有声称自动解析和数值对齐已完成。已有逐位通过的独立结果继续保留，新的整图结果按本次 OMC、输入和状态单独登记。
