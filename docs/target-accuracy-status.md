# Target 精度状态

阶段与标准更新：2026-11-13；**CPro Attention Prefill／Decode 对独立数学参考五项全部通过**，relative-L2 分别为 **0.11175% / 0.11438%**，各自 10 次输出一致。原 Prefill captured golden 错误已确认。原整层 GDN FP32 请求批次 **6/6 编译成功，但末态 ABI 为 FP16**，准备阶段拦截，设备未执行。安装注册及源码确认使用 FP16 状态接口；用户已选择按 FP16 重新导出六图，精度结果待回传；GDN 48 个独立边界仍为 **43 PASSED / 5 FAILED**。Cache 为 Prefill **4/4 PASSED**、Decode **4/4 FAILED**，历史保留问题暂后处理。

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

[交互 HTML](assets/target-architecture-interactive.html)统一展示[各模块最新指标 CSV](../specs/target-latest-boundary-metrics.csv)中的 150 项：38 项 QLinear 使用 scale 适配后结果、8 项 Cache 使用最新浮点／整数对照、4 项 RoPE 使用 FP32 旋转后的零误差结果、2 项 Attention 使用独立参考验证后的五项结果，其余 98 项保留最近一次实测值。Attention 的 reference_basis 显式标为 INDEPENDENT_NUMPY_FP64_FINAL_FP16；其他行沿用 captured golden。新结果按模块替换旧值，Prefill／Decode 各 75 项；模块汇总不代表一次完整图执行。relative_l2 是比值，乘 100 才是百分比。

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

按用户最新指令，整层 GatedDeltaNet 使用 FP16 状态接口。五个投影、ConvState、卷积、SiLU、门控、Q/K 重复、GDN、gated norm 和 out_proj 在同一张图串联；初态和 writeback 数值转换为 FP16，g 保留原计算并在 GE 算子入口 Cast 为 FP16。输出 hidden、new_conv_state 和 FP16 recurrent_state，分别与同一次捕获的三个原始 golden 比较。raw-state golden 保持 FP32，误差包含状态 FP16 舍入，五项门槛保持。

使用显式 --gdn-state-dtype float16，源 AIR 检查算子端口与最终 writeback，编译后核对本次请求的 FP16 ABI。原 float32 选项仍保留严格检查，不会被自动降级。私有 exporter 的 stock FP32 Meta 通过 GE 输出 Cast 桥接，GDN 实际输出和层内 writeback 仍为 FP16；该桥接不恢复精度，也不证明内核内部累加类型。

保留捕获 T256/T1、effective_length、chunk64/1、权重和 golden；六图共 18 输出，见[当前 FP16 测试命令](target-batch-rerun.md#7-整层-gdnfp16-状态测试)。CSV 新增 actual_dtype/golden_dtype，FP16 state 对 FP32 golden 的 dtype_preserved=false 是声明的类型差异，数值和稳定性仍须通过。本地 CPU 与模拟 GE 检查已通过，新批次尚无 CPro 编译／设备结果，也不包含连续 Prefill→Decode 状态反馈。

此前六图均 COMPILED，但源 AIR／保留 dump 的 FP32 末态请求在最终 OMC 中变为 FP16。当前安装检查进一步确认：两份注册的 g/initial_state/last_recurrent_state 全为 FP16；C++ 第 1855／1858 行直接使用 half 指针，源码哈希与历史一致。扫描 PARTIAL 只因两个候选目录缺失。完整[接口观测记录](../specs/target-gdn-state-contract-results.json)来自用户回传，原始文件未在本地重新核验；尚未确认精确编译 pass、选中二进制或内部累加 dtype。

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

## Attention 独立参考验证

**L3 CPro Attention 的 Prefill、Decode 均对独立数学参考通过五项指标，10 次输出各自一致。原 captured golden 的 Prefill 输出错误；Decode captured golden 对独立参考通过。** 本次没有修改 CPro 算子、Q 调用、mask 或精度门槛。当前这两张小图退出复跑，下一步测试整层 GDN。

用户回传的 reference-summary.txt 基于同一捕获 Q/K/V、页表和 additive mask，以 NumPy FP64 计算 `softmax(scale × QKᵀ + mask) × V`，最后转 FP16。B=1、Q heads=16、KV heads=2、D=256、block_size=32；Prefill Q/KV 长度为 256/256，Decode 为 1/257，mask 行跨度均为 2048。没有裁 query、增加 mask 或覆盖原 golden。原始文件仍在用户机器，本地记录的是回传结果。[完整数值记录](../specs/target-attention-reference-results.json)。

| 与独立参考比较 | max_abs | relative-L2（百分比） | cosine | KL | RMSE | MeanRelativeError | 五项 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CPro Prefill | 0.0146484375 | 0.1117467475% | 0.9999993757 | 1.7123e-8 | 0.0004336238 | 0.0115462897 | PASSED |
| CPro Decode | 0.003173828125 | 0.1143773601% | 0.9999993464 | 5.8212e-8 | 0.0004121828 | 0.0037892429 | PASSED |
| captured golden Prefill | 1766.916015625 | 6014.1885000% | 0.2089101783 | 0.0133700640 | 23.3375481543 | 18.1030350203 | FAILED，五项全部超限 |
| captured golden Decode | 0.007080078125 | 0.2030495626% | 0.9999979614 | 8.6723e-8 | 0.0007317317 | 0.0103761936 | PASSED |

两组 native dump 的 invocation 0–9 分别具有相同 SHA，因此覆盖各自 10 次调用。此为同一捕获输入的重复稳定性，不是十个不同输入或连续生成验证。检查入口顶层 `numerical_validation=NOT_RUN` 保留了“未重跑原 captured-golden 验收”的含义；本次结论取自 `independent_reference` 的实际计算与五项比较，不把入口 INSPECTED 状态当作精度结论。

Prefill golden 错误有相互印证的数值证据：

- 整个输入 V cache 的绝对值最大仅 **3.494140625**。KV head0 范围为 [-2.078125, 3.494140625]，head1 为 [-2.4375, 2.39453125]。正常 softmax 权重非负且和为 1，输出各通道应落在对应 V 通道范围内。
- CPro Prefill、Decode 均无元素超出上述通道范围。Prefill captured golden 有 **588,094 个越界元素**，最大越界量 **1766.576171875**；不属于普通浮点舍入差异。
- head6/query239/channel15：独立参考 **2.083984375**、CPro **2.0859375**、captured golden **1769**。这一巨大差值位于 captured golden。
- captured golden 所有 head 的 query16–31 全零，但这些行的 mask 分别允许 17–32 个零偏置 key，其余为有限 -65504，完全没有全 -inf 行。全遮罩不能解释该整块零值。

原先以错误 Prefill captured golden 计算的高 L2 不能继续作为这张 CPro 图失败的依据。参考异常已确认，但具体发生在 310P 参考算子的分块计算、归一化、写回还是采集环节，尚未定位；不据此指认某条内核指令。

最新 CSV 与 HTML 已把 Attention 两行替换为 actual 对独立参考的整输出五项结果，并显式记录 `reference_basis=INDEPENDENT_NUMPY_FP64_FINAL_FP16`。原捕获与异常诊断记录保留，比较脚本的原 golden 不被静默替换。复核方法见[离线参考命令](target-batch-rerun.md#51-attention-输出检查)。

Q 长度常量与 source AIR 合同已核对，源码 step=16 下 T256 应为 mActual=16、T1 为 1；这次结果支持保留当前调用。实际 tiling 没有单独解码，数值通过不冒充取得 tiling 记录，也不外推其他 S、batch 或页表场景。

可以继续整层 GDN 与四层 Prefill 联调，完整模型仍需以下验证：

| 范围 | 尚需处理或验证 |
| --- | --- |
| L0–L2 GDN | 独立边界 43 PASS / 5 FAIL；按已确认的 FP16 接口重导出六张整层图，比较 hidden、conv state、recurrent state |
| 整图参考 | 原 Prefill L3 Attention 错误会影响依赖它的后续 reference 输出；修复参考路径或建立可信整图参考后重采 hidden／logits，不能直接沿用这些下游 golden |
| ConvState、MoE、Router、LM head | 仍需整层／整图数值覆盖；独立 residual 的 golden 输入比较不能证明内部正确 |
| Decode 状态反馈 | Cache 历史保留错误仍未解决；Attention 独立输入通过不代表连续 Prefill→Decode 状态正确 |

L0–L2 位于 L3 之前，可继续用现有捕获做独立 GDN 层函数测试；这种通过不自动保证由错误参考 logits 选出的 Decode token 或整网生成正确。已通过的 Attention、RoPE、Prefill Cache 不为本次 GDN 测试重复编译。

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

当前执行整层 GDN；Attention 两图对独立参考通过，RoPE 数值已对齐，Cache 恢复后仅复测 4 个失败的 Decode 图。下表与[风险 CSV](../specs/target-operator-accuracy-risk.csv)同步各模块最新结果，不把已通过模块加入复跑；新标准缺项不补造。

| 区域 | 已观察结果 | 当前处理 |
|---|---|---|
| CacheUpdate | 最新 Prefill 4/4 PASSED；Decode 4/4 FAILED，L2 99.69%～99.81% | 暂后：检查历史保留与 functional 输出契约 |
| LinearAttentionConvstateUpdate | 12 项小图编译失败，无数值；原 T64 整图已编译 | 保留失败定位；对照 RefData、双输出和消费者 |
| RoPE | FP32 旋转后四图零误差，旋转区／未旋转区均数值一致 | 独立边界已对齐；不再复跑，待整图集成验证 |
| ChunkGatedDeltaRule | 独立 Prefill core 1.56%～7.12%，raw state 12.10%～44.60%；整层六图已编译，末态 ABI 全部 FP16 | 安装接口已确认 FP16；按用户指令重导出 FP16 整层图，精度待回传 |
| AdnFusedInferAttention | 独立参考五项通过：Prefill L2 0.11175%、Decode 0.11438%；10 次输出一致；原 Prefill golden 错误 | 本轮小图退出复跑，继续整层 GDN；完整图参考需恢复可信路径 |

各小图使用独立捕获的 golden 输入，不统一归因于上游 QKV 误差传播。整缓存通过也不证明图内 alias 或连续状态反馈已经通过。MoE、Router、共享专家、LM head 与完整 40 层仍待覆盖。[旧静态 SVG](assets/target-architecture-accuracy.svg)为历史快照。

[单模块复测命令与结果](target-batch-rerun.md) · [ConvState 编译对照](target-compile-triage.md) · [Attention 输出范围检查](target-batch-rerun.md#51-attention-输出检查)
