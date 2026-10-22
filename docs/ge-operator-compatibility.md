# 直接 GE 图的算子兼容性核查

2026-10-22，基线 `e496b5f`，核查 tiny/full 的 target/draft、序列长度8，共 **39种实际生成的 GE 类型**。完整结构仅生成描述，未读取35B权重。核查依据为[固定版本的官方 GE schema](https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/python/torchair/_ge_concrete_graph/auto_generated_ge_raw_ops.py)、蓝区 `kirin9030-next-6.0.1.0` 的实际推导注册和 AIR 转换表，以及用户内部 X90 的 OMG 日志。

**蓝区注册存在不等于内部 X90 的完整形状、dtype 和内核验证通过；蓝区缺少注册也不等于较新内部版必然不支持。** 当前核查不能代替内部 OMG 和 HIAI 上板测试。算子清单、数量和库身份见 [审计记录](../specs/ge-operator-audit.json)。

| 算子 | 核查结果与处理 |
|---|---|
| Cumsum | 内部日志明确缺少形状推导；名称与公开定义一致。本次将 FP16 GDN 衰减与整数位置前缀分别分解，见下文。 |
| BroadcastTo | 有推导注册；小型静态 FP16 广播存在 CPUCL 风险。本次提前展开为共享 Const，动态广播保留。 |
| ArgMaxV2 | **内部转换未验证**。蓝区只有 ArgMaxExt2，未找到 ArgMaxV2 转换表入口；但同一蓝区表也没有内部已能转换的 SelectV2，不能据此断言不支持。保留现实现，避免用 TopK 替换时改变并列最大值或 NaN 的选择语义。 |
| GatherElements | 先前只核对 schema、注册及端口，未覆盖旧 IR 的数据类型限制。最新内部日志明确拒绝 INT32；draft 的整数 anchor 选择改用 GatherV2，导出端拒绝非 FP16 GatherElements。 |
| TopKV2、GatherV2、SoftmaxV2、Cast、Sigmoid、Softplus | AIR 转换表有实际入口，对应旧 IR 推导存在。保留当前映射；此前 SoftplusV2 已改为受参数约束的 Softplus。 |
| SelectV2 | 用户内部日志已显示转换成 Select；保留三个输入显式同形及 BOOL 分支整数载体。 |
| ReduceSum、ReduceMean | 有推导注册；本次 FP16 前缀分解复用 ReduceSum。实际累加精度须在板端验证。 |
| Power、Swish | 有推导注册；日志中 Swish 已推导成功。保留 RMSNorm 的 Power 和静态小常量预计算。 |
| Reshape、Transpose、Slice、ConcatD | 有推导注册；旧日志曾显示部分 Reshape/Permute 推导成功；最新 PID157562 明确显示 Reshape 将 `55 0` 改成 `-1 0` 后推导失败，现将该算子的 shape 参数改为受范围检查的 INT32。其他形状/轴参数保持原类型。 |
| Add、Sub、Mul、RealDiv、Maximum、Neg、Exp、Rsqrt、Cos、Sin | 有对应推导注册；未发现同类缺少推导函数的证据。 |
| Equal、Greater、Less、LessEqual、LogicalAnd | 有对应推导注册，保留 dtype 与布尔语义。 |
| MatMul、Data、Const | 有对应推导注册；Data 工厂前置修正已有用户 AIR 成功报告。 |
| FileConstant | 有 ConvertFileConstant 转换注册；真实全权重载荷及文件路径处理仍未验证。 |

36种类型有自动生成的官方 schema，实际发射的端口和属性名核对通过。Data/Const/Cast 使用已有手写基础接口。该结论是构图契约检查，不是39种算子的 X90 执行认证。

## 本次修正的范围

GDN 的 FP16 前缀输入为 `-exp(A_log) * softplus(a + dt_bias)` 及其零填充，非正且当前每段长度64。各前缀用 Slice、ReduceSum、ConcatD 构成，只在前缀输出处存储 FP16。相较逐步 FP16 Add，它避免每加一次就做半精度舍入；不以三角矩阵乘法计算前缀，避免后续 NaN/Inf 经乘零影响此前位置。路径限定 `gdn_nonpositive` 模型域且长度≤256。任意正负混合的 FP16 扫描明确拒绝，因为 FP32 累加次序的变化在抵消输入上也可能超过现有容差。主机回归保持默认 FP16 门槛；这不是逐位等价或板端精度声明。

整数位置前缀使用分阶段移位相加，保持 INT32/INT64 的整数运算。验证包括溢出、正负轴、单元素和实际 draft 路径；不将整数前缀转成 FP16。

静态广播通过复制元素字节直接生成 Const，保留正负零、Inf 和 NaN 载荷。同内容和目标形状的常量全图复用；每个扩展载荷≤1MiB，累计≤8MiB。只查看已存在的内联常量，不读取 FileConstant/大型权重；超预算保留 BroadcastTo。常量在真正被输入或输出使用时才写入图，避免广播替换后留下不连通的原标量节点。

这些改动不修改 prepared 的权重、PyTorch 路径、OMG 参数或 HIAI ABI，不新增自定义 NPU 内核。需要[重新导出 AIR 后再转换 OMC](direct-model-air.md#内部-omg-兼容性修正与重试)；旧 prepared 可复用。原生压缩整数权重映射、完整35B质量与内存、速度仍是独立的未完成项。

## PID157562 补充证据与修正

`v974` 和 `v975` 的 ConvertReshape 日志新增了正维度改为 `-1 0` 的证据；这比此前单纯的 `1 0` 回显更具体。INT64 中这两个32位字对应4294967295，不能表示有符号-1。修正将全部 Reshape 构造点的 shape Const 显式存为 INT32，包括普通 view、矩阵投影展平/恢复和 argmax keepdims 恢复；官方 schema 允许两种宽度。未修改计算张量 dtype，也未推断所有 INT64 参数都不兼容。

Slice 类型错误紧接 ConstantFolding begin，确认是常量折叠阶段；具体失败节点仍未报告。新增小型 FP16 Slice/Reshape 离线复制移除 tiny 已知的32个静态专家切片。完整权重的巨大源张量依旧跳过，动态 FP16 Slice 保留原实现。TE 接口错配另通过 OMG 子进程 DDK 路径优先和 CANN 路径隔离处理，内部实际转换仍需重试。详细运行方式见直接 GE 指南第3节。


## PID157847：TE 初始化成功后的 BatchMatMul 拒绝

用户补充日志确认并行编译与 TE 内部初始化成功；最终拒绝 v1115 五维 BatchMatMul。该节点的 M/K/N、FP16类型和批次对应关系正确，前面的某个 custom_ascendc_lib 子库拒绝不等于所有后端均拒绝；已见 v10 的三维 BMM 常量子图完成 CPUCL 编译，但不能据此宣称动态三维 FP16 BMM 在 NPU 通过。

蓝区 `libai_npucore_ascendc.so` 的实际 BatchMatMul builder 继承 MatMul 支持检查，读取 `adj_x1` 后在为真时返回失败（嵌入源码位置 ascendc_matmul_op_builder.cpp:575，消息 `Not Support Matrix A Transpose`）。它未显示 rank>3/4 禁止条件，Python 后续检查源码在本地包缺失。新发射器对相同批次前缀合并成 rank3，保留每个矩阵，并显式执行左转置以避免该蓝区属性限制；内部 X90 的行为仍待复验。未采用逐批展开以避免完整模型数十万矩阵产生大量图节点，也不使用改变舍入边界的乘法加归约替代。

Kirin 蓝区存在 `get_tuling_soc_version` 平台映射和 Ascend310B 内部标识；不能只看 soc_version 日志就改变 kirinx90 目标。所有修改只在本项目导出端完成，未修改共享 DDK。证据见 [批量矩阵乘验证记录](../specs/direct-ge-bmm-verification.json)。


## PID158166：GatherElements 的整数类型限制

实际注册和公开 schema 的存在不能替代内部 IR dtype 验证。draftv4 的原始数据是 INT32 token IDs，用户日志明确拒绝。蓝区 GatherElementsVerify 的数据类型集合只有 FP16/FP32，索引允许 INT32/INT64；因此先前兼容性表对输入类型的判断不充分，本节更正该结论。

batch1 的 anchor 选择改为 INT32 vector `[S]` 加 INT32 indices `[1,1]` 的 axis0 GatherV2，得到 `[1,1]`，无浮点载体。蓝区实际 AIR 映射到 GatherV2D，前端允许 INT32数据、要求INT32索引；CPUCL支持表与 Run 分支均支持 INT32并调用DoGather<int>。NPU算子表未列出INT32数据，所以这里不承诺NPU执行。所有当前model的 GatherElements 已消除，普通 FP16嵌入/TopK Gather 保留。

target 已编译推进到 draft 属于用户日志及脚本顺序推断，完整产物与设备验证仍未取得。证据见 [整数 Gather 验证记录](../specs/direct-ge-integer-gather-verification.json)。
