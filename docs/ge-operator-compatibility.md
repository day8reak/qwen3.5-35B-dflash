# 直接 GE 图的算子兼容性核查

2026-10-22，基线 `e496b5f`，核查 tiny/full 的 target/draft、序列长度8，共 **39种实际生成的 GE 类型**。完整结构仅生成描述，未读取35B权重。核查依据为[固定版本的官方 GE schema](https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/python/torchair/_ge_concrete_graph/auto_generated_ge_raw_ops.py)、蓝区 `kirin9030-next-6.0.1.0` 的实际推导注册和 AIR 转换表，以及用户内部 X90 的 OMG 日志。

**蓝区注册存在不等于内部 X90 的完整形状、dtype 和内核验证通过；蓝区缺少注册也不等于较新内部版必然不支持。** 当前核查不能代替内部 OMG 和 HIAI 上板测试。算子清单、数量和库身份见 [审计记录](../specs/ge-operator-audit.json)。

| 算子 | 核查结果与处理 |
|---|---|
| Cumsum | 内部日志明确缺少形状推导；名称与公开定义一致。本次将 FP16 GDN 衰减与整数位置前缀分别分解，见下文。 |
| BroadcastTo | 有推导注册；小型静态 FP16 广播存在 CPUCL 风险。本次提前展开为共享 Const，动态广播保留。 |
| ArgMaxV2 | **内部转换未验证**。蓝区只有 ArgMaxExt2，未找到 ArgMaxV2 转换表入口；但同一蓝区表也没有内部已能转换的 SelectV2，不能据此断言不支持。保留现实现，避免用 TopK 替换时改变并列最大值或 NaN 的选择语义。 |
| GatherElements | 找到实际 Infer、Verify 和注册对象；dim 属性、输出索引形状和输入数据类型匹配，无证据需要替换。 |
| TopKV2、GatherV2、BatchMatMulV2、SoftmaxV2、Cast、Sigmoid、Softplus | AIR 转换表有实际入口，对应旧 IR 推导存在。保留当前映射；此前 SoftplusV2 已改为受参数约束的 Softplus。 |
| SelectV2 | 用户内部日志已显示转换成 Select；保留三个输入显式同形及 BOOL 分支整数载体。 |
| ReduceSum、ReduceMean | 有推导注册；本次 FP16 前缀分解复用 ReduceSum。实际累加精度须在板端验证。 |
| Power、Swish | 有推导注册；日志中 Swish 已推导成功。保留 RMSNorm 的 Power 和静态小常量预计算。 |
| Reshape、Transpose、Slice、ConcatD | 有推导注册；用户 Reshape/Permute 日志已显示推导成功。形状/轴的 INT64 常量符合公开 schema，不因调试输出 `1 0 ...` 擅自改宽度。 |
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
