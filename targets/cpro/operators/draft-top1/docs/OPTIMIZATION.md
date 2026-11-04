# Performance and generation prerequisites

hardware_reference_policy = use；用户没有选择opt_out。所选X90当前没有匹配的能力画像、pre-generation roofline或CAModel证据，时延预算与饱和度不可用。按 develop-custom-operators / estimate-cce-operator-roofline 工作流，写性能相关设备内核前还需核对实际目标画像；不套用9030容量或给出虚构毫秒数。

冻结逻辑工作量：15×248320=3,724,800个FP16输入，理想流式读取7,449,600字节；每行串行比较需要248319次，共3,724,785次。输出15个INT32共60字节。这只是逻辑账本，不等于实际指令/事务/GM流量。真实尾块、UB中间值、局部索引、workspace、同步、cacheline和输出尾写会增加物理工作。

最小正确实现→完整同接口CPU对照→3次预热+10次设备调用→相同热路径CAModel诊断→有证据后优化。独立Top1性能目标待完整基线后确定，不因此阻塞当前合同/环境工作。整Draft≤50ms不是此算子已实现的目标；不能从局部CPU7.4633ms或混合19–40ms直接推算收益。

输出稳定门槛：10/10字节一致。计时保留每次样本及分布，API墙钟与kernel-only分开；完整模型回归单独验收。当前没有NPU性能结论、近似变更、优化候选或可发布产物。
