# 当前阶段：40 层 Prefill／Decode 集成准备

更新：2026-11-16。显式 ConvState 整层 GDN 六图：**11 PASSED / 7 FAILED，18 项全部重复稳定**。ConvState 6/6 通过；Decode 8/9 通过，仅 L1 hidden 的 relative-L2=7.41582% 超限。Prefill 3/9 通过，hidden 与 recurrent state 的六组已报告指标与旧基线相同。

保留有效适配：显式 concat／tail ConvState、单高度 Conv2d、FP32 RoPE 旋转后恢复输出 dtype、QLinear weight scale FP16 与 GE per-token reshape／FP16 cast，以及 GDN FP16 状态接口。已生成供私有源码使用的 patch；本地主机检查通过，未执行这份整模 patch 的目标编译或上板。

Attention Prefill／Decode 对独立数学参考的五项已通过，relative-L2 为 0.11175%／0.11438%；原 Prefill 捕获参考有误。RoPE 四项零误差。GDN 相邻独立边界仍为 43/48 通过；Cache Prefill 4/4 逐位通过、Decode 4/4 失败。QLinear 38 项已有适配后数值，历史缺五项指标的记录不升级为五项通过。

用户确认下一步跑原模型 40 层，包含 MoE、输出头和状态反馈；精度、可运行性、性能分别验收。先使用 B=1、有效长度等于静态长度的 Prefill 和单 token Decode；部分 padding 的 ConvState 更新未覆盖。新图必须重新读取编译 ABI，不能沿用旧输入槽位。40 层的 golden、设备精度和时延仍待取得，旧四层捕获不替代整模参考。

[详细结果](target-accuracy-status.md) · [交互 HTML](assets/target-architecture-interactive.html) · [完整流程](run-end-to-end.md) · [已完成的小图命令](target-batch-rerun.md)
