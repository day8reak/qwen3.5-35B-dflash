# 当前阶段：Target GDN 独立边界精度定位

标准更新：2026-11-13；以下实测结果回传于 2026-11-12。**RoPE Prefill／Decode 的 Q/K 四项全部数值对齐**：整输出及前 64 维旋转区、后 192 维未旋转区均零误差，FP16 输出且重复稳定。保留显式 FP32 旋转乘加、最后恢复输出类型的改动；不是重新计算 sin/cos。原始 METRICS_RECORDED 状态保留。

**QuantBatchMatmul 保留的改动**：weight scale 在导出前真实转 FP16；per-token scale 在 PyTorch 前端保留 FP32，通过 meta 后在 GE 边界 Reshape 为 [M] 并 Cast FP16。packed W4 不变。38 项 QLinear 已有适配后数值，8 项量化 out_proj 曾按旧 relative-L2 <2% 口径通过。当前 CPro / CHS 统一五项标准：cosine >0.99、relative-L2 <0.05、KL <0.005、RMSE <1、MeanRelativeError <1；全部满足才通过，Cache 仍要求逐位一致。历史缺项仅用已有 dump 补算，不重编／上板。[改动位置、结果与集成要求](target-accuracy-status.md)。

2026-11-13 按用户最新安排，先测 **GDN 相关独立边界**：L0–L2 × Prefill／Decode 的 decay_beta、repeat_qk、gdn、recurrent_writeback、linear_attn_norm，共 48 个输出图。已有匹配的 fixture/native 可只补算五项指标；否则按[当前测试命令](target-batch-rerun.md)导出、编译并上板。当前完成本地流程准备，尚无本轮新设备结果。

GDN 历史 Prefill core relative-L2 1.56%～7.12%、raw state 12.10%～44.60%，6 个 raw state dtype 不保持。本轮保留原捕获长度、effective_length、初始状态与真实源码表达式，分别检查 core 和 state。Attention Prefill（99.67%）、MoE／LM head 和完整末态仍待验证。ConvState 12 项小图编译失败不等于整图必然不能编译，旧四层 T64 整图已有成功编译记录。

[交互 HTML](assets/target-architecture-interactive.html)统一显示各模块最新结果：QLinear 38 项、Cache 8 项、RoPE 4 项、其他边界 100 项，共 150 项。RoPE 的旧偏差已替换为零误差，未复测模块保留最近实测值。用户回传清单尚未在本地核验执行文件绑定；独立边界对齐不等于整图精度通过。

已对齐的 RoPE、已通过的 Prefill Cache 和 QLinear 不再复跑。Decode Cache 四项仍失败，K/V relative-L2 99.8086%／99.6897%，历史保留问题暂后处理；不影响先单独推进初始空缓存 Prefill，但仍阻塞后续生成。[Cache 原因分析](target-accuracy-status.md#decode-cache-原因分析)。Draft 保留约 60 ms 的既有测量基线，性能实验暂停；计时口径见[结果](results.md)。
