# 当前阶段：Target 精度定位

更新：2026-11-12。QLinear／Cache 候选共 46 图，上一轮 OMC **42 成功／4 失败**，42 项数值已回传。scale 适配后 **36/38 QLinear relative-L2 <1%**；仅 L0 out_proj 的 Prefill／Decode 为 **1.3766%／1.7625%**。Cache 浮点控制组 Prefill K/V 零误差，Decode 仍失败（99.81%／99.69%）。

Cache 整数四图没有数值。乘减求余后的最新 AIR 为 **42 EXPORTED／4 FAILED**，四个整数图均被检查器拦下。回传图显示 `Mul(INT32, INT32)` 的输出描述是 FP32，随后 Cast 回 INT32；候选导出器已加入仅对该乘法生效的 GE 元数据适配。保留类型检查，按用户确认仅保留 Cache 与明确 FAILED 项，重跑[8 个 Cache 图](target-batch-rerun.md)；新 AIR／OMG／数值均待实际环境验证。

生产 CPro 合入要沿实际 QLinear 工厂和 GE converter 应用适配：保留源码的 `utils.py` 导入的是 NZ QLinear，只改 CPro 同名类可能不生效。前端 per-token 仍需 FP32，GE 边界才转 FP16。Cache 另查更新区／历史保留区与实际被写对象，不能把索引编译修复当成精度修复。见[源码落点与定位表](target-accuracy-status.md)。

[交互结构图](assets/target-architecture-interactive.html)可切换候选 42 项和修改前 146 项。GDN、RoPE、FusedAttention 的数值仍来自旧基线；ConvState 12 项无数值，MoE、LM head 和连续状态反馈待覆盖。用户回传的编译与数值清单尚未在本地核验执行文件绑定。

Draft 保留约 60 ms 的既有测量基线，性能实验暂停；计时口径见[结果](results.md)。历史过程查阅 Git 历史。
