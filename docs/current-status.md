# 当前阶段：Target Cache 精度定位

更新：2026-11-12。最新 **8 个 Cache 候选均有数值**：Prefill 的 K/V × 浮点/整数 **4/4 PASSED**；Decode **4/4 FAILED**，K relative-L2 **99.8086%**、V **99.6897%**。两种索引在回传精度下相同；整数图此前的导出／编译阻塞已被本次运行越过，索引改写没有修复 Decode。

补充的 Decode V 明细中，**130,966 个不一致元素全部在 golden 相对输入未变化的区域，golden 变化区不一致数为 0**；最大误差处实际为 0、golden 为 3.494140625。已定位到该片段的历史保留失败，优先核对 CacheUpdate functional 输出与原位写入契约、历史复制和输出绑定。片段未带 case ID／输出哈希，不能直接推广为四个 Decode 输出逐字节相同。见[原因分析与下一步](target-accuracy-status.md#decode-cache-原因分析)。

按用户明确的新验收口径，量化 **out_proj 的 relative-L2 <2% 即通过**：已有 8 项全部满足，含 L0 Prefill／Decode 的 1.3766%／1.7625%。原始 CSV 的 METRICS_RECORDED 保留，验收结论单列；其他模块不套用 2%，Cache 仍要求整缓存逐位一致。后续仅复测 4 个失败的 Decode Cache 图（K/V × 浮点/整数）；已通过的 Prefill Cache 和 QLinear 不再导出、编译或上板。

[交互 HTML](assets/target-architecture-interactive.html)统一显示各模块最新结果：QLinear 38 项、Cache 8 项、其他边界 104 项，共 150 项；新结果直接替换对应旧值，未复测模块保留最近一次实测值。下一步先[读取已有比较报告](target-batch-rerun.md#先读已有报告无需重编或上板)，无需重复导出全部模型。用户回传清单尚未在本地核验执行文件绑定；完整状态反馈仍未验证。

ConvState 12 项小图编译失败仍待处理。GDN、RoPE、FusedAttention 等旧指标仅作历史记录，MoE、LM head 与完整模型待覆盖。Draft 保留约 60 ms 的既有测量基线，性能实验暂停；计时口径见[结果](results.md)。
