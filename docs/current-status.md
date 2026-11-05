# 当前阶段：Draft 精度定位与性能方案

2026-11-05：`untouched` 回放已消除本轮首例额外12928个零，14路输出在两模式间逐位一致。离线head检查的reference／device hidden分别复现7/7个golden／设备Top1，hidden差异足以在CPU head下解释翻转；首个偏差层尚未定位，整图仍未通过原数值门槛。详细结果见[结果页](results.md)。

## 性能优先级

当前同步API约66ms；绑定profiling的一次inference为66.556ms，要到50ms需省16.556ms。下表的目标只是用于筛选实验的预算，**不是已测收益或硬件下界**；尚缺匹配X90的能力标定，编译节点与源码区域的shape映射也需核对。

| 方向 | 已测相关开销 | 准备验证的目标 |
|---|---:|---|
| 特征投影＋布局 | Conv2D 8.753ms＋Transpose_1 5.284ms | 直接读取ND特征做完整K=16384点积，争取区域降至6–8ms |
| LM head分块＋Top1 | MatMul_30 19.965ms，Top1 0.415ms | 先验证矩阵乘本体能否降至12–15ms，再合并分块归约 |
| KV更新 | 十二个Gather合计2.857ms，另有Select等 | 直接写新增区间、保留完整旧bank，争取省1–2ms |
| 无新增context的proposal | 当前仍计算固定16行context | 仅在运行时valid_rows=0时跳过无消费者的投影和context KV分支 |

head融合必须保留全词表、15行之间的权重复用、FP16 logits舍入与并列取最小索引；公开的hidden输出仍需保留。约1.017GB的head权重远大于7.45MB logits，仅省logits往返不能推断20ms矩阵乘大幅加速。融合减少中间搬运的机制可参照[华为Ascend C指南](https://www.hiascend.com/doc_center/source/zh/canncommercial/800/developmentguide/opdevg/Ascendcopdevg/CANN%E5%95%86%E7%94%A8%E7%89%88%208.0.0%20Ascend%20C%E8%87%AA%E5%AE%9A%E4%B9%89%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E6%8C%87%E5%8D%97%2001.pdf)，具体接口和收益以所选X90 DDK验证为准。

假设其他部分不变、区域收益能落到API墙钟：入口8ms、head15ms、KV省1ms时约54.6ms；入口7ms、head12ms、KV省2ms时约49.6ms。这是有条件的串行预算，算子duration之和不等于关键路径。50ms有可测试路径，尚不能承诺达标；全context场景不能沿用valid_rows=0的收益。

纯catchup可另查只生成context KV的入口，但需证明调用方只消费这些输出；其收益不能充当proposal≤50ms。当前不优先重写StableArgMax（0.415ms）或仅softmax（六层合计约0.87ms）。

## 现在需要执行什么

本轮回放、head审计和profiling均已回收，无需重复上板、编译或采集。保留当前失败夹具、原始CSV和head-audit.json；下一步优先核对入口投影／布局映射和KV更新边界，再实施一个有独立正确性检查的小范围候选。新算子、模型图和运行器尚未改动，验收容差保持。

首次构建见[完整流程](run-end-to-end.md)；离线审计入口仍为 `validate_incremental_draft.py --audit-head`，本轮已完成。
