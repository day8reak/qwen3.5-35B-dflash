# 当前阶段：Target 精度定位

更新：2026-11-12。最新回传 **146 项指标，51 项 relative-L2 >1%**；Prefill／Decode 各 73 项。补齐两个 Norm 和两个 Conv1d 后，高危结论不变。ConvState 12 项仍无数值，MoE 内部、LM head 和连续状态反馈待覆盖。

优先检查 **QuantBatchMatmulV4444 所在 QLinear 链路**的输入、权重、两路 scale 的 dtype／排布／转换。新增只读入口自动检查全部量化投影已有 AIR／OMC／编译参数；命令与判读见[精度状态](target-accuracy-status.md#qlinear-输入与权重检查)。先核对数据契约，不预先修改 Cast 或 packed 权重。

其他重点：Decode CacheUpdate、Prefill ChunkGatedDeltaRule、Prefill AdnFusedInferAttention，以及两阶段 RoPE。问题与阶段有关，不能据此断定每次调用都失败或已确认内核 bug。见[重点定位表](target-accuracy-status.md#高危算子重点定位表)和[交互结构图](assets/target-architecture-interactive.html)。

本次回传表未附 OMC／执行清单，尚未核验其产物绑定。此前 90 成功／68 失败是重编前记录；全批命令已补回 `--tbe-diagnosis`，保持 kirinx90，见[批量流程](target-batch-rerun.md)。不得把旧报告的数值直接认作另一批 OMC 的验证结果。

Draft 保留约 60 ms 的既有测量基线，性能实验暂停；计时口径见[结果](results.md)。历史过程查阅 Git 历史。
