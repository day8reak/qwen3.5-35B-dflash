# 当前阶段：Target 全部分模块验证

更新：2026-11-11。已复查小图导出器全部 16 类：修正普通 Norm 的 NZ FP16／CPro FP32 权重差异，量化权重由原替换工具重建核对，其余片段绑定原 CPro forward／helper。代码已做本地源码与 CPU 检查；新 AIR／OMC／上板数值仍需在权重服务器执行。

## 用户现在执行什么

使用[完整批量命令](target-batch-rerun.md)，本轮从 AIR 重新开始；新增 `--target-config "$TARGET_CONFIG"` 绑定原 HF／W4 权重。随后 OMC → 上板 → 回收 → 精度 CSV，ConvState 失败不阻塞成功项。

| 已收到的结果 | 数量 |
|---|---:|
| 边界输出总数 | 158 |
| 累计编译成功／失败 | 146／12 |
| 已回传数值／已编译待回传 | 142／4 |
| relative-L2 >1% 的输出 | 51 |

这些是此前回传，新全批结果尚待执行；编译成功和输出稳定都不等于精度通过。

优先单独定位 **QuantBatchMatmulV4444 所在量化链路、CacheUpdate、ChunkGatedDeltaRule、AdnFusedInferAttention**，另查 RoPE 组合子图。异常与阶段有关，不能据此断定每次调用都失败或已确认内核 bug。范围和指标见[高危表](target-accuracy-status.md#高危算子重点定位表)及[交互结构图](assets/target-architecture-interactive.html)。

剩余 12 个 ConvState 小图在 format config 阶段失败；成功 T64 整图存在专用 builder／executor，构建上下文差异待定位，见[编译对照](target-compile-triage.md)。MoE 内部、LM head 和连续状态反馈仍待覆盖。

Draft 保留约 60 ms 的既有测量基线，性能实验暂停；它与旧端侧闭环的运行模型／计时口径不同，见[结果](results.md)。历史单算子调用和逐次排障查阅 [Git 历史](https://github.com/day8reak/qwen3.5-35B-dflash/blob/459060e/docs/current-status.md)，不作为本轮待执行步骤。
