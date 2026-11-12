# 当前阶段：Target 全部分模块验证

更新：2026-11-12。用户回传重新导出后的 `all-boundaries/omc`：**90/158 编译成功，68 项失败**，Prefill／Decode 各 45 成功、34 失败。尚未收到本轮上板数值；本地源码／CPU 检查不代表 OMG 编译通过。

## 用户现在执行什么

已有本轮 AIR，无需重新导出。GDN 日志已确认 TE worker 在 `te_update_version()` 更新平台参数时退出；此前全批命令漏传已有的 `--tbe-diagnosis`，现已补回。按[编译排障](target-compile-triage.md)直接全批重新编译，目标保持 kirinx90；90 个成功项可独立继续[批量上板与比较](target-batch-rerun.md#4-批量上板)。编译脚本末尾的提示不会自动启动设备。

| 批次 | 编译成功／失败 | 已回传数值 |
|---|---:|---:|
| 本轮源码对齐后 | 90／68 | 0 |
| 旧导出（历史） | 146／12 | 142，其中 51 项 L2 >1% |

本轮失败为 QLinear 38、GDN 12、CacheUpdate 4、FusedAttention 2、ConvState 12。前四类共 56 项此前能编译；当前只收到一例 GDN 的实际失败栈，其余项是否同因待验证。修复命令后的服务器编译尚未回传。不能把旧数值挂到新 OMC；编译成功和输出稳定都不等于精度通过。

优先单独定位 **QuantBatchMatmulV4444 所在量化链路、CacheUpdate、ChunkGatedDeltaRule、AdnFusedInferAttention**，另查 RoPE 组合子图。异常与阶段有关，不能据此断定每次调用都失败或已确认内核 bug。范围和指标见[高危表](target-accuracy-status.md#高危算子重点定位表)及[交互结构图](assets/target-architecture-interactive.html)。

旧批 12 个 ConvState 小图在 format config 阶段失败；成功 T64 整图存在专用 builder／executor，构建上下文差异待定位，见[编译对照](target-compile-triage.md)。本轮是否同一错误仍需日志。MoE 内部、LM head 和连续状态反馈仍待覆盖。

Draft 保留约 60 ms 的既有测量基线，性能实验暂停；它与旧端侧闭环的运行模型／计时口径不同，见[结果](results.md)。历史单算子调用和逐次排障查阅 [Git 历史](https://github.com/day8reak/qwen3.5-35B-dflash/blob/459060e/docs/current-status.md)，不作为本轮待执行步骤。
