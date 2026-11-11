# 当前结果

更新：2026-11-11。设备数据来自用户回传；本地主机未重跑服务器编译或设备任务。流程接通、精度通过和性能收益分别记录。

## Target 精度

范围为 L0–L3 的 Prefill T256／Decode T1 独立边界小图，共 158 个输出 case。

| 项目 | 当前结果 |
|---|---|
| OMC 编译 | 累计 146 成功；12 个 ConvState 小图失败 |
| 数值回传 | 142 项：2 PASSED、2 FAILED、138 METRICS_RECORDED；另 4 项待回传 |
| relative-L2 >1% | 51 项：量化投影 38、RoPE 4、Prefill GDN 6、Prefill Attention 1、Decode Cache 2 |
| 新增 Norm／Conv1d | 22 项已回传，relative-L2 分别为 0.039795%～0.189404%、0.021476%～0.027210%，dtype 保持、重复稳定 |
| 覆盖限制 | 尚未证明整图、MoE 内部、LM head 或连续状态反馈精度通过 |

四个自定义算子重点为 QuantBatchMatmulV4444 所在链路、CacheUpdate、ChunkGatedDeltaRule、AdnFusedInferAttention；逐项证据见[Target 精度表](target-accuracy-status.md)。大误差是定位线索，尚未全部定责到内核。

## Draft 保留基线

2026-11-06 的完整 packed Draft，38 次诊断调用的加权平均**准备＋同步 API 为 60.4067 ms**；离线 W.T 候选为 73.6574 ms，未保留该优化。包含 CPU pack、SetInput／ReuseInput 和同步 API，排除加载、文件 I/O、KV 回读、快照与提交；不是预热稳态或端到端基准。

| 模式／阶段 | 调用数 | packed 准备＋API ms |
|---|---:|---:|
| isolated catchup | 11 | 60.8652 |
| isolated proposal | 8 | 61.3763 |
| feedback catchup | 11 | 59.8145 |
| feedback proposal | 8 | 59.6208 |

原 CPU 数值门槛仍 FAILED：每种模式 Top1 为 **83/86 匹配**，各 8 例 hidden 失败，isolated／feedback 分别 13／18 例 KV 失败，未写区失败 0。与离线 W.T 对照的 488 个区域中有 1 个 hidden 区域不等价，因此 baseline_equivalence 仍 FAILED。

另一轮 2026-11-05 的 packed 改动与原 RoPE 模型 488 区域逐字节等价，准备＋API 由 66.4573 降为 60.7651 ms，观测减少 8.565%；它与上表不是同一次模型／测量，不混算。后续 packed profiling 的 load／inference／unload 为 162.214／60.214／9.318 ms，主要热点为词表 head 方向 MatMul 19.960 ms、特征投影方向 Conv 8.691 ms；算子时长不作为可相加的关键路径。

历史测试目录曾获准清理，保留数值记录不代表原重放输入仍在；复测先确认实际模型和输入可用。改动与收益概览见[优化策略](optimization-strategy.md)。

## 已有端侧闭环

旧接线模型已完成文字编码 → Prefill → 常驻 Draft／Decode → 文字解码。以下是同次旧闭环，不能用新的 60 ms 独立 Draft 测量替换其中一项后宣称端到端收益。

| 项目 | 观察值 |
|---|---:|
| Prefill 同步 Run，1 次 | 450.087 ms |
| Decode 同步 Run，8 次均值 | 109.617 ms |
| Draft 同步 Process，8 次均值（1 补齐＋7 proposal） | 608.155 ms |
| Prefill 加载 | 10.300 s |
| 完整请求墙钟 | 32.569 s |
| 接受／实际考虑候选 | **0/7，0%** |
| 纠正／Bonus | 7／0 |
| 输出 | 8 个 token，文本为 `=3'!=%3%` |

同步 API、加载和请求墙钟范围不同，不能直接相加。完整语言质量与普通 Decode 对 DFlash 的配对加速尚未通过。

当前执行入口见[完整模型流程](run-end-to-end.md)和[Target 批量验证](target-batch-rerun.md)。逐次实验／失败记录保留在 [Git 历史](https://github.com/day8reak/qwen3.5-35B-dflash/blob/459060e/docs/results.md)。
