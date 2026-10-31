# 当前结果

最新实机记录：`dflash-board-resident-preread8-recovery`，PID21363。原token输入prompt=8、生成8，混合接口常驻、dump关闭、Draft预读开启。以下来自用户回传；文字输入组合尚未回传结果。

## 推理时延

| 阶段 | 调用数 | 同步SDK API平均时延 |
|---|---:|---:|
| Prefill / Run | 1 | 483.990 ms |
| Decode / Run | 8 | 110.890 ms |
| Draft / Process | 7 | 733.126 ms |

这是同步API墙钟，不是纯NPU内核时间，不含模型加载。Decode外层调用均值198.625ms，包含张量复制等开销；API均值折算约9.02次decode调用/秒，不代表整个DFlash的输出吞吐。

## 加载与完整请求

| 项目 | 耗时 |
|---|---:|
| Prefill加载 | 9.174 s |
| Prefill加载＋运行＋关闭 | 10.164 s |
| Decode加载（一次） | 9.692 s |
| Draft加载（一次，含预读） | 3.883 s |
| 其中：Draft顺序预读 | 1.621 s |
| 其中：Draft SDK Load | 2.251 s |
| Decode外层运行合计 | 1.589 s |
| Draft外层运行合计 | 5.129 s |
| 常驻模型关闭 | 0.398 s |
| **完整请求墙钟** | **31.455 s** |

表内含嵌套计时项，不应全部相加。上轮无预读时Draft加载206.640s、请求234.506s；本次改善主要来自启动阶段，尚非配对的DFlash算法加速结论。

## 接受率与完成范围

| 项目 | 当前结果 |
|---|---|
| 接受 / 考虑候选 | 0 / 7，接受率0% |
| 纠正 / Bonus | 7 / 0 |
| 生成Token | `[4,0,4,0,2,28,8,0]` |
| 原Prefill＋常驻Decode/Draft闭环 | 已实机执行 |
| 板端文字编码／解码＋预读组合 | 主机测试通过，实机待运行 |
| 数值正确性、语言质量、重复稳定性 | 未验收 |
| 相对普通Decode的配对加速 | 未测得 |

原始记录保留在`specs/board-resident-preread-observation.json`。后续先定位Target输出与状态差异，再验证Draft对齐和批量verify收益。
