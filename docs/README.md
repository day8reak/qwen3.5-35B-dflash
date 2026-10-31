# Qwen3.5-35B-A3B · 端侧 DFlash

当前主线复用原40层target（prefill64、decode1）和完整六层draft S64，在板端自主完成prefill、草稿提议、逐token验证和状态反馈。decode与draft常驻；默认关闭中间张量dump，最后一次回收结果。

最新token输入请求已完成：prefill1／decode8／draft7，开启顺序预读后总墙钟31.455秒、draft加载3.883秒。接受率0/7，数值正确性与稳定加速尚未验收。文字接口已通过主机验证，下一次将tokenizer与预读放进同一板端请求。

## 现在运行：只按这三步

1. **[离线准备 tokenizer](tokenizer-offline.md)**：服务器不需要联网或证书；使用锁定的三个本地文件。
2. **[文字＋预读完整命令](board-text-input-output.md)**：设置问题，启动同一个自主板端进程，再回收文字和时延。
3. **[当前状态与下一动作](current-status.md)**：查看最新证据、当前限制及后续精度任务。

已有匹配的模型、权重和板端缓存时，沿用当前产物；上面的入口会构建当前runner。新请求使用新的输出目录，恢复继续指向原执行目录。

## 查阅导航

| 要做什么 | 入口 |
|---|---|
| 交接、阶段顺序、估计天数 | [执行总计划](dflash-delivery-plan.md) · [机器可读状态](../specs/dflash-execution-state.json) |
| 查看预读为什么有效、计时口径 | [加载与预读记录](board-resident-load-timing.md) |
| 配环境、开发或导出其他模型路径 | [开发指南](guides/README.md) |
| 查模型结构、接口和代码职责 | [技术参考](reference/README.md) |
| 查以前的失败、探针和首次跑通记录 | [历史索引](history/README.md) |

`guides/`中的旧ABI双图／新状态图是独立研发路径。当前实机操作以本文三步及当前状态为准，历史记录保留当时结论。
