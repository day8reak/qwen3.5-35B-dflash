# 混合接口常驻：首次板端完整闭环

> 历史记录：保留当时的观察、失败和候选命令。当前操作请从[文档首页](../README.md)或[当前状态](../current-status.md)进入；不要直接重复本页旧请求。


2026-10-31用户回传 `dflash-board-resident-mixed8-recovery` 成功。**原prefill一次、V2 decode与V1完整六层draft各加载一次、板端多轮串行验证、最终回收均已执行通过。** 这是单次请求的功能与常驻执行里程碑；P5的数值等价、重复稳定性及内存释放验收继续保留。

原始来源是用户粘贴的compact摘要，助手没有连接设备或独立重取文件。候选代码为 `060ea26`，未从本次远端artifact重新验证执行源码SHA。结构化观察见[实机记录](../../specs/board-resident-mixed-first-completion.json)。

## 执行结果

- attempt：`/data/local/tmp/qwen35_board_serial_1a8537eaa0954c1a87ecb6a2e0c7e474`，进程5008。
- prefill执行1次，decode执行8次，draft执行7次；decode/draft各加载1次、关闭1次。
- token：`[4,0,28,18,2,2,2,0]`；接受0／7、纠正7、bonus0。
- prefill与decode的Init、InitWeights成功，Run完成，DeInit正常返回；draft的Load、Process、Unload成功。未调用单独DeinitWeights，清理错误为空，锁已释放。
- 原三OMC不重编，物理S64／逻辑8的prompt控制、15候选最多考虑前2个仍保持原实验范围。服务端没有逐token计算或调度。

## 时延口径

| 项目 | 次数 | 合计 | 每次平均／说明 |
|---|---:|---:|---|
| prefill | 1 | 13.833秒 | 含加载11.435秒、运行包装1.931秒、关闭0.462秒及少量其他开销 |
| resident decode | 8 | 9.194秒 | 1.149秒；含tensor复制、读回、哈希／buffer检查 |
| resident draft | 7 | 5.233秒 | 0.748秒；同样不是纯NPU时间 |
| decode/draft一次性加载 | 各1 | 17.265秒 | 8.885秒＋8.380秒 |
| pair关闭 | 1 | 0.420秒 | 结束时关闭 |
| 启动源文件哈希 | 1 | 19.899秒 | 单次启动检查 |
| 整个supervisor | 1 | **88.680秒** | 尚有22.836秒未由上列大项细分 |

旧逐次加载版本总墙钟305.788秒，本次88.680秒，单次记录比值3.448、用时下降约71%。旧decode/draft阶段含每次加载和卸载，新阶段排除了常驻的一次性加载／关闭，且生成token不同。这是运行框架改进的观察，不能视为kernel加速、与普通decode配对的DFlash算法收益或最终2倍性能验收。两次接受率都为0／7，未观察到接受候选减少target调用的收益。

## 下一步先定位首个差异

旧token `[4,12,10,8,0,4,32,2]` 与新token的首项同为4，第2项即分叉。`board_serial_main.cpp` 的Execute先Prefill、StartResident，再Commit初始anchor，之后才进入Draft循环；Commit先执行Target(anchor4)，其logits决定下一token。故首次分叉的值在第一次Draft Process之前已算出。draft此时已经加载，仍不能排除双manager共存对target执行的影响。

先核对两个已保存prepared请求的模型、prompt、表、控制和80状态映射身份；首token相同不能证明prefill状态／features相同。随后比较prefill输出与首decode的完整89输入和82输出，遇到首个差异即停止沿分叉后的history比较。输入不同时先追状态／embedding／控制来源；输入相同而logits不同时检查target adapter、buffer初始化、复用和重复性。

两次compact回收都不提供完整张量重放证据。下一工程动作是实现有界首步捕获及同runner的普通target基线，再由用户做最小实机对照；目前不提供不存在的target-only CLI。保持原OMC、权重和精度策略。必要的后续隔离轴是target-only与pair已加载但尚未调用draft Process，避免把首decode差异先归因于draft提议质量。

## 内存与完成范围

本次报告的VmHWM随采样阶段下降且各次等于VmRSS，原样保留为待核对的遥测值；不能从它们宣称真实累计峰值或专用NPU显存。MemAvailable也受进程外资源和回收延迟影响。正常DeInit返回及下一模型加载成功支持本次执行可行，不能单独证明全部外置权重已立即释放、重复请求无泄漏。

P0端侧自主完整执行与P5单次混合常驻执行均已完成。精度、非零接受分支、普通基线一致性、重复稳定性、长上下文、批量verify及配对性能仍按[总计划](../dflash-delivery-plan.md)继续；本次不要求重复已成功的完整请求。
