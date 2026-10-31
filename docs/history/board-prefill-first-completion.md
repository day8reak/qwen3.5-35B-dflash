# 原prefill驱动的端侧DFlash首次闭环

> 历史记录：保留当时的观察、失败和候选命令。当前操作请从[文档首页](../README.md)或[当前状态](../current-status.md)进入；不要直接重复本页旧请求。


2026-10-30，用户回传 `dflash-board-prefill8-recovery` 成功结果。**P0执行阶段完成**：一次原prefill、8次decode、7次完整draft在同一板端请求中完成，生成8个token，完成后回收小型结果并释放本请求锁。助手未连接板子，证据来源为用户终端摘要，原文件仍由用户保留。

- execution：`$AI_RUN_DIR/dflash-board-prefill8-run`
- recovery：`$AI_RUN_DIR/dflash-board-prefill8-recovery`
- attempt：`/data/local/tmp/qwen35_board_serial_07858071594b4747bc36207cf8038278`
- tokens：`[4,12,10,8,0,4,32,2]`
- accepted／considered：0／7；corrections7，bonuses0。
- 状态：`BOARD_SERIAL_DFLASH_EXECUTED_UNVALIDATED`；锁为`NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT`。

| 阶段 | 次数 | 总墙钟 | 平均墙钟 |
|---|---:|---:|---:|
| prefill | 1 | 15.802秒 | 15.802秒 |
| decode | 8 | 116.692秒 | 14.5865秒 |
| draft | 7 | 137.128秒 | 19.5897秒 |
| 请求整体 | 1 | 305.788秒 | — |

子进程时延包含加载、执行和卸载。请求总墙钟还包括19.754秒启动源哈希以及调度／文件交接，不含服务器部署和结果回收。`prompt_ingest`只是prefill的别名，不能重复相加。该结果没有纯NPU时间或与普通decode的配对速度结论。

使用原prefill64的显式短有效长度实验，逻辑P8、物理64，其后使用完整真实历史；无模型重编、无S128、无逐token服务器决策。当前成功API是target prefill/decode的HIAI V2与内嵌draft的legacy-buffer。模型仍按次加载。

本次为compact模式，仅回收最终tokens、计数／时延和进程完成记录；没有在主机重放完整80状态链。精度、私有算子有效长度语义、真实接受／bonus分支和物理断连实测仍未验收。P0通过不表示这些后续阶段通过。

保留本次成功目录作为常驻改造的回滚基线。用户已要求下一步统一V2、让decode和draft各加载一次后重复运行；常驻方案见[resident-runtime-plan.md](resident-runtime-plan.md)。机器可读用户摘要见[board-prefill-first-completion.json](../../specs/board-prefill-first-completion.json)。
