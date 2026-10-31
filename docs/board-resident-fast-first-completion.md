# 常驻内存模式：首次实机结果与加载问题

2026-10-31，用户回传 `dflash-board-resident-fast8-recovery`：原 prefill 1 次、decode 8 次、完整 draft 7 次，两个 resident 模型各加载和关闭 1 次；全部完成，锁已释放。候选为 `3042e63`。助手没有连接设备或独立验证远端源码身份；来源是用户粘贴的 compact 摘要，结构化选录见[实机记录](../specs/board-resident-fast-first-completion.json)。

`runtime_profile=compact-memory-v1`、`intermediate_tensor_io=NONE_MEMORY_ONLY`，逐 token 张量诊断关闭。生成 `[4,2,24,2,4,22,0,2]`，接受 0／7。执行里程碑通过，数值等价和算法加速仍未验收。

## 时间拆分

| 项目 | 上次常驻、含诊断 | 本次内存模式 | 本次同步 API |
|---|---:|---:|---:|
| decode 每次平均 | 1149.25 ms | 195.75 ms | Run 111.64 ms |
| draft 每次平均 | 747.57 ms | 740.71 ms | Process 741.11 ms |
| prefill 一次 | 13.833 s | 10.285 s，含加载与关闭 | Run 484.542 ms |
| 启动 payload 哈希／绑定 | 19.899 s | 0.001 s | — |
| draft 一次加载 | 8.380 s | **202.958 s** | 未拆分 |
| 整个请求 | 88.680 s | **230.735 s** | — |

decode 包装时间在两次单次记录中下降约83%，比值5.87；其中本次外层与 Run 的差额约84.11ms，包含 SDK 输入复制、输出初始化、输出读回与必要运行检查。不能把这个差额直接归为纯内存拷贝，或把跨请求差额全部归因于某一条优化；两次生成序列也不同。

Run 111.64ms 的倒数约8.96次／秒。会议所述11token/s对应90.91ms，相差约20.73ms；其模型、量化、上下文、设备与计时范围还未绑定，不能认定同条件差距。Run／Process是同步API墙钟，包含SDK内部工作，不是NPU kernel-only。

draft API平均略大于包装平均不是负开销：包装累计逐次取整的毫秒，API累计微秒；7次相差2.738ms，在毫秒截断范围内。保留原值，不修改测量来制造正差。

## 203秒在哪里

draft 加载占本次总墙钟约88%。剔除这个观测项，其他全部阶段合计27.777秒；这只是账面分解，不是“修好后耗时”承诺。生成阶段两个执行器包装共6.751秒，draft占约77%。本次接受0／7，且现有decode1串行verify仍每个提交token推进一次target，没有证明投机算法减少了target调用。

代码中 `resident_draft_load_ms` 包围整个 `HiaiStatefulExecutor` 构造，包含读取target已有buffer身份、V1 Init／版本查询／设备查询、mmap OMC、SetModelBuffer、Load、描述符查询、tensor分配与初始化。**它不是单独的 Load API 计时**，现有结果不能定位其中哪步消耗203秒。

候选原因只有假设：移除前置全量哈希可能改变2.85GB draft的文件缓存状态，按需映射的读取成本随后落在构造阶段；本次加载pair前MemAvailable约9.43百万KiB，上次约18.05百万KiB，系统内存状态也不同。现有摘要不能证明磁盘冷读、分页、内存泄漏或SDK等待中的任何一种。

## 后续加载诊断（已更新）

下面的首次日志提取已完成，用户补回的加载记录仍不足以拆分203秒。当前下一动作见[加载计时版](board-resident-load-timing.md)。保留当时的只读命令供追溯（无需rg）：

```bash
"${HDC:-/usr/bin/hdc}" -s "$BIP" -t "$SN" shell 'hilog -x -P 12858' |
  grep -iE 'load|model|buffer|memory|alloc|weight|timeout|fail' | tail -n 160
```

attempt 为 `/data/local/tmp/qwen35_board_serial_e5406ec6e21241259be1769fbb2e3faa`。如果日志没有足够阶段时间，下一代码变更应只补一次加载的子阶段墙钟及进程I/O／缺页计数；不恢复逐token哈希或dump，不先更改加载顺序、预热整表、换模型或编S128。计数只支持相关性，不能单独证明底层阻塞原因。

之后继续精度主线：第2个生成token本次2／上次resident0／旧nonresident12。首decode发生在第一次draft Process之前，但draft已加载；相同首token4不证明prefill状态或features相同。需绑定prepared身份并补有界prefill／首decode对照和同runner普通target基线，不能从本次compact补回未保存张量，也不能归因于关闭校验就放过差异。P5完整验收仍待数值等价、重复稳定性及独立权重释放。
