# 板端统一 HIAI V2 常驻执行

更新：2026-10-30。原 prefill 完整板端闭环已经取得用户实机完成结果；见 [成功基线](board-prefill-first-completion.md)。本次新增 `--resident`，将 decode 和当前内嵌权重 draft 都接到 HIAI V2。**首次设备请求在 target 的 InitWeights 返回 INVALID_PARAM，draft 尚未加载，常驻未通过。** 正在核对原成功工具的加载实现，详见[失败记录与诊断修复](board-resident-validation.md)。默认入口保留已成功的按次加载方式，旧请求仍按自己的准备记录恢复。

## 执行方式

| 阶段 | 已成功的默认模式 | 新 `--resident` 模式 |
|---|---|---|
| prefill | 一次原工具 V2 子进程，退出后交接状态 | 保持相同方式，退出后才加载常驻模型 |
| decode | 每 token 启动工具并加载／卸载 | 一个 V2 manager，加载一次，多次同步 `Run` |
| draft | 每轮独立 CLI，embedded `legacy-buffer` | 同一进程内另一个 V2 manager，embedded `file-v2`，加载一次、多次 `Run` |
| 主机参与 | 一次部署、启动、最终回收 | 相同；中途无主机查表、状态传输或接受判断 |
| 原模型 | 三个既有 OMC／原权重 | 不变；仅编译 native runner，无 AIR／OMC 重编译 |

新二进制为 `qwen35_board_resident`。一个 native session 内保留两个 manager；prefill 完成后先加载原 decode，再加载 draft，然后交替调用。两者使用不同的 built-model 名称。没有两个 worker、板端 IPC 或每轮新进程，也不会在 V2 失败时静默回退 legacy API。

内嵌／外置权重是存储方式，V2／legacy 是运行接口。已经成功的内嵌 draft 原本使用 legacy-buffer；这不能代替新内嵌 draft 的 V2 实机验证。历史外置 draft 的 Init 失败也不能直接判定当前内嵌版本不支持 V2。

## 生命周期与输入输出

原 decode 复用 `HiaiOriginalTargetExecutor`。准备阶段从已绑定的原生 `model-io.json` 导出独立 `resident-abi.tsv`，绑定实际 89 输入／82 输出的名字、维度、类型、raw dtype／format 和字节大小。不能把旧 draft 的 ABI 或原图 shape 推断套给 target。

当前外置 target 实验顺序为 `RestoreFromFile → SetName → CreateModelManager → InitWeights → Init`，退出时 `DeInit → DeinitWeights`。用户提供的头文件证明这些接口存在，未证明其顺序满足所选私有实现；每阶段失败记录并停止，不自动换顺序重试。

内嵌 draft 复用 `HiaiStatefulExecutor` 的 `file-v2` 路径：`RestoreFromFile → SetName → Init → Run`。保留完整六层原 S64 的 3 输入／2 输出 ABI；每次输入完整 prefix 的 features、IDs 和 mask，每次输出 15 个候选与 hidden。每轮最多考虑的候选数仍由原参数控制，默认 K=2。

`ResidentSession` 一次加载并分配两模型的 176 个输入／输出缓冲区；多轮期间复用。输入全部检查后复制，检查实际读回内容，检查输出对象／地址稳定；返回的是独立输出副本。任一失败使 session 停止复用并清理已创建的实例。会检查两模型缓冲区区间不重叠，不能依赖全零内容来判断是否共享内存。

80 状态仍按逐一绑定的输出→输入关系交接，只推进已提交 token。首次版本保留板端私有文件及字节副本，compact 模式清理已消费的成功 scratch，dump 模式保留详细证据。**常驻模型不等于状态零拷贝**；未来再单独优化状态 bank／内存交接，避免把生命周期改动与状态所有权改动混在一次实验里。

## 结果和计时

常驻结果独立使用 loop schema 3、逐调用 receipt schema 2，并记录 session ID、同一进程 PID 和连续 call index。单次 `Run` 返回不冒充子进程退出；只有 prefill 有实际子进程记录，整个请求最终仍核对 supervisor 完成及锁归属。

普通输出保留 token、接受率和 prefill／decode／draft 时延，并补充：

- `execution_mode=resident-v2`。
- `resident_lifecycle`：每模型加载次数、执行次数、关闭状态与加载／Run／关闭墙钟。
- `resident_memory`：加载前、两模型加载后、关闭后进程 VmRSS／VmHWM 与系统 MemAvailable；不可读取时标记缺失。它不等于专用 NPU 内存统计。

prefill 仍是一次子进程墙钟，含加载／执行／卸载。decode／draft 是 session 调用墙钟，排除一次性模型加载／关闭，包含输入复制、读回校验、同步 Run 和输出复制；不是纯 NPU 推理时间。总墙钟另含启动哈希、调度和私有文件 I/O。不能直接把旧子进程均值与新 Run 均值相除宣布算法加速。

完整回收验证逐状态、候选、提交和同一 session 调用顺序；compact 仅回收最终 token 与小型完成摘要，不宣称主机重放了 80 状态链。恢复必须使用原 attempt，不能切换 resident 或 dump 模式后重跑。

## 实机运行与验收

完整启动／恢复命令见 [当前执行入口](current-status.md)。复用已成功的 prefill 请求参数，添加 `--resident` 并使用新 output。该模式要求 `--prefill-feedback`；不把旧 serial prompt 路线作为替代。首次目标为相同 P8/G8、一次 prefill、8 次 decode、至少两轮 draft，以及 decode／draft 各加载一次、正常关闭。

对照原默认模式的相同输入、控制和 OMC，检查 token 是否仍为 `[4,12,10,8,0,4,32,2]`；若不同，先保留新结果并定位 API／状态差异，不能因旧版本本身未达数值精度而忽略运行方式带来的差异。原结果接受 0／7；不能声称已经覆盖实机接受／bonus 分支。

所选 V2 同步 `Run` 接口没有 timeout 参数；当前 CLI 等待时间不是可取消的每次设备 Run 截止时间。未知执行状态时保留进程／锁并恢复查询，不自动重跑。

失败时保留确切阶段、session／PID、attempt 和锁记录；根据完成证据恢复同一请求。不要删 output、删锁、改变 Init 顺序或改用旧 API 后将结果归类为常驻成功。先验证实际 DDK 编译和两个模型初始化，再核对多轮与内存；不根据约 18.8GB 外置 target 权重和约 2.85GB draft 文件大小推断实际峰值 RAM。

prefill 卸载和 decode 加载暂不重叠。它只影响一次转换；先取消每 token／每轮重复加载。只有确认输出独立、runtime 并发生命周期支持、实际峰值内存余量后，再作为独立优化实验。

P5 实现／设备联调预算仍为 0.5–2 个有效工作日，私有 SDK 或内存问题可能增加时间；完整数值与性能验收依赖 P1/P2。主机测试、代码身份和设备待验项目记录在 [常驻验证记录](board-resident-validation.md)。
