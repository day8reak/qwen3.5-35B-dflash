# 端侧自主调度器发布检查

日期：2026-10-30。当前结论：**实现与主机检查通过，真实板端完整请求待执行。** 这些检查不证明模型精度、物理断连后的设备完成或性能收益。实机命令见 [current-status.md](current-status.md)。

| 检查 | 去重通过数 | 范围 |
|---|---:|---|
| `tests/test_board_serial_dflash.py` | 8 | 完整表大小／SHA、源身份、S64双输出合同、一次detached请求、原attempt恢复入口 |
| `tests/test_board_serial_transport.py` | 31 | 一次部署、缓存复用、启动响应丢失、只回收不重跑、HDC返回码误报、精确锁owner、失败完成与不确定子进程 |
| `targets/cpro/runtime/tests/test_board_serial.py` | 32 | 实际C++调度器，接受／拒绝／bonus、完整状态链、错误IO、旧receipt变更、SIGHUP后继续、PID发布失败 |
| `tests/test_board_serial_integration.py` | 3 | 实际C++调度器＋现有实际draft CLI，模拟HIAI／target，完整89输入／82输出字节大小与非序号状态映射 |
| `tests/test_serial_dflash_smoke.py` | 11 | 旧主机联调入口回归，不能当端侧自主通过 |
| 合计 | 85 | 74项新端侧路径检查＋11项旧入口回归 |

独立完整集成产生8个生成token，16次target、5次draft调用，包含2个接受候选、1个bonus、4个纠正。target使用重排的80状态输出槽位、无编号含义的文件名和逆序工具日志；检查仍通过。模拟结果只在显式host-fake验证模式下接受，默认实机验收拒绝它。正常路径没有父进程回调或逐token主机数据。

最终native源码SHA256：`d6010e64c8638d770238f4d8028cb7e3023301943c049efcd7fce3083121b09f`。最终变更在fork前记录`STARTING`，fork后先记录当前PID／`RUNNING`再写PID文件，避免写盘失败时误把旧子进程的`EXITED`当成当前进程完成。实际注入PID文件写入失败的检查通过；不确定状态继续保留锁。

CMake Release构建`qwen35_board_serial`通过。这里使用主机编译器；用户DDK主机执行入口时，沿已有`build_hiai.py`和选定OHOS SDK构建ARM64调度器及draft runner。没有重新编译OMC。实际target工具flags和draft原生入口沿用已有成功路径。

本次workspace run：`20261030T095702Z-10259-5ab5b7`。检查产物在该run的`out/`，Python汇总`board-python-release.xml`为50项；native去重采用`native-board-final.xml`、`native-board-last.xml`、`native-board-journal.xml`，独立集成采用`board-native-integration-final.xml`及后续单例复核。早期失败记录保留，不能把旧记录覆盖成通过。

执行限制：完整prefix≤64；P0固定P8/G8、关闭EOS、使用实验控制值；模型按次加载并在板端保留中间张量，尚未实现模型常驻。板端子进程挂住时保留PID并等待，不自动kill／重试。部署阶段远端操作完成状态不明时保留锁，需要检查原调用记录；恢复不会猜测清锁。发布检查没有调用真实设备，也没有停止用户正在运行的任务。

后续助手应以真实设备的`loop.json`、取回校验及断连／恢复记录推进P0状态。`STARTED`、主机测试、历史主机调度循环、单个模型exit0均不能把P0改为通过。
