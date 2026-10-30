# 端侧自主调度器发布检查

日期：2026-10-30。当前结论：**实现与主机检查通过，真实板端完整请求待执行。** 这些检查不证明模型精度、物理断连后的设备完成或性能收益。实机命令见 [current-status.md](current-status.md)。

## 同日运行开销修复

本轮`20261030T105923Z-14268-375427`修改默认板端入口的非模型开销，保持原89输入／82输出和完整80状态提交。用户旧host任务继续运行，未连接设备或改动其任务目录。

- `build_hiai.py`明确配置并构建Release，记录native构建类型。此前脚本未指定构建类型，具体旧SDK最终优化选项未测；不把该修复写成已测加速倍数。
- 复用刚验证的文件记录；同一次稳定读取用于哈希与消费logits／features／draft输出。下一步状态副本直接与已绑定父输出SHA比较，去掉重复内存哈希；80私有副本、89输入执行后核验和最后清单复核保留。
- 完整请求取回的N个文件已做远端批量SHA检查，随后直接接收并按原清单做本地稳定SHA，去掉N次独立重复远端SHA查询。N次文件传输仍存在；不能宣称已变成单次打包下载。
- 启动源哈希及子进程起止有日志，`supervisor_timing`记录单调墙钟毫秒。子进程时间含加载／推理／卸载，不是纯NPU耗时。

本轮109项去重检查通过：native35（原32加3项同字节写入身份变化／错误输出SHA），transport36（原31加5项RPC计数、接收后篡改／截断／文件缺失／读取身份变化），独立实际C++集成3，prepared8，旧host入口11，native构建配置16。报告分别为`native-io-tests.xml`、`board-serial-transport.xml`、`io-integration.xml`、`io-regression.xml`、`build-config.xml`。另以实际主机CMake Release构建重新通过接受／全拒绝／混合三种完整P8/G8分支（`io-release.xml`，与前述去重数重合）；这不替代OHOS编译和实机验证。新增native源码SHA：`4a7304b5bf2fb8dd743d2c7a759926654ee57e88586df417e304f7dc5f9d856b`。

常驻实例尚未接入默认循环。所选DDK外置权重API来自用户贴回的实际构建记录，初始化调用顺序仍需单独实验；具体后续步骤见[常驻执行方案](resident-runtime-plan.md)。以下保留首版发布证据。

## 首版发布记录

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
