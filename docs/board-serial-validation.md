# 端侧自主调度器发布检查

日期：2026-10-30。用户已回传旧完整dump入口的首次实机完成结果：16次target、7次draft，完整回收成功、锁已释放，见[首次实机结果](board-first-completion.md)。下列主机记录和新增代码不能自动继承该实机通过结果；模型精度、专门物理断连实验和性能收益仍未验收。命令见 [current-status.md](current-status.md)。

## 原prefill接入：主机检查通过，设备待运行

工作区run `20261030T115722Z-20143-364305`接入`--prefill-feedback`。既有prefill64、decode1和完整S64 draft均复用原OMC；P8填充到物理64，首个anchor在绝对位置8进入decode，然后使用完整9行历史运行draft。合同及未验证控制语义见[board-prefill-integration.md](board-prefill-integration.md)。

本版126项主机检查通过：准备与CLI26项（`prefill-binding-tests.xml`）、native新增14项（`native-prefill.xml`）与旧路径回归12项（`native-prefill-legacy.xml`）、传输72项（`prefill-transport-final.xml`）、实际C++调度器＋实际draft CLI两模式集成2项（`prefill-integration-final.xml`）。此前已发布的gear字段兼容修复另有36项通过（`prefill-feedback-tests.xml`）。主机绑定、native、transport和发布命令均完成独立审查。

两种模式使用模拟HIAI／target，均完成1次prefill、8次decode、5次draft并生成tokens9至16；完整模式还拒绝重新计算SHA后篡改的logits选行、跨owner映射、有效feature前缀和首次decode父状态四种反例。验证覆盖独立prefill缓存和精确锁所有权、失败保留、同请求恢复、compact时延别名不重复计数。早期失败XML保留，最终通过以带`final`的上述两份运输／集成报告为准。

native源码SHA256：`bc15237844e0bc24d5a3a0fb616b3c9cf302bf58dc9928a0e7f77820de38a8d2`；transport源码SHA256：`9ba330b5ae97dbe55a301d5e12b8e567d6846c490c42563dae1f45044a2d0968`。其余源码hash见[执行状态](../specs/dflash-execution-state.json)的`prefill_board_integration`。

没有调用设备；OHOS编译及真实prefill多轮执行由用户运行入口验证。P0仍为进行中。完整dump可以在主机复核输入、80状态与特征链，compact只校验小型完成结果；均不证明私有kernel控制语义、精度或加速。以下为既有serial模式的历史验证，不能替代本版本实机结果。

## 默认精简结果与可选dump

工作区run `20261030T113250Z-17949-689ad3`增加`--dump`，新请求默认关闭，旧prepared未记录dump时维持原完整模式。普通回收只接收tool.pid、tool.exitcode、retrieval-manifest.tsv、loop.json、progress.json五个小文件。精简schema2验证最终token、模型／配置身份和已结束子进程记录，不冒充schema1的逐候选／80状态主机重放。恢复不能切换原请求模式。

板端执行保留全部真实输入／输出和状态消费校验；原工具文件ABI仍需临时写盘。已消费的target输入及旧父输出、已验证的draft调用文件在子进程明确退出后清理；最后成功时不保留tensor、receipt或final-features.bin。失败或未知执行的scratch保留，精简回收只带小型失败信息。只清理当前attempt，draft的自身hardlink解除后严格更新预期nlink／ctime，原模型内容不变。

两个模式均输出accepted／considered／rate，以及prompt_ingest、decode、draft的calls／total_ms／mean_ms。prompt_ingest明确为decode1逐token输入prompt；时间为子进程墙钟，含加载／Run／卸载，纯NPU时间未提供。正常输出还包含最终token和supervisor总墙钟。

本轮共129项主机检查通过：native48项（`native-dump-tests.xml`，包括旧模式、三种接受分支、失败与未知child）、传输65项（`board-serial-transport.xml`）、真实C++调度器＋真实draft CLI的替身集成4项和prepared／CLI12项（`dump-integration.xml`）。完整对照两种模式生成相同的8个token，16次target、5次draft、2接受／4纠正／1bonus均相同；精简模式结束后无中间文件，模型hardlink数量恢复。该fixture的回收清单从231文件／78,999,820字节降至2文件／3,602字节，另有3个小型进程／清单文件；此处文件数量与大小是主机fixture观察，不是实机时延或加速比。native源码SHA256：`5e465d464ce7f5cfdf8fe2ee9804c2d68a1d636316b71acd0a2dd98af4b8998f`。

经验审计：临时工作文件与调试dump必须分开；不能把当前80状态提前删除。复用旧prepared恢复时不能静默改变模式。常驻PID／完成状态和精度证据不能因精简回收而混为一谈。以上约束固化在native清理／未知子进程测试、transport模式／小清单测试和真实CLI对照测试中；没有新增通用硬件规律或修改共享技能。该版之后用户已回传compact的serial子流程成功，原prefill完整P0尚未实机通过。

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

后续助手应继续实现并运行原prefill→draft／串行verify→多轮generation。已完成的serial子里程碑见[实机记录](board-first-completion.md)：compact tokens `[26,18,2,4,6,2,0,2]`，16target／7draft，接受0／考虑7，总墙钟367.791秒；prompt输入74.389秒、decode108.209秒、draft137.646秒。此为包含加载／执行／卸载的墙钟，不是NPU纯推理或配对性能。compact没有主机逐状态回放，物理断连未专门实测。

P0整体保持`in_progress`。`STARTED`、主机测试、历史host循环、单模型exit0或已经成功的serial-prompt流程，都不能替代原prefill驱动的完整阶段验收。
