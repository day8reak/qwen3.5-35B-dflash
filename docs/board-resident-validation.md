# 统一 V2 常驻入口：主机验证与首次加载失败

更新：2026-10-30。实现入口为 `run_board_serial_dflash.py --resident`，二进制为 `qwen35_board_resident`。完整参数见 [当前执行命令](current-status.md)，生命周期与计时定义见 [常驻方案](resident-runtime-plan.md)。原 prefill P0 成功保存在 [独立记录](board-prefill-first-completion.md)，不能用它证明本次常驻路径已在设备运行。

## 首次设备结果与诊断修复

用户已在实际 DDK／设备执行常驻请求。prefill 完成一次；随后 target 在 `InitWeights` 返回 `3=INVALID_PARAM`，此时尚未调用 manager `Init`，draft 尚未加载。请求失败结果已回收，所持锁已释放。[设备记录](../specs/resident-v2-first-load-failure.json)保留用户回传的阶段、计时与接口声明，未把错误码解释为内存不足。

构造函数抛出异常时，原实现没有把部分生命周期传给 session，导致摘要中的 attempted／cleanup 字段保持默认值。诊断修复在清理后保留实际调用、返回状态、最初失败阶段及清理错误；未调用或异常无返回使用 `null`。普通输出也直接列出执行错误及执行失败阶段，回收的 `stage=complete` 仍仅表示回收完成。

本次修复保持加载顺序不变，不解决 `InitWeights` 参数拒绝。主机验证：85 项传输／摘要回归、4 项实际 adapter CTest、1 项实际 supervisor＋adapter＋FakeHIAI 加载失败 JSON 集成通过。源码和测试产物见[诊断修复记录](../specs/resident-v2-failure-reporting-validation.json)。下一步核对原成功工具的 V2 加载实现；不为验证摘要而重复整轮推理。

## 本次实现

- 一次原 prefill 工具调用完成后，同一板端进程保留原 decode 和 embedded draft 的两个 HIAI V2 manager，分别从文件恢复、加载一次、交替多次 Run、请求结束关闭。
- 两模型使用独立且经 GetName 检查的名称；原 target 绑定原生 89／82 描述符，draft 保持 S64 的完整 3／2 ABI。原三个 OMC、权重、候选生成和已提交状态规则不变。
- 保留176个独立输入／输出缓冲区，检查区间不重叠、地址和对象稳定；每次输入读回及输出复制有记录。80 状态仍通过板端私有文件／独立副本传递，未实现零拷贝。
- 常驻 receipt 用 session ID、同一个进程 PID、连续 call index；仅 prefill 记真实子进程退出。新 loop schema3 与旧请求回收规则分开，不放宽旧版 PID／退出规则。
- 失败停止复用，关闭已加载的模型。draft 的显式 Close 捕获 DeInit 异常、继续关闭 target、记录失败；析构不重复调用已尝试的关闭，不因清理异常静默退出。没有 legacy fallback 或自动重新推理。
- 普通输出附接受率、prefill／decode／draft 墙钟、一次性加载／关闭耗时与进程 RSS 摘要。默认仅回收 token／小摘要，`--dump` 保留和验证详细链路。

## 验证范围

工作区 run：`20261030T122833Z-23441-5814f9`。日志／测试产物都在该 run 的 `out/`；源码与结果文件哈希见 [机器可读验证记录](../specs/resident-v2-host-validation.json)。以下各组有回归和定向复验重叠，不将次数相加冒充独立测试数量。

| 检查 | 结果及范围 |
|---|---|
| 原准备／部署／恢复 Python 回归与新 ABI／生命周期检查 | 最终101项通过，包含相同session、phase计时、RSS、恢复不重启和坏摘要拒绝 |
| 原 native 调度路径 | 62 项通过，保持按次加载行为 |
| 新 resident 调度协议 | 16 项通过，两种 dump 模式与接受／纠正／混合分支、状态排列、实际prefill子进程、失败保留；使用明确标识的 Session 替身 |
| 实际 HIAI adapter／ResidentSession | Release CTest 4／4 通过；包括新 session 与原 target、legacy draft、file-v2 draft 回归 |
| runner 构建配置 | 16 项通过，模型编译状态保持 NOT_RUN |
| 真实 session＋adapter 的完整调度集成 | 实际 C++ supervisor／ResidentSession／HIAI adapter 接 FakeHIAI；关闭修复后最终2项通过，含full成功、compact成功及失败Run回收三个请求 |

实际 session 测试观察两个 Restore／Init、176 次 tensor 分配仅发生一次、交替 target3＋draft3 共6次 V2 Run、两次 DeInit 与一次外置权重 DeinitWeights，没有 legacy Load／Process／Unload。覆盖13种失败：部分加载、错误描述符、同零内容跨模型别名、输出替换、运行异常、缺输入、重复使用及双方清理异常等。

完整调度集成保留生产 89／82 描述符及80状态输出重排；不是把 session 返回值整体写死的测试。只有底层 HIAI 和 prefill 工具是明确的 host fake。验证一次 prefill、8 次实际 adapter decode Run、多次 draft Run、完整prefix／feature／state／候选／greedy提交，以及独立full／compact回收。失败 Run 的回收不得凭空制造模型子进程退出。

## 尚未取得的结果

- 本机所声明 CPro profile manifest 缺失，没有本地实际 OHOS／DDK 链接结果；未修改共享环境。用户已经构建并运行首个版本，但新增诊断修复尚未取得设备结果。
- 没有执行设备／HDC 探测，遵守用户“此处连不上，不再检查”的要求。
- 原 external-weight target 的 InitWeights-before-Init 尝试已被拒绝，具体参数／顺序原因待定位；当前 embedded draft 的 V2 初始化、两个manager同时驻留及内存余量尚未验证。
- 新版多轮 token／state 是否与旧成功运行一致、实际耗时改善、重复请求内存稳定和物理断连尚未验收。FakeHIAI 的运行时长不作性能数据。
- P1/P2 数值精度及有效接受率未完成；既有完整请求接受 0／7，新运行方式不能自动提升精度。

下一动作是[独立 target 加载／关闭实验](target-resident-load-probe.md)，只尝试一次显式 Init-before-weights，复用缓存且不调用 Run。实际成功工具的 V2 源码仍未取得，用户提供的另一份源码片段走旧接口；不能据此推断 V2 契约。实验通过后才接入完整常驻路线并使用新目录运行。
