# 板端模型常驻与加载生命周期

2026-10-30，用户要求旧host循环继续运行，同时减少新版端侧执行开销，并优先考虑模型常驻。**本文是实现方案，不是已取得的常驻实机结果。** 保留`56ac4e9`的原工具路径作为回滚；两个OMC、权重、完整词表与状态语义均不变。

## 当前与目标

| 项目 | 当前已发布入口 | 常驻目标 |
|---|---|---|
| 调度／查表／接受判断 | 全在板端 | 全在板端 |
| prompt | decode1逐token读入P8 | 先沿用；P3桥接后用一次prefill |
| target | 每token启动原工具，反复加载 | 一个V2实例，多次Run |
| draft | 每轮启动原CLI，反复加载／模型SHA | 一个legacy-buffer实例，多次Process |
| 跨端数据 | 启动前与完整请求结束后 | 同左 |
| 实机状态 | 单模型曾运行；自主完整请求待结果 | 常驻初始化、双模型内存、重复执行待验证 |

现有`HiaiStatefulExecutor`在构造时加载模型并分配张量，重复`SetInput → Execute → CopyOutput`不会主动卸载，析构时才释放。因此常驻需要改执行器生命周期，不需要重新导出AIR或重编OMC。原target约15MB主OMC加18.8GB外置权重，必须保留文件／外置权重加载路线；不能按文件总大小推断实际内存占用。

## 先完成哪些适配

1. **原target V2实例。** 使用已绑定的89输入／82输出的名字、类型和大小；特别支持INT8 embedding与INT64 allQLen。旧draft的ND形状限制不能直接套用原target的物理描述符。运行前绑定实际描述符，保留80个逐一映射的状态。
2. **外置权重初始化。** 用户给出的所选DDK头文件有`Status InitWeights(const std::string& weightDir)`和`Status DeinitWeights()`，另有`Init`、同步`Run`、`DeInit`。头文件未说明初始化顺序；第一版独立实验采用RestoreFromFile、创建manager、InitWeights、Init，再多次Run。每阶段失败停止并记录；不自动换顺序重试，不冒称该顺序已经实机通过。
3. **完整draft实例。** 复用已成功的embedded legacy-buffer路线，绑定一次模型／ABI及其生命周期内身份；每轮只换完整prefix的features／IDs／mask。不得继续每轮建立新模型hardlink后又宣称旧ctime／nlink不变。
4. **独立session记录。** 常驻worker的PID可以跨调用相同；用session ID、连续call index和请求nonce绑定每次Run返回值，独立记录最终进程退出。不能把单次Run成功写成进程已退出，也不能放宽旧v1“每次新进程”的receipt校验来假装支持常驻。
5. **全板端调度。** 优先用两个常驻worker隔离target V2与draft legacy库环境，IPC只在板端。INIT后READY；STEP带输入及输出目标；成功响应绑定实际输出记录；完成所有调用后SHUTDOWN。失败停止，不自动重启再advance。主机只部署、启动和最终回收。

原80状态私有副本先保留。要进一步消除文件交接，再建立独立输入／输出buffer bank与提交所有权，防止RefData原地修改破坏上一状态或拒绝分支。常驻加载和状态零拷贝是两个改动，不混成一个未验证结论。

## prefill用完卸载及重叠加载

P3跨导出桥接成立后，单个请求先加载prefill并运行；等待同步Run完成，把80状态、features、logits复制到session持有的存储，再卸载prefill。随后加载decode一次，并在生成期间保持decode和draft实例。循环结束后统一释放。P0当前没有独立prefill加载，不能把其中8次decode prompt调用算成“一次prefill”。

卸载prefill与加载decode可以作为后续重叠实验，**目前没有接口并发及内存证据支持默认开启**。必须满足：无prefill未完成任务；输出不再引用即将释放的内存；runtime允许不同manager并发初始化／销毁；实测prefill未释放、decode正加载及draft驻留时的峰值内存有余量。文件大小不是峰值内存证明。优先收益是取消每token／每轮重复加载；启动交接重叠只优化一次转换，不作为首个闭环的前置条件。

## 验证和交接

- 主机替身验证：一次InitWeights／Init，多次Run，退出时正确释放；缺输入、错误ABI、输出对象替换和失败后继续调用均拒绝。
- 真实设备先验原target V2初始化和两次状态接续，再验两模型同时驻留。不得从元数据Restore通过推断外置权重加载通过。
- 常驻P8/G8按原完整请求验证16次target执行、多轮完整draft、80状态链和最终8token；原旧工具结果作为同输入对照。P0仍允许数值误差，P1/P2完整数值验收未取消。
- 记录每模型load count、实例ID、调用次数、加载／卸载／Run墙钟和峰值内存；是否减少总耗时以设备结果为准。Init失败不静默退回逐步加载后报告“常驻成功”。

预计常驻适配与联调0.5–2个有效工作日，私有权重初始化或内存问题另计。工程实现可在P0期间并行；完整数值和性能验收仍依赖P1/P2。当前默认入口的I/O优化可独立使用，不需等常驻完成。
