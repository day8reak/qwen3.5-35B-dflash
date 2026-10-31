# 全 V1 常驻路线验证

> 历史记录：保留当时的观察、失败和候选命令。当前操作请从[文档首页](../README.md)或[当前状态](../current-status.md)进入；不要直接重复本页旧请求。


当前新执行入口为 `run_board_serial_dflash.py --resident`，显式准备 `execution_mode=resident-v1`。它与历史V2请求分开，完整命令见[当前入口](../current-status.md)，流程见[常驻方案](resident-runtime-plan.md)。

## 路线选择与证据

用户提供了官网V2弃用说明并明确要求全部接口改V1，同时提供V1模型管家非线程安全、每实例只Load一次及实例/模型数限制。公开[FormatMode说明](https://developer.huawei.com/consumer/cn/doc/hiai-References/formatmode-0000001092198778)也注明该V2接口自DDK5.0.1.0废弃并建议V1替代。完整下架日期来自用户所贴说明；不能将弃用直接认定为本设备DeinitWeights失败的根因。

旧P0执行基线使用原工具V2 target与legacy draft；它的完整token及执行次数在[旧基线](board-prefill-first-completion.md)。新V1外置target的加载/Process、prefill到decode连续性和两模型常驻尚需实机验证，不能继承旧路径通过状态。

## 迁移范围

- 一个线程中的独立prefill实例，Load/Process/Unload各一次；释放完成后才创建decode和draft两个实例。
- decode、draft各Load一次后交替同步Process，请求结束各自Unload。无RunAsync、异步队列或多线程共享模型管家。
- 全部新推理路径调用V1；prefill也不再启动旧V2模型工具。V2类及旧schema保留供历史工具/回收，不作为自动fallback。
- target继续89输入/82输出，prefill与decode分别绑定ABI；V1实际shape/format/allocated bytes要匹配，无法查询的SDK名字/dtype不伪造。旧描述符来源明确标为历史元数据。
- 所有OMC和权重不变。模型绝对路径与旁置权重位置保持，SetModelBuffer之后仍检查SetModelPath的路径保留；外置权重解析是实机待验项。
- 最初失败和关闭失败分开保留，不重复Load/Unload，不因清理失败丢弃已经生成的token。

## 主机检查

本次工作区run为 `20261030T133340Z-32275-390e70`。新增V1生命周期、协议、完整调度和历史恢复回归均只使用明确标识的FakeHIAI/FakeHDC；逐项结果、最终源码与产物哈希见[主机验证记录](../../specs/resident-v1-host-validation.json)。所选覆盖为159个独立pytest用例及5个CTest组；初轮失败及定向复验均保留，已修正缓冲区写入前检查和构造后段失败回收。最终native实际适配器在full/compact模式均完成prefill1、decode8、draft3，零V2调用及零子进程；这组token来自fake模型。主机测试不属于真实设备执行或性能数据。

V2关闭失败回归的单项测试已完成，证据在[历史记录](../../specs/resident-v2-close-failure-retention-validation.json)；它只能证明已完成的生成在关闭失败后仍保留，不能计为V1通过。

## 实机验收

用户侧新请求要核对prefill1、decode8、draft多轮，以及三个模型分别一次Load、所有Process走V1、两模型循环期间驻留。与相同输入旧token `[4,12,10,8,0,4,32,2]` 对照；不一致先检查迁移差异。

尚未验收：真实外置权重加载、V1 tensor ABI/slot顺序、完整state/token数值、正常卸载与重复请求内存稳定，以及配对加速。单次初始化成功或主机fake通过不足以补齐这些项。
