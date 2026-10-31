# 板端混合接口常驻执行

更新：2026-10-31。用户已明确允许prefill／decode用V2、完整六层embedded draft用V1，优先推进常驻。全V1候选的HCL空buffer失败保留，当前命令见[current-status.md](current-status.md)。

## 单线程生命周期

```text
板端进程
  prefill V2：RestoreFromFile → Init → InitWeights → Run一次 → DeInit → 释放
  decode  V2：RestoreFromFile → Init → InitWeights ─┐
  draft   V1：Init → SetModelBuffer → Load一次 ────┤交替多次Run／Process
  结束：draft UnLoadModel并释放 → decode DeInit并释放
```

prefill释放后才加载常驻二模型，不并行加载卸载。每个实例仅加载一次，不跨线程共享、无异步队列。模型中途失败停止推进，保留首个错误及清理异常，不重试加载、不自动切换API。所有热路径工作在板端，服务器只部署、启动和最终回收。

## 模型与输入输出

复用原prefill64、decode1、draft S64三个OMC及权重，均不重编。target原89输入／82输出、80状态映射保持；draft3输入／2输出，每次15候选，最多考虑前K=2。物理S64、短prompt实验控制与P+G<=64容量限制不变，不新增draft KV ABI。

V2 prefill／decode分别绑定自己的named ABI并逐槽核对name、dtype、format、shape及字节数；不把decode1描述符用于prefill64。V1 draft使用已有完整embedded buffer路径，保留模型buffer到Unload结束。检查两实例tensor区间不重叠且反复调用中不变；输入／输出和状态仍采用独立副本，不宣称零拷贝。

## V2关闭策略及限制

新mixed请求在构造前选择 `ManagerDeinitOnly`，准确输出 `MANAGER_DEINIT_ONLY_NO_SEPARATE_WEIGHT_UNLOAD`。调用DeInit后才销毁manager，随后释放built model与tensor对象；部分初始化失败也按同一所有权范围清理。called与returned分开记录；void接口不伪造status0。未调用DeinitWeights则called=false、status=-1，历史返回1／3不改写。

这是一条明确的实验策略，不能宣称私有InitWeights的全部权重已被独立证明释放。报告保留 `external_weight_release=NOT_INDEPENDENTLY_VALIDATED_MANAGER_DEINIT_ONLY`。DeInit抛异常仍报告失败，析构不再次重试。旧V2 probe保持原显式weight-unload顺序，历史回执按原合同验证。

prefill加载前／加载后／关闭后，以及常驻pair加载前／加载后／关闭后记录RSS/HWM与系统MemAvailable。它们是诊断观察，不能直接换算为NPU显存或作为权重释放PASS。

## 报告与验证

`execution_mode=resident-mixed`，`model_load_api=target-v2-draft-v1`，三模型source API分别file-v2／file-v2／legacy-buffer。真实manager调用与wrapper计数分别保留；prefill是本进程一次调用，无虚构子进程PID。普通输出仍为最终token、接受率和prefill/decode/draft时延；运行时间包含复制、读回、哈希检查，排除常驻模型的一次性加载/关闭，不能称纯NPU推理时间。

主机测试验证调度、持有期及错误处理，随后用户直接跑完整板端候选，对照旧同输入token与调用计数，再验重复请求内存和性能。首次实机闭环尚未回传前不宣称mixed常驻、清理、数值或加速通过。旧请求仍按原schema和配置恢复，原成功P0基线见[记录](board-prefill-first-completion.md)。
