# 板端统一 V1 常驻执行

更新：2026-10-30。用户明确要求prefill、decode、draft的新执行接口全部使用V1。此前V2加载两次成功但关闭失败，用户随后提供官方弃用说明；V2路线保留为历史诊断和恢复，最新命令见[当前入口](current-status.md)。

## 单线程生命周期

```text
板端进程
  prefill V1：Init → Load一次 → Process一次 → UnLoadModel → 释放
  decode  V1：Init → Load一次 ─┐
  draft   V1：Init → Load一次 ─┤交替多次Process → 各自UnLoadModel → 释放
```

同一执行线程操作各自独立的模型管家，不跨线程共享；每个实例只调用一次Load，不用异步队列。每实例只装一个模型，prefill结束才创建两个常驻实例。符合用户提供的单实例Load一次、同步最多63模型、APP最多256实例约束；异步31模型／512队列限制不涉及本路线。

prefill也使用native V1执行，原V2模型工具不参与新请求。服务器只做部署、启动与最终回收。读取embedding表、候选生成、decode1 verify、接受／纠正、状态与特征更新都在板端，模型只加载一次，不在每轮卸载。

## 模型与输入输出

复用原prefill64、decode1和六层draft S64的三个OMC及权重，均不重编。target维持原89输入／82输出和80状态映射；draft维持3输入／2输出、15候选、最多考虑K=2的逐token验证。短prompt控制及P+G<=64容量边界不变。

V1加载沿用已有legacy路径的模型描述和映射读取；保持原OMC绝对路径及邻接的外置权重文件。不得拼接权重进OMC、暗改weight目录或在失败时切V2。外置target在这条V1路径能否正确加载仍是实机待验证项。

两套target ABI来自已经绑定的源图及历史runtime描述符，新V1分别核对实际描述符shape、format、dtype对应分配与字节数。历史V2元数据只能作为来源；V1接口未给出的tensor名称或dtype不能标成V1已观察。相同尺寸不足以证明不同state层语义，保留原逐slot映射和实验范围。

所有buffer由各自模型实例持有，Process同步返回后检查输出对象/地址与实际输入读回。80状态和特征仍采用独立副本及板端私有文件，不在本次生命周期迁移中引入零拷贝。

## 结果、失败与验证

新结果明确标注resident-v1和V1 lifecycle policy，记录各模型Init、Load、Process、Unload尝试/结果及耗时。同实例不重复Load；清理失败仍保留错误且不重复卸载。任一运行失败停止推进，不回退V2。

prefill是同进程的一次调用，不再伪造子进程PID/退出码。decode和draft的计时排除一次性加载/关闭，保留输入复制、输出读回和校验的实际范围；不是纯NPU计时。默认只返回最终token、接受率、时延与进程RSS摘要；dump启用时保留逐步证据。

验证顺序为主机生命周期/失败回归，随后用户实机首次V1初始化、prefill→decode/draft多轮、与旧成功token对照，再做重复请求内存与性能。主机fake SDK结果不证明真实外置权重兼容、精度或加速。

旧V2/fresh-process请求仍按原schema、参数、PID和哈希回收，不把旧V2结果冒充V1。原成功基线见[记录](board-prefill-first-completion.md)，V2历史验证见[记录](board-resident-validation.md)，新的V1验证见[V1记录](resident-v1-validation.md)。
