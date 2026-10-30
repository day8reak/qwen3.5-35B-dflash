# 原prefill进入板端DFlash请求

本次实现复用原40层prefill64、原40层decode1和完整6层S64 draft三个OMC。服务器一次提供prompt token IDs、准备配置并启动；查原始embedding表、prefill、draft、逐token验证、状态推进及停止都在板端进行。这里只记录功能接入与主机验证，尚无这版代码的实机完成结果，P0保持进行中。

## 请求与状态流

以P8/G8为例：

1. 原prefill64实际运行一次，初始80状态为零。前8行来自原始INT8 embedding表和独立FP32 scale表，其余56行为零填充。
2. 选择`output_pos=7`的logits，保存前8行实际特征。全部80状态依据已重新核对的跨模型owner／原生端口映射交给decode。
3. prefill选出的首个token在绝对位置8进入decode；得到新状态和第9行特征后，draft收到完整9-token历史。
4. 原draft每轮输出15个候选，当前最多考虑2个。候选与当前target next-token比较，只有被接受或纠正后的已确认token进入decode；拒绝的候选不会污染target状态。
5. 重复至生成8个token。预期计数为`prefill_native_calls=1`、`target_native_calls=8`（此处只统计decode），draft次数由实际接受情况决定，至少2轮。

prefill子进程完成并退出后才调用decode；本版没有并行卸载／加载，也没有decode或draft常驻。`--detach`之后主机不逐token驱动，`--recover`只回收相同attempt。

## 显式填充实验

物理gear64和逻辑长度P分别记录。`position_ids`与`new_kv_cache_pos`都是0至63；填充KV写入未来位置，后续decode在覆盖前一直用mask屏蔽它们，不能让填充位置全部写0而破坏有效前缀。有效query按因果mask可见，填充query只允许key0以避免全负无穷mask。`allQLen=P`、`effective_length=P`、`conv_state_pos=[P,0,...,0]`（128项）和`output_pos=P-1`均保存于准备清单。

这些控制以`PREFILL64_PADDED_SHORT_PROMPT_EXPERIMENTAL`标记。原私有卷积／递归算子是否完整遵守短有效长度仍未数值验证；执行成功不关闭该问题。被消费的logits与有效特征必须有限，80状态按字节绑定；没有增加状态数值等价结论。

仍要求`P + G <= 64`，不裁真实历史、不重置位置。完整64-token prompt再加首个anchor需要65行，不能直接套用当前S64 draft；长上下文另按总计划解决。本版的8-token提示词是独立短请求。

## 证据和恢复

用户已回传`target-prefill64-decode1-feedback-plan-v2`成功，80个状态匹配；这只是结构映射，未执行反馈。新请求在主机重新检查图、编译产物、原生描述符和owner关系，板端重新执行prefill，绝不复用旧prefill的状态或logits作为新请求结果。

prefill与decode保留各自的模型、编译清单、缓存锁和权重目录。只有确认整个请求及其子进程退出后，恢复入口才释放本请求拥有的锁；失败或完成状态未知时保留现场，不自动重启模型。

默认compact仅回收token、小型完成记录、接受率和prefill／decode／draft时延。`--dump`还校验原表输入与全部控制、prefill零初态、80状态交接、完整有效特征历史和候选提交链。compact在下一步确实消费且验证成功后清理本次临时文件；失败保留现场。时延是包含加载／执行／卸载的子进程墙钟，`prompt_ingest`为`prefill`别名，不可再次相加。

命令见[current-status.md](current-status.md)。主机替身覆盖与真实C++集成记录见[board-serial-validation.md](board-serial-validation.md)。原始用户成功的serial请求保留为[独立子里程碑](board-first-completion.md)，不能替代本路径的设备验收。

## 本轮经验记录

- 原prefill计划使用`gear=64`；检查器兼容旧`sequence_length`，两者同时存在必须一致。拒绝冲突或错误类型，不改旧receipt。
- 真正的编译报告约10MiB，沿用32MiB的有界读取；双模型准备清单使用64MiB预算。不能用几KiB的诊断文件预算读取完整编译报告。
- 物理序列长度、有效历史长度与draft容量是三个独立约束。主机替身验证交接与调度，不能证明私有kernel的padding语义。
- 本轮没有新的通用技能规则或算子优化结论；这些实现约定及未验证假设保留在本模型文档和自动检查中。
