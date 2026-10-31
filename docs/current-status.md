# 当前执行计划

<a id="用户现在执行什么"></a>

**混合接口常驻的完整板端闭环已经由用户实机跑通。** `dflash-board-resident-mixed8-run/recovery` 完成原prefill1次、decode8次、完整六层draft7次，decode和draft各加载1次，正常关闭并释放所持锁。原三OMC及权重没有重编。本次是用户回传compact摘要，助手没有连接设备重取原artifact。

结果：`[4,0,28,18,2,2,2,0]`，接受0／7，总墙钟**88.680秒**。decode调用平均1.149秒，draft平均0.748秒；均包含复制、读回、哈希／buffer检查，非纯NPU时间。prefill加载／运行／关闭合计13.833秒，其中运行包装1.931秒。

与旧逐次加载版本305.788秒相比，单次总耗时减少约71%。这是框架常驻改进的观察，两个版本的token路径和phase计时范围不同，不作为DFlash算法加速或最终性能验收。详情见[首次常驻实机结果](board-resident-mixed-first-completion.md)及[结构化记录](../specs/board-resident-mixed-first-completion.json)。

## 当前下一动作：运行无诊断的内存模式

用户要求先移除 `dump=false` 的校验和临时文件开销。本版新请求默认使用 `compact-memory-v1`：80 状态与特征在内存反馈，不写中间张量，移除逐步哈希／读回诊断／重复重叠扫描／细粒度日志，已有权重缓存不重新哈希，原表只读 mmap。独立 `native_api_latency` 给出同步 Run／Process 时间，与外层包装时延分开。

**完整启动与回收命令见 [快速模式](board-resident-fast-path.md#在-ddk-主机执行)**，使用 `dflash-board-resident-fast8-run/recovery` 新目录。历史成功的 88.680 秒结果仍保留，不能当成本版测速；本版设备结果待用户回传。新模型不重编，旧请求按原模式 recover。

## 随后的精度工作：定位首decode差异

旧输出 `[4,12,10,8,0,4,32,2]` 与新输出第2项起不同。prefill给出首token4后，调度器先执行decode(anchor4)，再执行第一次draft Process；因此第二token的新0／旧12在draft提议之前已经算出。draft此时已加载，共存影响仍需保留为待隔离因素。

后续执行助手按以下顺序推进：

1. 离线对照两个保留prepared请求的prompt、模型、embedding表、控制与80状态映射身份。相同首token不证明prefill状态和features相同。
2. 补充有界的prefill＋首decode捕获和比较，绑定完整89输入／82输出，在首个差异处定位；两个compact结果无法补出已经没有保留的中间张量。
3. 在同一runner实现普通target-only基线，对照target-only与pair已加载但未调用draft Process的首步输出和重复性。该功能尚未有可运行CLI，新命令在实现和主机验证后提供。
4. 先完成target参考与控制语义校验，再处理完整draft真实feature/anchor精度与接受率。之后验重复请求的内存、稳定性和同条件普通decode对比。

本次先按用户要求验证无诊断快速路径；随后回到P1，不需编译S128或修改原模型。P5的单次常驻执行子项已完成，P5完整验收继续等待数值等价、重复稳定性及内存观察。阶段、依赖和预计天数见[总计划](dflash-delivery-plan.md)。

## 保留的请求与运行方式

- 新成功：`$AI_RUN_DIR/dflash-board-resident-mixed8-run`，恢复目录 `$AI_RUN_DIR/dflash-board-resident-mixed8-recovery`。
- 旧成功：`$AI_RUN_DIR/dflash-board-prefill8-run`，恢复目录 `$AI_RUN_DIR/dflash-board-prefill8-recovery`。
- 上次实机对应候选：`060ea26`；当前快速模式在它之上继续实现。显式 `--resident`，prefill/decode为file-v2，embedded draft为legacy-buffer。同一板端进程和线程；prefill一次后关闭，decode/draft各加载一次并多轮运行。
- 输入是token ID文件；物理S64、逻辑8的prefill控制仍是实验，尚未包含板端分词／文字解码。draft每次15候选，本次最多考虑前2个，实际每轮首候选都未接受。
- 新请求默认compact在内存运行，回收最终token、SDK与外层计时和小摘要；独立新请求的`--dump`保留详细证据。旧请求按其原配置恢复。

此前主机186项单元、7个唯一C++循环／恢复场景和18组native CTest通过，见[主机验证](../specs/resident-mixed-host-validation.json)；与本次用户实机观察分别记录。

本次快速模式主机192项单元、13个C++／fake SDK混合场景及18组native回归已通过，见[快速模式验证](../specs/board-resident-fast-host-validation.json)。最终6个快速路径场景重验通过；实机速度仍待下一次用户执行。

## 清理与内存的准确结论

本次V2 `Init → InitWeights → Run → DeInit` 具体调用已成功执行，DeInit正常返回、无清理异常。新策略不单独调用DeinitWeights，历史两种显式卸载顺序返回3／1的结果保留。

报告中的 `external_weight_release=NOT_INDEPENDENTLY_VALIDATED_MANAGER_DEINIT_ONLY` 仍成立：API关闭和对象释放不独立证明全部外权重立即回收。VmHWM报告值随阶段下降且始终等于VmRSS，暂不能用作真实累计峰值；RSS和MemAvailable也不等于专用NPU显存。下一阶段以重复运行和可解释的资源观测验收，不从单次正常退出推断无泄漏。
