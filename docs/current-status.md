# 当前执行计划

<a id="用户现在执行什么"></a>

**常驻内存快速模式已经由用户实机跑通。** `dflash-board-resident-fast8-run/recovery` 完成原prefill1次、decode8次、完整六层draft7次，decode和draft各加载1次，正常关闭并释放所持锁。原三OMC及权重没有重编。本次是用户回传compact摘要，助手没有连接设备重取原artifact。

结果：`[4,2,24,2,4,22,0,2]`，接受0／7。decode包装平均**195.75ms**，其中同步Run **111.64ms**；draft包装740.71ms、Process741.11ms。prefill加载／运行／关闭10.285秒，其中Run484.542ms。均非纯NPU时间。

上次含诊断的resident请求88.680秒，本次**230.735秒**；主要问题是draft一次加载**202.958秒**（上次8.380秒）。decode包装下降约83%，但总耗时并未改善。不同请求的token路径不同，不是配对性能验收。详情见[快速模式首次结果](board-resident-fast-first-completion.md)及[结构化记录](../specs/board-resident-fast-first-completion.json)；[上次常驻结果](board-resident-mixed-first-completion.md)保留。

## 当前下一动作：draft顺序预读对照，文字入口可用

最新 `dflash-board-resident-loadtiming8-recovery` 已完成：总234.506秒，draft加载206.640秒，**SDK Load本身206.630秒、major fault 128,122次**；mmap59µs、SetModelBuffer1µs。decode Run112.819ms，draft Process739.192ms；接受0/7。观察已记录在[结构化结果](../specs/board-resident-load-observation.json)，不代表数值验收。

已准备可选 `--draft-preread`：在SDK Load前通过最多1MiB缓冲区顺序读取完整draft OMC，保留同一只读映射，分别计预读和Load时间。完整启动时间包含预读。使用[新请求与恢复命令](board-resident-load-timing.md#ddk主机下一次完整请求)，新目录为 `dflash-board-resident-preread8-run/recovery`。默认关闭，待用户设备对照；不改OMC，不改加载API，不清系统缓存。


用户要求增加真正的文字入口。新版 `--text` / `--text-file` 在同一板端进程中执行原tokenizer编码和最终文本解码，完整命令见[板端文字输入输出](board-text-input-output.md)。默认原单user聊天模板、thinking关闭，原三OMC和权重复用，服务器只在请求准备与最终回收做一次参考对照。文字请求同时包含以下加载分段计时，无需为两项功能各跑一次。

原S64预算仍是模板后输入＋生成≤64；生成策略仍为固定token预算，尚未实现EOS早停和多轮会话。文字接口不消除既有数值差异。

### draft加载203秒定位

用户要求先移除 `dump=false` 的校验和临时文件开销。本版新请求默认使用 `compact-memory-v1`：80 状态与特征在内存反馈，不写中间张量，移除逐步哈希／读回诊断／重复重叠扫描／细粒度日志，已有权重缓存不重新哈希，原表只读 mmap。独立 `native_api_latency` 给出同步 Run／Process 时间，与外层包装时延分开。

加载边界现已取得：SDK Load占据几乎全部draft构造时间。全量哈希曾提前读过模型，关闭哈希与慢Load同时出现，但尚未建立因果。顺序预读对照先保持原P8/G8 token输入；文字请求可另跑。不得仅因Load时间下降就宣布改善，需比较包含预读的完整启动时间。

## 随后的精度工作：定位首decode差异

旧输出 `[4,12,10,8,0,4,32,2]`、上次resident `[4,0,28,18,2,2,2,0]`、本次fast `[4,2,24,2,4,22,0,2]` 都从第2项分叉。prefill给出首token4后，调度器先执行decode(anchor4)，再执行第一次draft Process；因此差异在draft提议之前已经算出。draft此时已加载，共存影响仍需保留为待隔离因素。

后续执行助手按以下顺序推进：

1. 离线对照两个保留prepared请求的prompt、模型、embedding表、控制与80状态映射身份。相同首token不证明prefill状态和features相同。
2. 补充有界的prefill＋首decode捕获和比较，绑定完整89输入／82输出，在首个差异处定位；两个compact结果无法补出已经没有保留的中间张量。
3. 在同一runner实现普通target-only基线，对照target-only与pair已加载但未调用draft Process的首步输出和重复性。该功能尚未有可运行CLI，新命令在实现和主机验证后提供。
4. 先完成target参考与控制语义校验，再处理完整draft真实feature/anchor精度与接受率。之后验重复请求的内存、稳定性和同条件普通decode对比。

无诊断快速路径单次执行已完成；先定位加载耗时，再回到P1，不需编译S128或修改原模型。P5完整验收继续等待数值等价、重复稳定性及内存观察。阶段、依赖和预计天数见[总计划](dflash-delivery-plan.md)。

## 保留的请求与运行方式

- 最新加载定位：`$AI_RUN_DIR/dflash-board-resident-loadtiming8-run`，恢复目录 `$AI_RUN_DIR/dflash-board-resident-loadtiming8-recovery`。
- 首次快速模式：`$AI_RUN_DIR/dflash-board-resident-fast8-run`，恢复目录 `$AI_RUN_DIR/dflash-board-resident-fast8-recovery`。
- 上次常驻：`$AI_RUN_DIR/dflash-board-resident-mixed8-run`，恢复目录 `$AI_RUN_DIR/dflash-board-resident-mixed8-recovery`。
- 旧成功：`$AI_RUN_DIR/dflash-board-prefill8-run`，恢复目录 `$AI_RUN_DIR/dflash-board-prefill8-recovery`。
- 本次fast候选：`3042e63`，上次mixed候选：`060ea26`。显式 `--resident`，prefill/decode为file-v2，embedded draft为legacy-buffer。同一板端进程和线程；prefill一次后关闭，decode/draft各加载一次并多轮运行。
- 此前实机输入是token ID文件；新增板端文字入口待用户实机执行。物理S64的短prompt padding控制仍是实验。draft每次15候选，本次最多考虑前2个，实际每轮首候选都未接受。
- 新请求默认compact在内存运行，回收最终token、SDK与外层计时和小摘要；独立新请求的`--dump`保留详细证据。旧请求按其原配置恢复。

此前主机186项单元、7个唯一C++循环／恢复场景和18组native CTest通过，见[主机验证](../specs/resident-mixed-host-validation.json)；与本次用户实机观察分别记录。

本次快速模式主机192项单元、13个C++／fake SDK混合场景及18组native回归已通过，见[快速模式验证](../specs/board-resident-fast-host-validation.json)。最终6个快速路径场景重验通过；单次用户实机观察单列，未提升为配对速度或数值验收。

## 清理与内存的准确结论

本次V2 `Init → InitWeights → Run → DeInit` 具体调用已成功执行，DeInit正常返回、无清理异常。新策略不单独调用DeinitWeights，历史两种显式卸载顺序返回3／1的结果保留。

报告中的 `external_weight_release=NOT_INDEPENDENTLY_VALIDATED_MANAGER_DEINIT_ONLY` 仍成立：API关闭和对象释放不独立证明全部外权重立即回收。VmHWM报告值随阶段下降且始终等于VmRSS，暂不能用作真实累计峰值；RSS和MemAvailable也不等于专用NPU显存。下一阶段以重复运行和可解释的资源观测验收，不从单次正常退出推断无泄漏。
