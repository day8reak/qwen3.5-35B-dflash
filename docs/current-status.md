# 当前执行计划

<a id="用户现在执行什么"></a>

**原prefill＋常驻decode/draft自主闭环已跑通；顺序预读已在本轮实机显著缩短启动。** 用户回传 `dflash-board-resident-preread8-recovery` 完成prefill1次、decode8次、完整六层draft7次，decode/draft各加载1次，正常关闭并释放所持锁。结果为 `[4,0,4,0,2,28,8,0]`，接受0／7。助手未连接设备重取artifact，数值质量仍未验收。

| 指标 | 上轮无预读 | 本轮开启预读 |
|---|---:|---:|
| draft总加载（包含预读） | 206.640秒 | 3.883秒 |
| SDK Load | 206.630秒 | 2.251秒 |
| 顺序预读 | 未执行 | 1.621秒 |
| 完整请求墙钟 | 234.506秒 | 31.455秒 |
| decode同步Run均值 | 112.819ms | 110.890ms |
| draft同步Process均值 | 739.192ms | 733.126ms |

主要收益在启动阶段。完整请求本轮减少86.6%；不能据此认定数值等价后的推测解码加速或重复稳定性。详细[加载记录](board-resident-load-timing.md)与[结构化结果](../specs/board-resident-preread-observation.json)保留；先前fast/mixed结果分别在[fast记录](history/board-resident-fast-first-completion.md)、[mixed记录](history/board-resident-mixed-first-completion.md)。

## 用户下一条操作：文字输入输出＋draft预读合并请求

按[离线准备指南](tokenizer-offline.md)传入三个原词表文件并设置`TOKENIZER_DIR`，再按[完整命令](board-text-input-output.md)执行。已有原tokenizer目录时跳过准备步骤；同一条启动命令同时给 `--text` 和 `--draft-preread`，新输出目录为 `dflash-board-text-preread8-run`，完成后恢复到 `dflash-board-text-preread8-recovery`。

板端同一进程完成文字编码→原prefill一次→加载decode与预读后的draft→常驻验证循环→文字解码，最终一次回传文字、token ID和时延。CLI预读仍显式开启；历史请求保持原策略。模型OMC和权重复用。

文字入口和预读单项已有主机验证，预读完整token输入循环已有本轮实机结果；组合文字实机结果待回传。模板后输入＋生成仍须≤64，单条user模板、thinking关闭、固定生成预算，不按EOS早停。输出数值与语言质量仍按后续P1/P2处理。

## 随后的精度工作：定位首decode差异

旧输出 `[4,12,10,8,0,4,32,2]`、上次resident `[4,0,28,18,2,2,2,0]`、本次fast `[4,2,24,2,4,22,0,2]` 都从第2项分叉。prefill给出首token4后，调度器先执行decode(anchor4)，再执行第一次draft Process；因此差异在draft提议之前已经算出。draft此时已加载，共存影响仍需保留为待隔离因素。

后续执行助手按以下顺序推进：

1. 离线对照两个保留prepared请求的prompt、模型、embedding表、控制与80状态映射身份。相同首token不证明prefill状态和features相同。
2. 补充有界的prefill＋首decode捕获和比较，绑定完整89输入／82输出，在首个差异处定位；两个compact结果无法补出已经没有保留的中间张量。
3. 在同一runner实现普通target-only基线，对照target-only与pair已加载但未调用draft Process的首步输出和重复性。该功能尚未有可运行CLI，新命令在实现和主机验证后提供。
4. 先完成target参考与控制语义校验，再处理完整draft真实feature/anchor精度与接受率。之后验重复请求的内存、稳定性和同条件普通decode对比。

无诊断快速路径和显式预读单次执行已完成；当前先运行文字＋预读组合，再回到P1，不需编译S128或修改原模型。P5完整验收继续等待数值等价、重复稳定性及内存观察。阶段、依赖和预计天数见[总计划](dflash-delivery-plan.md)。

## 保留的请求与运行方式

- 最新预读成功：`$AI_RUN_DIR/dflash-board-resident-preread8-run`，恢复目录 `$AI_RUN_DIR/dflash-board-resident-preread8-recovery`。
- 上轮加载定位：`$AI_RUN_DIR/dflash-board-resident-loadtiming8-run`，恢复目录 `$AI_RUN_DIR/dflash-board-resident-loadtiming8-recovery`。
- 首次快速模式：`$AI_RUN_DIR/dflash-board-resident-fast8-run`，恢复目录 `$AI_RUN_DIR/dflash-board-resident-fast8-recovery`。
- 上次常驻：`$AI_RUN_DIR/dflash-board-resident-mixed8-run`，恢复目录 `$AI_RUN_DIR/dflash-board-resident-mixed8-recovery`。
- 旧成功：`$AI_RUN_DIR/dflash-board-prefill8-run`，恢复目录 `$AI_RUN_DIR/dflash-board-prefill8-recovery`。
- 本次预读候选：`3c270a7`；fast候选：`3042e63`，mixed候选：`060ea26`。显式 `--resident`，prefill/decode为file-v2，embedded draft为legacy-buffer。同一板端进程和线程；prefill一次后关闭，decode/draft各加载一次并多轮运行。
- 此前实机输入是token ID文件；新增板端文字入口待用户实机执行。物理S64的短prompt padding控制仍是实验。draft每次15候选，本次最多考虑前2个，实际每轮首候选都未接受。
- 新请求默认compact在内存运行，回收最终token、SDK与外层计时和小摘要；独立新请求的`--dump`保留详细证据。旧请求按其原配置恢复。

此前主机186项单元、7个唯一C++循环／恢复场景和18组native CTest通过，见[主机验证](../specs/resident-mixed-host-validation.json)；与本次用户实机观察分别记录。

本次快速模式主机192项单元、13个C++／fake SDK混合场景及18组native回归已通过，见[快速模式验证](../specs/board-resident-fast-host-validation.json)。最终6个快速路径场景重验通过；单次用户实机观察单列，未提升为配对速度或数值验收。

## 清理与内存的准确结论

本次V2 `Init → InitWeights → Run → DeInit` 具体调用已成功执行，DeInit正常返回、无清理异常。新策略不单独调用DeinitWeights，历史两种显式卸载顺序返回3／1的结果保留。

报告中的 `external_weight_release=NOT_INDEPENDENTLY_VALIDATED_MANAGER_DEINIT_ONLY` 仍成立：API关闭和对象释放不独立证明全部外权重立即回收。VmHWM报告值随阶段下降且始终等于VmRSS，暂不能用作真实累计峰值；RSS和MemAvailable也不等于专用NPU显存。下一阶段以重复运行和可解释的资源观测验收，不从单次正常退出推断无泄漏。
