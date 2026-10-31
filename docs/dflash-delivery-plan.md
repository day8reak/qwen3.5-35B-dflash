# Qwen3.5-35B DFlash 后续执行计划与交接

更新：2026-10-31。旧主机联调基线：`4ea1c105c66b5014c2c1260ad9f757a096c38028`；当前端侧自主实现见本提交。

本文件是后续执行助手的总计划。机器可读进度见 [dflash-execution-state.json](../specs/dflash-execution-state.json)，立即可运行的命令见 [current-status.md](current-status.md)。每次接手先读这三份；旧验证记录用于查证，不按历史记录重新跑一遍。

## 1. 目标与本次约定

最终交付：原40层量化 target＋完整训练六层 draft，真实特征、正确的提议／验证／接受／拒绝／状态提交，支持实际提示词与多轮生成；通过精度和稳定性后，在相同条件下实测 DFlash decode-loop 中位延迟至少优于普通 decode **2倍**，同时报告包含 prefill 的总延迟。2倍仍是目标，尚无实测保证。

用户最新优先级：**第一阶段必须在端侧自主完成原prefill→draft／串行verify→多轮generation**。decode1串行输入prompt的成功只算子里程碑，不能替代原prefill。服务器只部署、启动与结束后回收；请求启动后，embedding查表、prompt输入、draft、verify、接受／拒绝、80状态与特征反馈以及停止都在端侧完成。中断主机连接不能成为板端停止生成的原因。主机调度循环只算联调，不能验收P0。允许当前阶段乱码／数值误差；之后逐步修精度和速度。该放宽只适用于 P0 执行里程碑，不永久取消最终精度、语言质量或性能要求。

固定模型边界：target40层、原量化权重；draft6层、hidden2048、intermediate6144、block16、15候选、完整248320词表；target第 `[1,6,11,16,22,27,32,37]` 层特征拼接为16384。不能用 tiny 结果替代完整模型，不减层、不缩词表、不悄悄换权重。4B仓库只作接口／调度参考，不能复制其模型尺寸、5层量化checkpoint或310P二进制。

## 2. 大致需要几天

以下是**从当前状态往后**的计划估计，1个工作日按6–8小时有效开发、原权重机器和设备基本可用计算。是工程预算，不是承诺；编译／传输／设备故障以及私有算子实现问题会增加时间。

| 到达的结果 | 顺利时的累计工作日 | 说明 |
|---|---:|---|
| P0：原prefill驱动的端侧自主多轮闭环，允许精度误差 | 已完成执行 | 用户回传prefill1／decode8／draft7成功，精度未验收 |
| P1–P3：可对照的普通基线、精度收敛、prefill→decode数值验收 | 3–7天 | 前提是原参考与控制实现可取得；底层算子是主要风险 |
| P4–P6：长上下文、常驻执行、批量verify及正确回滚 | 7–14天 | 含新增artifact编译与实机联调；允许部分并行 |
| P7–P8：稳定性、配对性能、优化迭代与交付 | 10–20天 | 按2–4个工作周安排；达到2倍取决于接受率和实测瓶颈 |

若参考资料齐备、数值问题较轻且设备联调顺利，完整交付最快约7–10个工作日。常规按10–20天预算；如果需要返工私有GDR／量化算子、处理两模型重复驻留不稳定，或draft接受率持续很低，另留3–7个工作日甚至更久，整体可能超过4周。不能到了预估日期就把未通过项改成通过。只想拿到首次闭环，预算看第一行；要可靠、可用且有2倍收益，看最后一行。

下面各阶段工时是局部估计，有重叠，**不要把它们再加到累计时间上**。每次完成一个阶段，应按实际耗时和新发现更新剩余估计。

最新进展：P0与P5单次混合常驻执行已由用户实机完成（88.680秒，prefill1／decode8／draft7、各resident模型加载1次）。P5无诊断内存模式也已单次实机执行通过（decode包装195.75ms、Run111.64ms）；当前先定位draft加载202.958秒的问题，见[首次快速结果](board-resident-fast-first-completion.md)，随后回到P1首decode差异隔离；P5完整验收仍需数值等价、重复稳定性与内存检查。总预算维持3–7天精度／10–20天完整交付的估计，不以单次执行成功替代验收。

## 3. 当前真实进展

| 项目 | 当前结论 | 限制 |
|---|---|---|
| 原target prefill64、decode1 OMC | 用户实机分别执行成功，89输入／82输出已取回 | 不能据此认定精度正确 |
| 同模型80状态接续 | 已观察父输出字节进入下一次实际输入 | 控制语义／普通生成精度未验收 |
| 完整六层draft | 已运行，固定fixture报告Top1 15/15；也运行过真实target特征单步 | 不代表真实多轮接受率／精度已通过 |
| row63 prefill试验 | PID28669成功；feature SHA不变，logits argmax由0变6 | 只证明该单变量试验改变了logits，未判定数值根因 |
| embedding两张原表 | 用户确认已复制完成 | 新入口仍核对文件大小和实际读取行 |
| 主机多轮调度＋两实机adapter | 旧任务在10次target／1次draft后遇runner复用路径错误；已修复，用户选择新版 | 仅联调工具，**不满足端侧自主P0目标** |
| 端侧serial调度器＋部署／恢复入口 | 完整dump及compact用户回传成功；compact tokens `[26,18,2,4,6,2,0,2]`，16target／7draft，0／7接受 | 历史 serial-prompt 子里程碑；另有原 prefill 完整 P0 成功结果，物理断连／精度／常驻／配对性能未验收 |
| 混合接口常驻闭环 | 用户实机prefill1／decode8／draft7；target/draft各Load1，总墙钟88.680秒 | 0／7接受；第二token与旧版分叉；重复稳定性、独立权重释放及配对算法收益未验收 |
| 增量draft缓存图 | `ge_incremental_draft.py`及独立主机等价测试已存在 | **没有新增AIR／OMC／实机增量ABI通过结果** |
| 精度／稳定性／2倍加速 | 待完成 | 不从有限值、exit0或host测试推断通过 |

旧64-token prompt把现draft S64的输入容量用满；标准下一token anchor需要第65行。当前采用独立8-token请求，完整历史保留，复用旧OMC。新prefill路径将8个有效token填充到物理64行，首个anchor推进decode后使用完整9行历史；合同已实现并通过主机检查，私有算子的有效长度语义仍待数值验证。它不是裁掉旧64-token请求后冒称等价，不自动指定S128。

## 4. 接手助手的执行规则

1. 推进状态文件的 active_phase，读依赖、下一动作和证据；用户已要求在 P0 完成后优先推进 P5 工程，P1/P2 数值门槛仍保留。可并行独立 host 工作，设备模型调用遵守共享锁。
2. 先复用已成功的源／模型／缓存和当前环境。不要重跑成功的serial请求、row63实验或旧metadata探针。当前按已成功的原prefill短prompt请求验证新常驻运行方式；不把S128当作默认前置条件，不通过截断历史或改manifest冒充新容量。
3. 每个动作写清“目的、输入、命令、预期输出”。已有脚本先读 `--help`；下文标为“待实现”的入口不能当作现成CLI调用。
4. host问题在host检查，只有需要实机结果时才让用户执行一次完整命令。能合并到入口的检查不拆成多轮手工问答。
5. 修复可解释的路径、大小、JSON字段、schema选择、缓存／输出取回等局部问题后，跑相关检查、提交并推送；不要为了省事忽略失败、改旧receipt或放宽精度阈值。
6. 每次设备失败先保留attempt、PID、日志和锁状态。exit0但缺输出属于取回未完成，不是模型数值失败，也不是成功；禁止盲目重跑同一步导致二次advance。
7. 已有授权范围内的常规可逆修复直接做。遇到需要偏离本计划已描述／冻结的模型语义／ABI／数值策略或不可解释的错误，按第7节保留材料，回找用户与负责规划的模型。
8. 每次结束更新状态文件的证据、实际结论和**一条**下一动作；新助手用第8节模板接续，不重新设计整条路线。

合同版本：`specs/dflash-contract.json`保留旧ABI v2 prefix-recompute合同，不能把它的两输入target接口套在原89输入target上；当前执行里程碑参考`specs/dflash-execution-smoke.json`。P4/P6新增ABI和验证合同须独立版本化，保留旧成功版本，不能通过覆写旧合同来消除冲突。sampling和多模态另立需求，不顺手扩展当前strict-greedy范围。

开发workspace存在时使用其声明的工具和run目录；用户DDK主机当前是`custom_pypto`环境，沿用已配置的`REPO_DIR/AI_RUN_DIR/HDC/SN/BIP`，不要要求它安装开发workspace或新建一套环境。不要改共享DDK／plugin。

## P0. 完成原prefill驱动的端侧自主多轮循环

**状态：passed（执行范围）。** [最新原prefill闭环](board-prefill-first-completion.md)实际完成prefill1／decode8／draft7，生成8个token，compact回收成功。精度、状态数值语义、常驻与性能另行验收。

主机循环 `run_serial_dflash_smoke.py` 仍只算联调。端侧入口新增 `--prefill-feedback` 后，用户已经完成一次真实 prefill 驱动的 P8/G8 请求；最新总墙钟 305.788 秒，prefill 15.802 秒、decode 合计 116.692 秒、draft 合计 137.128 秒，均包含逐次加载／执行／卸载。保留该结果作为常驻改动的对照，不要求重复旧 serial 路线。

本阶段已观察：原prefill实际执行一次，明确其80状态、logits和8层features到decode／draft的身份与字节交接；随后由板端完成提议、串行验证、接受／纠正及多轮生成。prefill／decode是不同模型variant，保留各自SHA，不能按RefData编号或相同shape猜对应关系。功能接入应在P0推进；P3保留同输入的数值／历史一致性验收。

当前实现`PREFILL64_PADDED_SHORT_PROMPT_EXPERIMENTAL`：8个有效prompt token右padding到原prefill64，使用显式控制配方和80状态映射；首个anchor在位置8推进decode后，完整9行历史进入原S64 draft。用户已回传该路径的compact实机完成摘要；复现命令见[current-status.md](current-status.md)，合同细节见[board-prefill-integration.md](board-prefill-integration.md)。原私有算子的短有效长度语义仍待数值验证。不裁真实历史、不重置位置，P＋G不得超过64，当前不需S128编译。

通过条件：

- 实际运行原prefill，并将同一次请求的80状态、logits／features接入后续decode和完整六层draft。
- 完成至少两轮真实draft及串行verify／generation，保留实际调用数、token、接受／纠正／bonus和模型variant身份；不能用原serial请求的16次target计数替代。
- 查表、prefill、提议、验证、接受／拒绝、状态／特征反馈及循环终止全部在板端；服务器不提供逐token数据或决策。
- 回收同一已完成attempt，保留完成进程及锁证据。compact报告只验tokens和完成状态；逐状态／数值诊断走`--dump`的对应证据合同。
- 物理断连实测当前未提供，单独保留状态；不把设计上的detached能力写成已观察到的断连结果。

验收允许零接受率或乱码，但不能因此取消原prefill或缩减模型。常驻适配可独立并行，不等于P0已经完成，也不能以先完成性能工作为由推迟这条功能路径。

## P1. 建立可信的普通target基线并定位精度问题

**局部预算：1–3天；底层算子返工另加3–7天以上。依赖：P0执行链可用。**

先做普通decode，暂不让draft影响生成。**当前没有`--ordinary-only`命令，需在板端调度器新增普通target模式**：复用其target调用与80状态反馈，读相同prompt和原始embedding行，从相同80零态开始，每次提交当前target argmax，保存token、logits、8层features、80state和控制值。主机参考比较可以独立运行，端侧生成不得退回主机逐token调度。第一版仍用decode1逐token读prompt，避免把跨导出映射混进数值诊断。

接着建立原量化PyTorch／原工程的同输入reference runner。现`original_target_source.py`是源码扩展／准备模块，不是已存在的数值reference运行器；旧`decode_data/golden_logits.bin`历史／控制不同，不能直接当P0新请求黄金。

在原权重机器和DDK之间对两张原embedding表做一次完整SHA对照并保留来源；后续每token仍只查必要行，不反复整表hash。默认lookup的行验证不等于整表来源已经认证。

按顺序定位：原INT8 embedding与独立FP32scale → 控制／绝对位置／mask → 首个block → 中间block → 8taps／最终norm → 完整logits → Top1。同时对比30个conv、30个recurrent、20个attention K/V状态。遇到首个偏差，缩小到该边界或该算子，避免反复全模型编译。

通过条件：

- reference与OMC输入的token、量化／scale、初始状态、位置、mask和参数身份一致；控制配方由实现或实验证据支持，不再只叫experimental。
- 普通target可多轮稳定生成；重复输入结果稳定，状态有正确前史，首个显著差异已定位并修复。
- numerical contract保存比较项和固定阈值。仓库tiny probe的`atol/rtol`不是完整40层容差；完整hidden误差阈值当前未定义，需按原reference建立并评审，不能从失败样本反推放宽阈值。
- 保留两个结论：①同一量化target的普通／DFlash调度一致性；②量化／FP16部署相对可信原模型的质量。①不能替代②。

产物：普通运行driver、reference runner、冻结fixture与输入hash、逐层／80状态对比报告、固定数值验收合同。若全模型不能在开发机运行，可在原权重机器生成小型reference输入／输出包返回，不复制整模型到磁盘不足的DDK主机。

## P2. 用真实且已校准的target特征收敛draft精度

**局部预算：0.5–2天。依赖：P1有可信features；参考准备可与P1并行。**

用同一prefix、anchor、mask、位置和15候选的固定block，对照原完整六层PyTorch draft与旧S64 OMC；保留Top1、hidden及必要的RMS／QK／softmax／PV／残差探针。已有fixture 15/15是一个样本，不替代真实features集合。

先查输入／anchor对齐和精度边界，再查计算。保留原71权重、6层、5层sliding＋1层full attention及原8taps。原BF16训练checkpoint与当前部署FP16/mixed policy差异单独报告，不能悄悄换策略。

通过条件：冻结测试集的draft数值与Top1达到已评审的合同，多组真实features均通过；接受率作为效果观察，零接受率触发诊断但不能靠改target判断规则“提高”。真实分支覆盖与普通target逐token一致性继续保留。

产物：真实feature fixture集合、reference／device对比、误差首发层、固定精度策略与版本。没有明确输入或算子证据，不重编整个draft。

## P3. 验证原prefill64→decode1的数值与历史一致性

**局部预算：0.5–2天。依赖：P1；可与P2部分并行。**

原prefill的功能接入已列为P0的必需内容，不能继续作为后续可选优化。本阶段在其基础上补同一完整请求的数值／历史一致性验收。可分别验证prefill64→若干decode1与draft容量，但完整P0仍须将prefill、draft／serialverify和多轮生成接在一起；不能仅凭单步桥接宣布完成。

现成工具：`prepare_target_prefill_decode_feedback.py`，参数见其`--help`。它生成80条参数owner→source→compiled FX→native→prefill文件→decode输入映射；**不是已完成的80输入执行器，`feedback_ready=false`也不是精度失败**。cross-model source的RefData编号或相同shape不能代替身份。

P0 已实现根据该映射交接实际状态、从本次 prefill logits 查原 embedding，并绑定两个模型 variant 的不同 SHA；本阶段继续验证其数值／历史一致性，不能伪装为同一个 OMC SHA。

核对output_pos的真实最后行、完整prompt embedding、全部80state、newKV／position／allQLen／conv／effective_length／mask控制。把prefill64→decode结果与**同一个完整64-token请求**的逐token普通baseline比较。

通过条件：80映射唯一且实际执行字节链完整；同prompt首步及多步logits／Top1／features／80state满足固定合同，普通生成一致。已有row63试验只保留作定位依据，不自动当正确seed。

产物：prefill+decode双variant session adapter、真实桥接输入／输出、逐tokenbaseline对照、64→1多步状态报告。

## P4. 支持长上下文：增量draft缓存与明确容量

**局部预算：1–3天。依赖：P2，正确target features；P3与此阶段可在host侧并行。这里才引入新draft编译。**

旧S64只能接受完整prefix≤64。跨过容量前，不能截断历史、重置RoPE、篡改manifest尺寸或循环旧features。S128全前缀可作独立对照方案，**不是DFlash算法必需步骤**，不与增量路线同时开两条大编译主线。

已实现：`ge_incremental_draft.py`，C16／C64新feature块、4个INT32控制（start_position／valid_rows／anchor／proposal_count）、12个FP16 K/V输入；包含hidden时14输出（Top1、hidden、12缓存）。主机结构与独立完整前缀oracle已测，尚未导出／编译／上板。

需要依次完成：

1. 冻结实际35B ABI、完整权重与舍入边界、logical history容量和physical buffer容量，记录越界处理。C16／C64是每次新输入行数，不是总历史容量。
2. 写原prepared权重只读引用的增量AIR导出与专用native ABI；不让旧“只留Top1／hidden”的direct-output改写删掉12缓存输出。
3. 先只编译一个gear，预算磁盘／内存后运行；不要默认重拷2.8GB权重或19GB target权重。新17输入含多个同shape标量／缓存，必须按实际native名字和typed view绑定。
4. 实机对比：多次append超过一个gear、跨64边界、valid_rows=0、首尾cache边界、K=1／短K／15、拒绝后追赶、未写前后缀逐位不变；prompt64之后标准anchor@64不重复计入context。

关键不变量：当前anchor之前的新已提交feature才进入draft缓存；anchor／MASK临时KV不能持久化；历史K已经旋转，不能再次RoPE。当前eager target状态包含anchor feature，但draft context必须排除最新anchor。旧S64始终生成15候选后只验证前K，与新图把inactive候选mask掉不是同一个数值oracle，末层双向attention尤其如此。

通过条件：新AIR／OMC、实际17/14 ABI、缓存读写与位置、完整draft数值、端到端tokens均过对应合同；旧S64成功版本保留可回滚。全新算子或模型语义变更需先明确方案并重新规划。

## P5. 在板端自主循环基础上建立模型常驻

**局部预算：0.5–2天。完整验收依赖正确普通baseline；按用户最新要求，接口适配与主机检查提前到P0期间并行。**

具体生命周期和实机待验项见[常驻执行方案](resident-runtime-plan.md)。全V1 path-only候选被所选HCL拒绝后，用户明确允许prefill／decode V2与embedded draft V1混合，优先推进常驻。旧请求只按原配置恢复；既有三个OMC及权重均保留。

P0已经要求端侧自主完整循环；旧主机每步HDC路径仅保留为联调。此阶段优化板端逐次子进程加载和文件交接，不能把这些开销计入DFlash算法收益。先分解模型load／Process／板端状态文件IO／调度／一次部署与最终回收成本。

新请求使用显式 `--resident`、`execution_mode=resident-mixed`：同一线程执行V2 prefill一次并关闭，再各加载一个V2 decode与V1 draft实例，交替Run／Process。循环结束关闭两个实例，无逐token主机交互或模型子进程。V2明确选择manager-only关闭的新实验策略，不独立调用DeinitWeights，标记外权重释放尚未独立验证；DeInit异常仍失败。用户已回传该完整板端请求成功，详见[混合常驻首次结果](board-resident-mixed-first-completion.md)；其数值等价与内存稳定性继续验收。

首次常驻版本保留 80 状态私有文件／输出副本、完整 prefix 特征与原 S64 draft；不引入新的 draft KV ABI，也不宣称零拷贝。之后再根据测量优化内存 bank 和文件 I/O。常规只回收 token／小摘要，`--dump` 保留详细证据。加载／Run／关闭时延分开，RSS 不等于专用 NPU 内存。

单次两模型同时驻留已执行成功；继续实测重复请求稳定性。target外部权重约18.8GB、draft OMC约2.85GB是文件大小，不能当作实际峰值RAM证明；需测实际加载、峰值内存、缓存bank和OS余量。如果同时驻留失败，先保留证据重新规划，不静默反复卸载重载后宣称已常驻。

通过条件：相同输入与结果合同不变，模型只按session加载，state/feature在设备侧正确持有，debug与常规模式token一致；实测峰值内存和重复轮次稳定，无设备reset／泄漏。

产物：常驻native runner/session接口、内存与load/transfer分解、普通／DFlash同runner基线、debugdump开关与回滚路径。

## P6. 批量verify及80状态正确提交／回滚

**局部预算：2–5天。依赖：P1/P2数值正确；有效DFlash接受率；P4长上下文和P5常驻具备可用边界。**

decode1串行verify仍对每个已提交token运行target，主要是正确性／执行基线，不能预设它会加速。真正批量verify需要一个可取回**每个候选位置logits／features**的target路径。原图只输出一个output_pos的logits，不能把它当多位置验证器。

先实现并用原量化PyTorch验证小K的verify：从相同parent state验证整个候选block，输出逐位置判断和可提交状态。再导出／编译明确的verify gear；不要用prefill64的最终80state冒充任意接受前缀的状态。

沿当前eager-anchor约定，pre-round target状态已经包含anchor。verify实际输入是候选p0..p(K−1)：p0比较pre-round logits，pi（i>0）比较verify输出row(i−1)；全接受后bonus取最后一行。若只接受a个后拒绝，则恢复pre-round状态、重放这a个已接受token、再推进correction。下一轮draft只同步旧anchor＋接受的a行，排除最新correction／bonus，不能把anchor算两遍。

回滚对象是完整80状态：30conv、30recurrent、20attention K/V。KV可裁剪或按区间写回不代表GDR／conv能恢复。先选正确方案：有精确前缀checkpoint时选择；没有时先回到旧bank并按已接受tokens串行重放，之后再优化。重放方案先允许慢，但不能算已达到最终加速目标。

必须验证每个r=0..K的接受边界、全拒绝／全接受／中间拒绝、EOS／尾预算，以及拒绝后至少再执行两个token：tokens、logits／features、全部80state与普通baseline对照。冻结active/inactive bank所有权、复制／DMA完成点、cache水位、最后correction／bonus的anchor位置。不得只改缓存长度或只看当前Top1。

通过条件：全部分支覆盖，普通／DFlash token和stop reason零不匹配，拒绝后后续状态也满足合同；真实verify原始输出与状态提交选择可追溯，编译／native ABI无歧义。

产物：verify reference、导出／OMC／native ABI、状态checkpoint或正确重放实现、r=0..K对照报告、多轮拒绝后续接报告。

## P7. 全模型稳定性与配对性能，逐项优化到目标

**局部预算：1–4天，复杂算子优化可更长。依赖：待测最终组合精度通过。**

先冻结模型／代码／输入／设备／固件／runtime／功耗和温度范围，再测。沿用合同：每种模式3次warmup，10次保留测量；10/10数值／token与stop gate通过。普通baseline与DFlash使用相同target、prompt、token预算、停止策略、cache和计时范围；不能把DFlash常驻与普通每步load/HDC对比。

至少报告：cold load、prefill、decode-loop、prefill+decode、tokens/s、draft/verify/state/transfer/host耗时、接受率、每轮接受token数、target/draft真实调用次数、中位数与分布。目标指标为普通完整decode-loop中位延迟除以DFlash同范围中位延迟≥2.0，同时报告完整请求是否也加速。

若不达标：先按实测比例定位load／host搬运／verify／draft head／CPUCL等热点，再一次只改一个有证据的因素。hardware_reference_policy保持use，所需profile必须匹配实际目标；CAModel ticks不能当设备延迟。自定义算子优化在单独边界工作项进行，回到全模型重新验证。改变近似公式、权重转换、精度或图／ABI边界，先提交具体方案重新规划，不能用放宽阈值换速度。

每个候选保留旧baseline，重新关闭精度并做相同10次实机测量。若2倍暂不可达，如实记录实测速率和瓶颈，保持性能阶段未完成，不用理论估计填通过。

## P8. 打包、复现与交付

**局部预算：0.5–1天。依赖：最终精度／稳定性／性能通过，或用户明确接受有记录的阶段性交付。**

阶段性交付必须单独标明scope，不把未通过的P7或最终P8改成passed。

冻结代码commit、模型与权重／量化策略hash、source/compile/ABI/native版本、环境与实际设备身份。打包一键运行入口、样例tokenizer输入、对照／benchmark命令、完整manifest/checksums、精度和10次性能报告、已知限制与回滚。内部库／权重不提交GitHub，生成包留在声明的run/release存储位置。

在一个新输出目录按文档完整复现，不能依赖未记录的交互式步骤。最终区分“完整模型执行通过”“精度通过”“常驻／长上下文通过”“速度达标”“阶段性限制”，不合并成一个模糊PASS。

## 5. 可并行和必须串行的工作

- P0运行期间：可并行准备reference fixture、读源码、检查已有ABI、写普通driver；不并发载入大模型、不争抢板端锁、不同时启动多个大编译。
- P1普通数值是关键路径；P2 reference准备和P3静态映射可并行，但不能拿错误target features关闭draft精度。
- P4增量draft与P5常驻runner可由不同助手负责，先冻结接口和状态所有权。能力不足的单助手按P4→P5顺序做，不同时重构多个公共模块。
- P6每个候选位的state对照、最终普通／spec全流程和P7配对性能按冻结版本串行验收。

## 6. 常见故障的最短处理路径

| 现象 | 先做什么 | 不要做什么／升级条件 |
|---|---|---|
| `Already up to date`但没有新脚本 | 查看当前repo、branch和commit，核对本计划基线及main | 不猜另一个目录存在；先纠正checkout路径 |
| embedding大小不符 | 核对复制完成，weight=508559360B、scale=993280B | 不把部分文件当精度失败，不复制途中覆盖 |
| input_ids与执行embedding不一致 | 用真实表查行，核对独立scale、tokenizer和历史 | 不靠输入文件名／历史golden推断当前token |
| schema／protobuf wire mismatch | 按export receipt绑定原TorchAir ge.proto，compiled OMC用自身schema | 不用同一schema硬解两种图，不从编号猜状态 |
| stat／ctime或mode变化但SHA相同 | 保留旧receipt，用已有限定角色的完整SHA/current身份重绑定流程 | 不全局忽略stat、递归chmod777或改旧manifest“过关” |
| HDC `Device not founded`／missing marker | 恢复选定server连接，保留失败call log和锁owner | 不认HDC进程returncode0即远端成功，不重启别人的任务 |
| model exit0但取回缺文件 | 保留同一次attempt，只恢复读取／取回缺少产物 | 新serial step尚无现成专用resume；旧continuation recovery不能硬套，需小范围适配 |
| lock retained／BUSY | 确认owner、实际PID与已知退出、相关attempt | 不猜锁过期就删；身份不明回找规划 |
| S64容量越界 | 检查真实prefix，先收小新请求预算，或进入P4 | 不裁历史／重置RoPE／改metadata尺寸 |
| finite但饱和到65504、Top1异常、接受率0 | P0可记录执行；P1/P2定位输入或首个数值偏差 | 不叫精度通过，不改accept规则制造接受率 |
| HIAI Load／Process失败 | 保留首条实际错误、stdout/stderr/PID/ABI/库身份，缩成最小复现 | 不盲切ROMC、不把原target放进draft legacy32位buffer |
| prefill/decode模型SHA不同 | 按P3建立两个artifact variant与跨模型80owner映射 | 不填相同SHA，不按RefData序号或shape代替映射 |
| 速度差／没有2倍 | 先测同runner下的真实分解和接受率 | 不对比不同load／HDC范围，不把host测试或单算子时间当网络加速 |

## 7. 哪些情况下回来重新规划

常规路径／参数／局部接口修复可继续做；下列情况停在最近成功版本，保留证据，回找用户与负责规划的模型：

- 同输入仍有无法解释的全零、饱和、首次错误层或buffer依赖；需要改私有GDR／量化／attention kernel。
- 需要改权重、层数、词表、近似公式、精度边界或冻结阈值。
- source／compiled／native任一身份只能靠编号／相同shape猜，或两个不同输出共享无法证明的来源。
- 需要切换原target加载API、改变runtime／compiler／共享DDK、解决同时驻留OOM。
- P6的verify／checkpoint／bank设计不能通过reference，或需要改用本计划之外的协议。正常P6工作先交付设计和reference，再实现设备版本。
- 同一故障经过两个有依据的修复仍失败，或在没有新增证据时准备进行第三次相似重跑。
- 某阶段超过其预算上界，或实测数据说明2倍目标在当前方案下明显不可达，需要调整路线。

回报材料只要一包：当前commit、阶段ID、准确命令、输入／model／ABI hash、最新成功点、第一失败点、最早有效错误与必要日志、PID／锁状态、已尝试修改与结果、待决定的问题。不得仅贴最后100行重复warning或直接写“跑不了”。

## 8. 交给下一位助手的提示词与结束模板

可直接复制给后续模型：

> 请先读取 docs/dflash-delivery-plan.md、specs/dflash-execution-state.json 和 docs/current-status.md。P0原prefill→draft／串行verify→多轮generation已由用户实机完成。P5单次混合接口常驻（prefill/decode V2，draft V1）已实机完成；当前先定位首decode第二token差异，再完成P1/P2精度与P5重复稳定性验收，不要求重复已经成功的逐次加载请求。不要自动指定S128，也不要裁历史或以串行prompt替代原prefill。按计划完成常规修复、验证、提交／推送和远端命令交付，保持旧成功产物。禁止用host测试或有限值冒称设备精度／加速通过。每次结束更新阶段状态、证据和一条下一动作；遇到计划第7节情况，保留最小复现，向用户汇报以便重新规划。

每次交接更新状态JSON，并用以下格式简报：

```text
代码版本：
当前阶段：
已完成及证据路径／hash：
未通过／尚未执行：
模型／输入／ABI是否改变：
实际设备调用与结果：
失败step／PID／锁状态（若有）：
本轮改动与相关检查：
下一条准确命令或待实现动作：
剩余时间预估是否改变及原因：
是否触发重新规划：
```

文档中的未来阶段不是已实现功能；只有产物和真实证据满足该阶段条件后，才能把对应status从pending／in_progress改为passed。

## 文字入口补充（2026-10-31）

原mixed resident快速模式已由用户运行完成。新增板端原tokenizer文字编码／解码已通过主机对照与fake HIAI全循环，实机待运行。执行[文字版命令](board-text-input-output.md)，同时收集七段draft加载计时；不要另编S128、draft或target模型。保持S64预算、单条user聊天模板和固定生成长度的当前边界。后续产品化单独完善EOS早停、多轮会话及更长上下文；数值差异和接受率仍按P1/P2优先收敛。


### 最新加载定位与下一条命令（2026-10-31）

PID17495完整循环成功回收，draft启动206.640秒中SDK Load占206.630秒、major fault128,122。开启顺序预读的单变量候选已实现；见[执行命令](board-resident-load-timing.md)。主机验证与设备效果分别记录，默认关闭。不要更换接口、重编模型或把Load时间转移到预读后当作加速；先比较总启动。板端文字接口已实现、实机结果待回传；后续P1精度、P4批量verify等阶段顺序保持原计划。
