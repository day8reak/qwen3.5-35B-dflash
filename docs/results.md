# 当前结果

当前开发集中推进PyTorch Draft，手写GE／AddCustom优化按用户要求暂缓。现有公共GE产物继续作为已接通的运行基线，PyTorch尚未替换它。

## 最新PyTorch独立执行：加载成功，数值失败

2026-11-02，用户回传尝试`ec3f7c3136ac43fcb927c25eef63f4e3`的完整比较摘要：`EXECUTED_OUTPUTS_RETRIEVED`、`NUMERICAL_FAILED`，任务完成且锁释放。两组各19次调用全部完成，说明本次已经跨过此前的Fill加载失败；38例均未通过原比较门槛。以下为用户回传，未独立读取原始板端张量，随后离线诊断报告OMC SHA256为 ce11666f9298ec7453bbdcdfe1e3a3a38d396cd12a0bd13f9faf6c82c4981ad6。

| 指标 | 参考KV输入 isolated | 板端KV反馈 feedback |
|---|---:|---:|
| 有效Top1一致数 | 0/86 | 0/86 |
| hidden相对L2范围 | 1.0～1.40758 | 1.0～1.40752 |
| 新增KV相对L2范围 | 0.004417～0.008284 | 0.004417～0.008284 |
| KV未写区失败 | 0 | 0 |
| Proposal：8次同步API均值 | 727.875 ms | 727.062 ms |
| Catchup：11次同步API均值 | 734.727 ms | 735.919 ms |

首轮空context的15个Top1全为0，hidden相对L2=1，最大绝对误差4.44921875；后续所有proposal输出均为314，更换anchor也未改变Top1。后续字节诊断确认两组首轮30720个hidden元素全零。context=18时anchor改为32／34／38，共同有效位置仍逐位相同，参考明显改变。15个位置仍有15个不同向量，context变化也影响hidden，不能称为全局固定输出。新增context KV独立于候选Attention计算，其误差较小并不能证明候选分支正确。原逐元素阈值仍使isolated的13例、feedback的18例KV失败；反馈中的两次相同输入重复输出逐位一致。

本轮为不同上下文及proposal数量的诊断调用，不是同输入预热稳态基准。约727ms远未达到50ms目标；与旧GE的profiling记录不构成配对性能对照。先定位候选embedding／临时block KV／Attention／hidden输出异常，保留数值失败与生产未提升状态。

新增`validate_incremental_draft.py --diagnose`读取已回收快照：检查四个控制量的文件hash及值，统计有效hidden的零行、逐位唯一行数、范数、跨调用及两序列对应行差异；可选prepared源权重检查mask／anchor embedding行。它先复算原比较，不改验收规则或结果文件，不调用模型。13项主机合成检查通过，覆盖全零、重复行、同Top1但不同hidden、不同物理控制槽、无效尾行、NaN／Inf、有符号零、文件篡改／截断拒绝及CLI只读行为。源码对照Target的NZ排布及端口未发现直接解释此错误的差异；用户已回传张量统计；控制文件hash和值通过，物理槽start／valid／proposal_count／anchor分别为0／1／2／3。源anchor与mask 248077的embedding均有限且无零元素，源表hash为 ab1a0ddd89ba1cb895b58259f78c18080420595215b02234ee5e84996cf31e28。这些检查不证明设备输入或Gather输出。

首次诊断因`KeyError: mask_token_id`中断：诊断器误读prepared配置的顶层字段，实际位于`draft_config.dflash_config.mask_token_id`。已复用模型的`Qwen35DFlashConfig.from_dict`解析器；原合成夹具的扁平配置改为仓库真实嵌套结构，16项检查通过，包括原检查、实际配置读取、嵌套字段优先和非法token拒绝。该修复只涉及离线诊断，复用原结果重跑同一命令即可；板端数值失败仍待定位。

2026-11-02，用户回传独立embedding探针编译及板端验证通过（未独立读取原始文件）。probe_draft_embedding.py从绑定的PyTorch AIR提取原完整表、axis及Gather，用显式INT32 ID切入，输出embedding和ID回读；原表未转换。DDK为100.520.020.010，编译为一个NPUCL子图。一次Load、3次预热、10次调用的两项输出均逐字节匹配、重复稳定，max_abs及relative L2均为0。同步API均值0.460ms，范围0.365～0.783ms，快照写盘在计时外；这是独立查表耗时。

| 身份 | SHA256或尝试路径 |
|---|---|
| 原始AIR | e976adece11fc82e0911fb82552cdf6fa9115ffc2c53e3221e4df05a1af53d95 |
| 查表探针OMC | 859820964436e3d4a63d2e40315954069ac04c7d20250219f0823d2cd720d794 |
| 板端尝试 | /data/local/tmp/qwen35_vector_dfb692ce94004b739c9207692d78188a |

此结果验证独立原表查表，未验证上游ID构造、缩放、RMS及完整图分区。下一步新增`--stage input-rms`，保留原anchor→mask拼接→Gather→缩放→第一层RmsNorm连接，输出实际ID、raw embedding、FP32 RMS。前两项零容差逐字节比较，RMS沿用完整Draft的0.01／0.01逐元素门槛；原gamma直接使用，不加1。主机合成图、独立参考与PyTorch公式对照、重排物理输出槽、不同类型比较及权重篡改拒绝检查通过；新探针实际编译及板端结果待回传。23项旧查表探针检查回归通过。完整Draft仍数值失败、未提升生产。执行命令见[当前阶段](current-status.md)。

输入RMS探针首次执行被源图检查拦截。2026-11-03用户回传详情：Cast_7输入Muls，输入描述1（FP32）、输出描述2（FP16），GE dst_type=1（FP16）；提取器此处却要求GE类型0（FP32）。这确认报错发生在RMS的浮点输入链，撤回此前索引链的推断。问题是提取器把最终RMS输入类型错误地施加到每个中间Cast，不能据此指认完整Draft的数值根因。

修复按顺序核对FP16源embedding／gamma到RMS的浮点链，保留原Cast及Muls字节，支持缩放前FP32提升及缩放后FP16舍入。最终RMS必须为FP32；缺少缩放后的FP16边界、整数Cast、明确的dtype／shape冲突及循环仍拒绝。原参考公式已有对应FP16舍入，未改参考权重或0.01／0.01阈值。33项主机检查通过，包含用户回传形式的合成图、节点字节保留、非单位缩放与PyTorch参考对照、重排输出槽及拒绝路径；23项旧查表检查回归通过。上游[TorchAir乘法converter](https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/python/torchair/_ge_concrete_graph/ge_converter/aten/mul.py)对FP16张量与标量生成Muls后再转换到输出类型；对应raw API未填写Muls输出dtype，因此补测无shape／无FX元信息的默认描述解析。这是已核对的上游模式，不证明用户安装版本或实际内部运算精度。随后用户已完成下述原前缀执行。

2026-11-03，用户回传原前缀 `pytorch_input_rms` 的板端结果：执行及回收成功，精度失败，尝试为 `/data/local/tmp/qwen35_vector_5c3026ffb81643c5a54ae12e6a4f868b`。一次Load、3次预热、10次调用，ID与raw embedding均逐字节匹配；RMS结果最大绝对误差27.2183837890625、最大相对L2=1，数值有限且十次输出逐位稳定。分图数量为CPUCL=1、NPUCL=1；同步API均值0.8178ms，范围0.552～1.118ms。以上来自用户摘要，尚未独立读取原张量，不能单凭L2=1断言RMS输出全零，分图数量也不能证明哪个算子出错。

随后用户回传同一尝试的离线分析，确认input_rms的32768／32768元素、16／16行全为有符号零；第0次快照行统计与十次逐位稳定共同支持该结论。参考RMS无零元素，行L2为44.20071～44.96104；raw embedding的16行及ID均逐字节正确。探针OMC SHA256为 `f37526e2c41736451db97570f3c4724e568fdd45fa8765124d2195df28ae5880`。源图scale=1，embedding路径为Muls→Cast_7(FP16)→Cast_11(FP32)，gamma路径为Cast_12(FP32)；Muls／Cast_7部分描述是未填写类型的默认值，路径解析不能证明编译后的物理类型。输出样本的正负零与输入符号一致是后续诊断线索，尚不能证明gamma为零或RMS内核错误。继续使用下述x／gamma探针。

工具支持 `validate_draft_vector.py --analyze` 只读分析现有三输出结果，并提供 `input-rms-taps`：原前缀不变，额外输出RMS的x和gamma逻辑边，仍比较原RMS输出。x的附加展平支路使五个输出的shape／dtype组合互异；按编译物理槽重排参考文件，禁止按输出次序猜角色。x／gamma要求逐字节匹配独立参考，RMS原0.01／0.01门槛保持。回读值的RMS公式仅作诊断，不参与验收；新增输出可能改变分图／融合，不是内核输入物理回读证据。14项主机合成及mock检查通过，另有33项原前缀与23项查表回归通过；新增探针真实编译及板端结果待回传。完整Draft仍数值失败，未提升生产。

用户随后回传 `pytorch_input_rms_taps` 编译成功：DDK 100.520.020.010，探针AIR SHA256为 `e2a9d3309be32a59a7fd9b2b9890cc09a1199836799106ddf2b1c49642b870a2`，打包OMC为 `160c9bcd31dc80c26aa2f33e633973469c30a660f4065875f2e5a5302bdea704`、1017155097字节。原embedding、gamma和mask侧文件hash保持。随后验证在prepare阶段报 `compiled output descriptor changed or is ambiguous`，`device_execution=NOT_RUN`、无已知自有锁；编译摘要中的board_attempt指向原完整Draft来源，不能算本探针的新板端尝试。

新增 `probe_draft_embedding.py --inspect-compiled` 只读现有OMC，报告全部公共输出槽的实际描述、连线、producer诊断信息及严格绑定结果。报错也补充槽号、实际shape／dtype和预期列表。尚未获取用户的实际编译输出接口，未放宽匹配或实施重排修复。12项主机合成检查通过，覆盖FP16/FP32变化、形状改变、输出重复匹配、缺失端口、未知enum、只读CLI与模型身份检查；14项taps、33项前缀、23项查表回归通过。完整Draft数值失败状态保持。

用户进一步回传同一OMC的接口检查：embedding、ID、RMS结果和x四项全部匹配；第4号槽gamma的FP32 dtype保留，但实际shape为空、input reference为空、记录size为32。`_meta`仍有 `[2048]` 字符串，不能据此补造接口或将32字节当成8192字节的向量。该现象确认原常量Cast直接输出方案未形成有效公共输出；未证明RMS内核所用gamma为零。

修订 `source-first-rms-input-taps-v2` 保留原RMS及其输入字节，额外增加FP32运行时全1因子、gamma乘该因子的Mul支路及因子回读。两输入六输出的角色仍通过互异shape／dtype组合绑定；显式空来源现在即使shape匹配也会拒绝。参考因子自动生成为1，gamma乘积和因子回读均要求逐字节匹配；若回读不是1，辅助RMS公式跳过，原验收失败状态保持。该观察支路可能改变编译分图，不是内核物理输入回读。15项探针／分析与13项接口主机检查通过，另有33项前缀、23项查表回归；新修订真实编译和执行尚未验证。

2026-11-03，用户回传六输出v2探针执行及离线分析：尝试 `/data/local/tmp/qwen35_vector_cdfbec2849b74e03abac5b749b5c7309`，OMC SHA256 `1995dc87eb59bdb4f548cfe03f2374745ed48c61c38c5c02c477ce3b20be5794`。执行与归档exitcode均为0，锁释放；3次预热、10次输出逐位稳定，但独立数值比较失败。ID、raw embedding、gamma乘积、全1因子回读均逐字节匹配；gamma的2048项均为1。rms_x和input_rms各32768元素均为有符号零，raw embedding无零元素。input_rms最大绝对误差27.2183837890625；rms_x相对L2=1。以回读x和gamma计算得到的零RMS与实际一致，仅为诊断一致性，不能替代独立参考。CPUCL=1、NPUCL=1；同步API均值0.8468ms，范围0.721～1.032ms，不是完整Draft性能。

观察把优先检查范围缩至Gather后Muls→Cast_7→Cast_11及其编译／观测路径，仍不能单独认定Muls或某个Cast的执行错误。源码对照Target CPro内置embedding分支为直接查表再转FP16，Draft额外乘标量1；保留为候选差异，尚未删除节点或更改RMS实现。新增 `probe_draft_embedding.py --inspect-prefix` 只读当前主图、CPUCL任务与哈希核验的保留IR，NPUCL任务不套用CPU GraphDef解码。14项主机合成检查及13项既有接口回归通过；具体编译属性待用户回传，完整Draft数值失败状态保持。以上设备结果均为用户回传，未在本地主机独立读取原张量。

用户随后回传同一OMC的只读编译检查：探针AIR SHA256为 `01cc6b789e756f765e901475979dadae6b90f3c64aa73a171dcbde922e60abe7`，模型1017175435字节。CPUCL任务仅含anchor改形和ID拼接；`SubGraph_1`的五项浮点输出均为FP16，其中RMS／x／gamma乘积／因子回读通过主图Cast转回FP32。运行时因子也先转FP16。该证据只证明这些接口降精度，不能证明内部指令或将全部零值归因于FP16。选定DDK schema库hash为 `542ff32272088008ca95cb13c9647cd06857f84a08c2fa7c908172b1b0f1f975`；编译报告无保留IR（`NO_DUMPS`），NPU任务保持未解码。

新增独立 `input-rms-no-scale` 对照，从同一源AIR提取六输出探针，仅旁路直接Gather后的FP16 Muls(value严格为1)，并将唯一后续Cast输入改接Gather；Cast描述／属性、RMS及其他节点字节保持。源scale和参考scale均须严格为1，有限FP16包含有符号零时数学等价；NaN payload不在该等价声明内。原参考、侧文件及观测ABI保持。13项主机改写检查通过，其中遍历63488种有限FP16位模式；23项探针／模拟执行及原参考门槛检查、13项接口和14项只读检查通过。对照实际编译和设备结果待用户执行；若恢复，只支持原标量路径或相关编译变换的因果嫌疑，不能单凭对照认定独立Muls实现或完整Draft均已修复。

用户回传单位乘法旁路对照通过：attempt `qwen35_vector_d8d60bcd95ec4148959e36063f1ae3fc`，OMC SHA256 `1231035460562818b02e105ea426d3af385bc6ec0c6a277c50eb28430f216e4f`，源AIR仍为 `e976adece11fc82e0911fb82552cdf6fa9115ffc2c53e3221e4df05a1af53d95`。删除Muls后，Cast_7改线payload hash为 `8575c925048e336c57de692a3b047a67fca17af3bec823df926030eeee104eda`。10次embedding／ID／rms_x／gamma观测／因子均逐字节一致，RMS按原0.01／0.01门槛通过；全部结果重复逐位稳定。RMS整体相对L2为0.0007162242，anchor行0.0023033992，其余mask行0.0004270585，最大绝对误差0.0621337891。API均值0.845ms、范围0.647–1.086ms，3次预热和10次测量，非完整Draft性能。

该A/B结果支持旁路单位标量路径用于完整模型，尚不能区分Muls内核、属性解析或相关编译变换。本轮将同一改写纳入完整PyTorch编译准备v5，保持源AIR、参考、权重和外部ABI；完整模型真实复编／验证待用户执行。主机完整图／编译预检16项、原改写13项、探针执行与分析23项通过。按用户要求，已完成探针整组删除，不保存本地模型或结果副本；新增限四组、核验退出与绑定的清理入口，11项主机临时目录检查通过。此处只保留用户回传的结论；远端删除是否完成以清理命令输出为准。

整体流程记录为用户回传的`fused/result`，板端PID34211。C16／cap192／K15增量Draft，18个prompt token、生成8个，混合接口常驻、dump关闭、顺序预读开启。完整流程接通；后续独立精度与profiling见下文。以下未独立读取板端原始产物。

## 增量Draft独立验证

随后用户回传独立测试及38组比较明细：同步调用完成并已回收，锁已释放。原逐元素规则下`numerical_validation=FAILED`，首个失败为`isolated / 0 / empty-context`的hidden超差。

| 数值指标 | 当前观察 |
|---|---|
| 有效Top1：参考KV输入 | 84/86一致；case6与case17各一处不同 |
| 有效Top1：板端KV连续反馈 | 86/86一致 |
| hidden相对L2 | 约1.09%～1.59% |
| 连续反馈KV相对L2 | 约0.44%～0.54%，随上下文增长基本稳定 |
| KV未写区 | 所有回传项`preserved=true` |
| feedback零新增行重复 | 回传误差指标一致；逐字节重复性由离线分析检查 |

L2口径为`norm(actual-reference)/norm(reference)`。现有KV指标来自整块缓存，isolated的正确历史行会稀释新增行误差；后续分析单独统计新增区。上述误差尚未定位到具体算子，不能直接认定只是FP16舍入。合成特征上的Top1一致率不等于实际DFlash接受率。

用户接受约1%的误差作为当前阶段的工程基线，继续采集性能数据；原始比较结果和阈值保留。Draft候选由Target验证，参考Top1偶有不同不直接等于最终输出错误，但可能影响接受率。真实特征及更多样本、Target修复后的接受率仍待验证。

| 输入方式 | 补齐次数／均值 | Proposal次数／均值 |
|---|---:|---:|
| 每次输入参考KV | 11／548.396 ms | 8／548.296 ms |
| 连续复用板端KV | 11／539.176 ms | 8／543.051 ms |

这是不同上下文长度下的诊断API墙钟，包含首次调用，不是同输入稳态基准。当前静态图中`proposal_count=0`只改变掩码，仍保留候选计算与词表head；所以补齐并非只执行KV投影。本轮proposal profiling见下节，补齐需另行测量。精度失败状态保留，未放宽比较阈值。

## 推理时延

最新用户回传增量C16的算子表、同轮模型CSV及厂商工具日志，执行成功。模型CSV的`inference end`记录518.990ms，确认整模型推理仍约0.52秒。

| 增量Draft热点 | CSV设备 | 记录耗时 |
|---|---|---:|
| `__ge/v23/full_dot_conv/output(Convolution)`，入口特征投影 | npu_aicore | 70.968 ms |
| `__ge/draft_logits/flat_output(MatMul)`，完整词表head | npu_aicore | 19.911 ms |
| `draft_top1(ArgMaxExt2)` | cpu | 13.883 ms |
| 用户表中算子耗时合计 | CPU与NPU混合 | 142.440 ms |
| 模型CSV `inference end` | 整模型推理事件 | **518.990 ms** |

Conv2D占算子合计约49.8%，其时长相当于整模型推理的13.7%。518.990与142.440的算术差为376.550ms；由于算子计时覆盖范围与重叠关系未核实，这不是已测得的“CPU开销”或可直接分配的时间桶。此前同步`Process`为539～608ms，与本轮整模型记录量级一致，但输入、上下文和首次调用条件不同，不能当成配对差值。

源码的API计时仅包住同步`client.Process()`，外部输入拷贝、输出文件、tokenizer与KV提交在其外。同轮模型CSV另列加载558.829ms、卸载56.097ms；工具的V1加载日志为606.042ms，计时范围不同，均不并入518.990ms推理。日志`time: 1`未标注耗时单位，不作为时延读数。

当前精度策略让FP32 RMS、score/Softmax与残差加法落在CPU，FP16矩阵乘落在NPU。多次跨设备分段的调度、同步及运行时内部搬运是待测原因，未将全部差额归给CPU。表中同一stream的Conv2D与其输出Reshape起点仅差2μs，而Conv记录70.968ms；起点与duration尚不能组成可信设备执行时间线。

用户进一步回传编译报告：**27个CPUCL、26个NPUCL子图，200个精度检查节点全部匹配**。这表示旧精度策略的编译检查通过，不是设备数值通过；节点列表只覆盖策略要求的节点。每层可见Q/K RMS、score／Softmax、两组残差加RMS四类CPU计算区域，矩阵乘位于NPU，解释了反复分段的结构来源。静态子图数量不等于实测切换次数，尚未量出各次切换耗时。

FP32不必然只能运行在CPU。[AscendC RmsNorm接口文档](https://asc.gitcode.com/api/SIMD-API/adv_api/normalization/RmsNorm.html)列出Kirin X90与float类型支持，但当前公开接口不证明用户安装的DDK已为GE节点提供相同实现。此前`AdnRmsNorm`的infershape失败不能直接用于判定另一算子`RmsNorm`。NPU归属、内部精度、设备数值与耗时分开验证。

用户回传5个FP32小图全部编译成功，编译归属如下：

| 小图 | NPUCL | CPUCL |
|---|---:|---:|
| RMS原语链，hidden形状 | 2 | 1 |
| 内置RmsNorm，hidden形状 | 1 | 0 |
| 内置RmsNorm，head形状 | 1 | 0 |
| Softmax FP32 | 1 | 0 |
| 残差Add FP32 | 1 | 0 |

两轮板端结果：原语RMS、Softmax、残差Add均执行完成且输出逐位稳定；内置RmsNorm的hidden和heads形状均在`HIAI Load`返回1，runtime为`800.631.120.010`，未进入推理。打包成功，任务均已结束并释放锁。错误码尚不能确定底层原因，也不能据此判定X90不支持FP32 NPU运算；编译为单个NPUCL不等于可加载或数值通过。

每个小图预热3次后测10次同步`Process`，不包含加载和快照写盘：

| 小图 | 均值／范围(ms) | 相对L2 | 最大绝对误差 |
|---|---|---:|---:|
| RMS原语链 | 1.5722／1.232–1.928 | 88.1225% | 11.5116937160 |
| Softmax | 1.7261／0.675–2.246 | 0.143379% | 0.001953125 |
| 残差Add | 0.8412／0.700–1.341 | 1.336101% | 0.989990234375 |

三者均未通过原小图阈值。Add普通行L2仅0.03065%，但大值抵消行达到708.78%：参考约0.01的位置可输出0或1，不能只看整体平均误差。Softmax大值行的峰值概率为0.998046875，参考为1.0。原语RMS的88%也不能视为约1%的阶段性可接受偏差。这些探针与原完整Draft的编译路径不同，原完整Draft精度结论保留。

用户回传的完整张量分析确认：默认权重批次的Add全部32768个输出逐位匹配`FP32(FP16(FP16(x)+FP16(residual)))`；FP32直接相加匹配0个，仅输出舍入匹配20414个，仅输入舍入匹配16357个。Softmax的106496个输出均能精确表示为FP16。这确认了数值舍入特征，不能单独证明内部指令类型。

随后显式请求`weight_data_type=FP32`：Add仍失败，整体L2为1.336101%，最大绝对误差0.989990234375，与默认权重批次的汇总误差相同；跨编译的完整输出尚未逐位比较。新批次内10次输出逐位稳定，均值0.6651ms，范围0.578–0.817ms，不能据此宣称加速。head RmsNorm仍在HIAI Load返回1，runtime为`800.631.120.010`。该选项对照结束，没有修复数值或加载问题。

显式FP32权重的Add主图已读取：`graphop_data_cast_0/1`将两个float32输入降为float16，`SubGraph_0`为输入输出均float16的NPUCL，`netoutput_cast_0`再将输出转回float32。输入精度损失由此定位到编译产物中的转换，而非仅凭数值推测。尚不能据此确定NPU内部每条指令的类型。没有独立IR dump不影响该主图结论。

编译日志随后确认保存内核为`executor_add_eltwise_half_nd`，大小7936 bytes。普通`Add`在`custom_ascendc_lib`的支持查询被拒绝，不能由此推断另一算子类型`AddCustom`的FP32能力。`AddCustom`与`ReduceSumCpro`的配置缺少`kernelBins`，日志提示只能用于JIT。用户进一步回传AddCustom注册：输入x/y、输出z，ND，dtype为float16／float32／int32，且precision_reduce=true；找到`add_custom.cpp`，但实际JIT编译、FP32边界与板端数值尚未验证。聚合库`libai_npucore_ascendc_kernel.so`及检查位置的bisheng缺失，ccec存在；这些文件观察不是已经确认的失败根因。

所选OMG帮助说明`weight_data_type`仅控制原始FP32权重且未指定`compress_conf`的场景，未列出中间算术保精度开关。Add的两个操作数都是动态输入，没有浮点权重。完整GE Draft保持现有CPU FP32策略；AddCustom独立探针复用了同一输入及参考，现已暂缓。仅在OMG目录的文本搜索零匹配不能排除其他DDK配置。直接删除Cast无法让既有FP16接口的NPU子图接受FP32输入。

随后用户回传`residual_addcustom_fp32`编译日志：OMG目标仍为`kirinx90`，`custom_ascendc_lib`进入AddCustom JIT；`BuildSingleOp`中的两个输入与输出均已选择`float16`。编译进程在`te_update_version()`报错后退出1，未产出成功OMC、未上板。日志同时打印`Ascend310B1`，但版本初始化失败的底层原因尚未确认；早期parser和缺失聚合库消息也不能单独视为终止原因。修复JIT本身不会证明FP32路径可用。用户决定暂缓这条路线，完整GE Draft未修改。

公开[AscendC Add文档](https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/basic_arithmetic/Add.html)列出X90支持`float`运算；[华为算子工程工具文档](https://developer.huawei.com/consumer/cn/doc/doccenter-capabilities/cannkit-creating-operator-project-msopgen)却注明KirinX90暂不支持float类型输入。它们描述的计算API和工程接口不同，当前DDK的自定义FP32输入接线仍待确认，不能将API支持当成可部署证明。

当前图固定计算16行context投影，即使只追加2行，或某轮`valid_rows=0`，掩码也不会删除这段计算。当前脚本默认profiling样本为case2、追加2行；尚未独立读取本轮request确认。上述热点作为PyTorch候选的对照；先完成候选导出、编译及精度验证，再依据新图profiling评估分段、投影表达与有效行计算量，收益以同一输入的整模型墙钟确认。入口使用Conv2D是因为此前选定OMG拒绝K=16384的MatMul，不能直接换回原MatMul就宣称可用。

当前性能目标为**C16／cap192／K15，常驻预热后proposal同步`Process`平均≤50ms**，包含API内部调度、同步和搬运，加载与初始补齐另计。计划相同输入3次预热、10次计时，并报告最大值；该目标尚未达到。当前518.990ms是单轮profiling读数，不作为已完成预热基准。优化顺序为减少CPU/NPU分段、减少context投影无效计算、检查词表head和ArgMax；不以约1%的阶段误差接受替代新图精度复查。

| 阶段 | 调用数 | 同步SDK API平均时延 |
|---|---:|---:|
| Prefill / Run | 1 | 450.087 ms |
| Decode / Run | 8 | 109.617 ms |
| Draft / Process | 8：1补齐＋7proposal | 608.155 ms |

以上是同步API墙钟，不是纯NPU时间，不含模型加载。Draft均值混合了补齐与proposal，本次未分别计时。Decode外层均值204.750ms，Draft外层均值607.750ms；两种计时分别采集，不能用微小差值推断负开销。不能把API倒数当成整个DFlash的输出吞吐。

| 增量KV | 观察值 |
|---|---:|
| 提交／缓存上下文 | 24行 |
| 提交次数 | 8 |
| 提交字节数 | 589,824 B |
| 提交累计时间 | 303 μs |
| 缓存主机中转／hidden回读 | 0 B／0 B |
| 存储方式 | 保留12个SDK输入Tensor，只回填新增行 |

KV提交开销很小，下一步优先定位`Process`内部耗时。此前S64 Draft API均值726.839ms与这次不是严格配对试验，不能直接作为优化收益结论。

## PyTorch此前的编译与加载定位

本节保留此前失败及修复的过程；最新加载和数值状态以上方独立执行结果为准。

用户随后回传独立执行`da070fd03aab4ab2b6cef0011270bb3f`：源图／编译槽位及参考权重绑定完成，19组CPU参考完成，新模型2851073026字节上传完成；packed OMC SHA256为`48d087e60ba505c7c23e7f4478b5c87d1d2fc14b692a1ae06cc4c3088776d83b`。runtime `800.631.120.010`完成ClientInit、运行版本查询、mmap、顺序预读及SetModelBuffer，SDK `MODEL_LOAD`在844976μs后返回1。native退出码1，任务已完成且锁释放，未执行推理／KV比较；844.976ms是失败加载耗时，不是proposal耗时。

随后用户重新回收并提供HiLog。候选会话PID15386、板端日期03-16、07:05:55.287—07:05:56.649，包含`qwen35_draft_accuracy`的初始化链：`Fill_1 / Fill`在CPUCL初始化时报`input1 dtype:1 is not support`，继而SubGraph_0 Prepare失败，最终Load返回1。此前缺少`libhiai_hcl_model_runtime.so`的搜索错误之后，流程仍继续至计算库、内存分配和CPU子图初始化，不能将它单独认定为终止原因。完整快照SHA256为`4f46b3dcd6da8eeb0a8a65ec52b4a55c0f9e2fc5d62fa28978458ad15ddd50c1`；旧任务没有记录native PID，仍保留“共享日志候选会话”的关联限制。运行日志中的数字dtype不直接套用AIR枚举。

源码的Fill来自缓存索引上限和六层注意力mask的负无穷填充。新增`draft-static-preparation-v4`在编译副本中物化静态Fill：只从实际AIR的字面shape／Shape连线推导尺寸，读取标量常量原始字节，保持输出dtype和数值。相同dtype保留标量位模式；FP16／FP32转换只接受精确可表示值，负无穷和有符号零保持，有损转换／未知值／未知shape提前失败。公共AIR、权重、17／14接口及比较门槛保持；复用现有AIR在DDK服务器复编即可。

29项静态Fill主机检查通过，覆盖稀疏描述、Shape(Cast(...))、广播mask、INT32上限、负无穷、负零、NaN原字节保留、空张量、控制依赖拒绝、有损转换拒绝、大小限制、其他节点字节保持及预处理组合；原转置／零常量20项回归和ABI／模拟传输24项回归通过。本地未运行用户完整AIR、OMG或设备。新OMC编译、加载、两序列KV数值验证及50ms目标仍待用户实测，不能由本次Fill失败推断RmsNorm／融合Attention已通过或不支持。执行命令见[当前阶段](current-status.md)。

首版Fill预处理在用户实际AIR的`Shape:0`上提前失败，OMG及权重快照均未开始。用户只读节点检查确认：`Fill`的输出为INT32，输入来自`Shape(dtype=3)`和`Cast_4(dst_type=3)`；`Fill_1`输出FP16，输入来自`Shape_1(dtype=3)`和`Cast_47(dst_type=1)`。这些Shape／Cast及Fill输入的描述均为`dtype=1, shape={}`且无`_meta`，属于该源图的中间占位描述。Shape/Cast属性是[GE API DataType](https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/python/torchair/ge/_ge_graph.py)，不能套用protobuf枚举：GE INT32=3、FP16=1，对应用户schema中的7、2。上游[full_like converter](https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/python/torchair/_ge_concrete_graph/ge_converter/aten/full_like.py)也明确生成Shape和标量Cast，再调用Fill。

已补齐显式属性到描述枚举的独立映射、占位检查及精确标量Cast（包含INT64→INT32的范围检查）。保留字面维度、可证明的静态连线和实际常量字节作为依据，拒绝未知属性、明确类型冲突、整数溢出和有损浮点转换。45项主机检查通过，含原29项与新增16项；夹具使用用户回传的枚举值和节点描述形式，父图及标量数据由源码语义构造，未读取用户完整AIR或运行DDK。该修正仍待同一AIR复编和新OMC加载验证。

2026-11-02，用户回传`PYTORCH_DRAFT_OMC_COMPILED / model_compilation=PASSED`，接口观察为17输入／14输出，CPUCL3、NPUCL2。特征输入FP16 `[1,16,16384]`，四个控制量INT32，12个输入／输出缓存FP16 `[1,8,192,128]`；Top1为INT32 `[1,15]`，hidden为FP16 `[1,15,2048]`。源图语义顺序不能直接作为编译输入槽位：编译Data为`arg*`名称，需源图连线与编译index联合绑定。未独立读取用户OMC，后续Load失败见上文。

新增独立验证入口复用原17／14 native runner，以源AIR控制关系及每个KV输出Select的保留分支确定输入角色，再绑定实际编译Data.index；输出依赖hash绑定的源／编译AIR顺序合同。原准备权重hash与编译输入快照核对，模型按SHA缓存首次上传；候选不修改旧GE生产绑定或精度状态。24项主机合成schema／OMC／模拟传输检查通过，实际尝试在Load失败，尚无两序列KV比较和推理时延。

编译主图仍有跨子图FP32↔FP16转换，包括 `[1,16]`、`[1,16,1]`、`[1,16,64]`，以及词表logits `[1,15,248320]` 的FP16→FP32。53个子图减少为5个是编译结构变化，不是已测加速；也不证明内部FP32运算保留。

以下保留本次编译成功前的定位过程。

最新用户使用`--tbe-diagnosis common/tbe-diagnosis`复编，OMG读取`prepared.air`，24个Reshape简化和7个零常量替换保持。日志已进入内存分配、`BuildModelRes`和任务生成，在`GetDataParam op=Transpose`后出现`inputTensorPtr is NULL`，随后`Permute ProduceTiling failed`／`Permute Transpose compileInfo is null`，导致SubGraph_1及整图编译失败。报错码为1443627009，具体含义未确认。随后源AIR检查确认：`arg15_1`为第0层Q权重，FileConstant形状为`[4096,2048]`，经常量`[1,0]`转置供`MatMul`右输入使用，两个transpose标志均为false。常量字节hash与INT64 `[1,0]`相符；实际改写仍逐节点读取排列字节。TBE审计为`APPLIED`，worker使用本次编译目录的模块副本；这不等于模型编译成功，也不能把空指针归因为权重文件丢失。

新增`draft-static-preparation-v3`，将符合条件的`MatMul(X, Transpose(W))`改为`MatMul(X, W)`并翻转`transpose_x2`。只处理FP16 ND、静态二维FileConstant、字面排列`[1,0]`且所有引用均为MatMul右输入的模式；保持权重文件、FileConstant、算术dtype、其他节点及公共输出。新的消费端描述保留原device/layout/dtype，更新原权重shape并移除失效的调试`_meta`。报告记录每个改写和跳过原因。20项主机检查通过：按用户稀疏描述构造的protobuf夹具、实际Q宽度的FP16输入／FP32主机矩阵乘结果完全一致、已有true标志、共享权重、ModelDef封装、幂等、原字节保留、各类非匹配／公共／控制依赖保护，以及与零常量物化组合。测试不使用用户完整AIR或DDK，实际OMG、设备数值与50ms性能仍待验证。

日志摘要增加这三类已观察到的任务生成错误，优先于早期Cast常量图诊断；5项主机检查通过，覆盖晚出现的Transpose失败、只读报告、Python traceback保留、warning不提升及旧ByPassNode错误保留。该诊断改动未调用OMG或设备。

此前用户复编已确认`draft-static-preparation-v2`生效：删除24个中间Reshape、物化7个ZerosLike，实际OMG读取`prepared.air`。随后六个Attention编译任务失败；完整traceback定位在`parallel_compilation.py`的`_prerun → _update_options → platform_info.te_update_version()`，抛出`RuntimeError: te_update_version() return error.`，worker退出1。该轮证据定位到内核编译主体之前的平台更新失败，不能归因于Attention尺寸或断言算子不支持。日志所列Q、KV、mask形状分别为`[1,256,16,16]`、`[7,64,32,16]`、`[1,1,16,224]`，与源适配器预期一致；并不证明该内核已编译或执行。

该历史失败轮次没有传`--tbe-diagnosis`。随后用户确认旧诊断目录已删除，改用同一DDK和`custom_pypto`恢复到`common/tbe-diagnosis`。最新复编已引用这个目录，并确认实际worker审计为`APPLIED`。现有兼容实现核验候选、对照及选定DDK来源，适配底层更新参数，OMG仍为`kirinx90`。日志摘要独立保留有界Python traceback及worker失败节点，避免早期Cast错误占满摘要；加载、数值和性能仍未验证。

本轮日志诊断的8项主机检查通过，覆盖第41986行才出现的worker异常、交错日志、截断上限、旧节点／pass错误保留，以及真实`--inspect-compile`入口读取模拟失败记录后文件字节不变。测试不调用OMG或设备。

此前用户复编回传：静态零替换停在`air-preparation`的`unsupported ZerosLike dtype/layout: ZerosLike_1`，`model_compilation=NOT_RUN`。只读检查进一步确认：首个索引ZerosLike为INT32→INT32；六个mask ZerosLike均为FP32→FP16，前驱Cast输出FP32；全部layout为ND，输入／输出及前驱shape均为`{}`。此前要求输入／输出dtype相同导致误拒绝；同时代码将空shape当成标量，属于检查器错误，不能依据旧主机夹具通过声称覆盖了实际AIR。

修正为保留已记录的FP16输出零值，允许上述FP32→FP16零张量形式；空中间shape视为未知，从源图常量和算子关系推导，不能补造期望尺寸。真正的标量Const需有一项已知类型的字面数据。预处理移到权重快照／OMG调用前，错误包含具体dtype／layout，`--inspect-air`也展示ZerosLike描述及其前驱。本轮9项主机检查通过：按回传形式构造的空shape和混合dtype夹具覆盖8档容量／候选组合，索引零常量为`[cap]`、mask零常量为`[1,1,K+1,cap+K+1]`，内容、dtype、未改节点字节与幂等均核对；未知shape不冒充标量，明确冲突仍拒绝；编译入口检查覆盖前置失败不复制权重／不调用OMG、快照变化拒绝及旧路线保持。用户DDK复编、设备精度与性能尚未验证。

前一轮用户复编回传：`prepared.air`已生效，24个中间Reshape被删除，OMG越过此前的ByPassNode失败阶段。本次终止在`ZerosLike create OpGraphBuilder fail`／`opGraphBuilder is null`，随后`Op Build4NPU failed. Node: ZerosLike`导致SubGraph_1构建失败。早期Cast常量图诊断、reshape-transpose融合候选的rank检查和FORMAT_RESERVED警告后仍继续编译，不能把它们当成本次已定位的终止点。

新增同一编译副本中的静态零物化：对缓存索引裁剪及六层mask的ZerosLike，从源AIR描述、常量shape／axis和有界形状规则推导输出尺寸，替换为同dtype的Const。FileConstant只读其shape属性，不使用占位输出描述，不读取权重数据；未知形状、dtype／布局冲突、控制输入或超出8MiB常量预算时停止。未修改节点保留原始protobuf字节；源AIR和原始编译快照保持，证据存入`air-preparation.json`的`zeros_like`。PyTorch源码两处同步改为固定零buffer，今后导出不再生成这7个ZerosLike。本轮用户可直接复用已传输AIR，仅在DDK服务器复编。

本轮11项主机检查通过：真实GE schema夹具覆盖ModelDef／GraphDef、稀疏描述、静态形状推导、常量内容、未改节点字节保持、幂等和非法输入拒绝；六层缩小宽度模型覆盖C16／C64、K15／7／3／1、滑窗开关共144步，替换前后的Top1、hidden、全部K/V逐元素完全一致，输入和未写缓存区保持；四档`torch.export`无zeros_like且17／14接口保持，真实宽度mask和NZ调用布局一致。模拟编译验证实际使用prepared.air、哈希／源快照保持、失败保留证据、未知形状不调用编译器，并检查ZerosLike终止诊断优先展示。这些是主机测试，不证明用户OMG、厂商融合内核精度或50ms目标通过。以下为前序实现与观察。

前一轮用户回传：融合版AIR选择83个文件、2807342555字节，OMG目标仍为`kirinx90`。编译已越过原先的Gather校验，最终在`ReshapeFusion: removing Reshape_49`之后出现`ByPassNode PreCheck(node) false`，随后`Pass Run failed`／`AfterInferShape failed to run infershape pass`并退出1。早期Cast_2／Cast_3常量图infershape诊断仍存在，但日志继续执行了大量图推导与reshape融合；不能仅凭最早报错定为根因，也不能把最后打印的Reshape_49当成已证实的非法节点。

此前新增编译副本的静态Reshape链合并：只删去单消费者、没有控制依赖／公共输出引用的中间Reshape，保留末端shape与顺序；INT32／INT64 shape常量可用，零维度、动态shape、无法确定元素数的双`-1`目标、dtype／布局冲突均不合并。不跨Transpose、Cast、Identity。改写用导出包自带schema，逐节点保留未修改的protobuf字节，并重新解码核对全部元数据；公共AIR、FileConstant、权重及17／14接口保持。OMG读取新`prepared.air`，保留原始快照和`air-preparation.json`。最新用户结果确认该简化使本次编译推进到NPU子图构建，整模型尚未编译通过。

本轮31项主机检查通过：24项真实GE schema夹具／数值布局检查覆盖ModelDef和GraphDef、缺失中间shape、节点逆序、长链、共享输出／控制依赖保护、非法shape不合并、四档完整宽度ND/NZ逐元素一致及重复处理幂等；7项模拟编译检查覆盖实际传给编译器的路径、源AIR／快照不变、失败保留改写记录、预处理失败不调用编译器及终止pass诊断优先展示。这些不是实际OMG、数值精度或性能结果。以下为前序实现与观察。

新增六层增量Draft模块，复用71份prepared权重和17输入／14输出的语义合同；使用Target CPro的`npu_rms_norm → RmsNorm`接口。RMS gamma不加1，候选在FP32乘gamma后转FP16，舍入位置不同于旧GE原语链。入口仍是完整dot的4×4 Conv2D表达，避免直接重试已被拒绝的K=16384 MatMul。

主机独立参考使用相同fused RMS公式、FP16 QK/PV和FP32 score/Softmax。缩小hidden等宽度但保留6层、8tap、block16拓扑，K15／7／3／1共37步：有效Top1全部一致、KV最大绝对差0、hidden最大绝对差0.001953125，预先设定atol=rtol=0.002；包含零新增行、连续增长、拒绝换anchor、重复、无效padding和容量边界。`torch.export`捕获32个测试RMS节点、17输入／14输出，无控制张量转Python标量，4组不同控制输入与eager一致。

单独用真实全宽FP16 `[2048,16384]`权重检查4×4投影，相对`F.linear`最大绝对差0.00048828125、L2为2.27e-5。完整模型5项主机检查、AIR／编译接线6项模拟边界检查，以及AddCustom工具7项检查通过。本地测试代码未发布。

随后用户在权重服务器的`qwen3_5_cpro`环境完成真实TorchAir导出，回传`PYTORCH_DRAFT_AIR_EXPORTED`。PyTorch为`2.8.0+cpu`，TorchAir版本未提供；源AIR为C16／cap192／K15，17输入、14输出、32个RmsNorm，schema为`ge_ir.proto / ge.proto`。导出保存80个外置常量文件，日志列出的参数与buffer包括71份模型权重及9份位置、索引、掩码和RoPE常量。deprecated和GMM tiling冲突warning未阻止本次导出完成，不能据此评价后续OMG或板端行为。

此结果是用户回传的AIR导出和源图检查成功；未独立读取服务器产物，也不证明数值精度、编译内部FP32或NPU性能。成功AIR归入公共`draft/air/`，迁移只移动文件且保留prepared源权重和当前OMC。

随后用户在DDK服务器从公共AIR编译PyTorch OMC，选择81个输入文件、2807233674字节，进入OMG后以退出码1失败。命令使用`kirinx90`、外置权重及`use_origin_format=true`。回收日志明确显示`GatherV2_1 / GatherV2D`校验失败：`Input[1] DataType INT64 is wrong`，参数shape为`[248320,2048]`，索引shape为`[1,16]`，对应embedding。源码的`ids.long()`导致token索引升为INT64，KV行索引也有同样的`local.long()`；已一起改为INT32。较早的Cast常量图infershape错误仍需复编观察，未据此推断RmsNorm或硬件不支持。

索引修复通过7项主机检查：保留六层拓扑、缩小宽度的K15／7／3／1连续KV共76步，Top1、hidden及全部K/V与修改前逐元素完全相同；`torch.export`每档32个Gather类调用的indices均为INT32，17／14接口不变；真实词表边界ID取值一致。使用真实schema夹具验证INT64索引在编译前被拒绝，而INT64的axis常量不被误判为indices。实际TorchAir重导出、OMG复编和板端验证仍待用户执行。当前OMC和runtime未替换，50ms目标仍未达到。

本轮继续参照Target CPro，六层接入`adn_fused_infer_attention → AdnFusedInferAttention`，使用原Target的GE端口映射、NZ query与block_size=32的分块KV。保留32个RmsNorm、INT32 Gather及17／14公共ND接口；只在注意力内部生成临时分块缓存，不引入原地CacheUpdate，也不改变Draft gamma和权重。减少了显式GQA复制和QK／Softmax／PV原语，入口Conv2D保持，尚无新性能结果。

本轮主机验证：真实宽度KV布局逐元素检查212992项；六层缩小宽度参考覆盖C16／C64、K15／7／3／1、滑窗开关共280步，CPU模拟Target调用与原语参考的Top1、hidden、全部K/V完全相同，输入未被修改、未写缓存区保留。四档完整宽度`torch.export`捕获6个测试融合节点及32个测试RMS节点，17／14接口、INT32 Gather均保持。原Target的长度端口和`pse_shift → all_seq_lengths_q`映射另通过调用记录检查。测试使用模拟算子，只证明主机布局、mask和接线；真实TorchAir导出、OMG、厂商融合内核精度及≤50ms目标仍待用户实测。

随后用户回传融合版导出失败：TorchAir已完成`82/82`文件保存，脚本在导出后检查`AdnFusedInferAttention`时抛出属性或NZ／mask描述不匹配。只读检查进一步确认：6个融合节点、32个RmsNorm、17输入／14输出、82个FileConstant；24项报错全是六层Q／K／V／mask的中间shape未记录。首层Q/K/V及mask的输入和来源输出shape均为null，FP16输入dtype存在；mask槽4连接`ConcatV2_11:0`，端口名未保留。属性全部匹配，scale浮点序列化差值在既定容差内。block_table消费端为INT32 `[1,7]`，长度端口为INT64 `[1]`。这定位到源AIR检查将缺失信息当成不匹配，并未证明中间实际shape正确或OMG可编译。

修正后，源图检查按Target converter的固定18槽识别输入，并读取可用的名称映射；属性、连接、dtype、已记录shape和INT32 Gather仍校验。缺失shape记为`DEFERRED_TO_OMG_SHAPE_INFERENCE`，源图状态为`PASSED_WITH_DEFERRED_SHAPES`，不从期望合同或FileConstant的占位输出描述补造实际值。新增`--finalize-air`可重新检查保留导出、更新报告并移动到公共AIR，不运行TorchAir、不改AIR／权重字节。OMG、接口绑定及板端精度仍未执行。

28项主机检查通过：真实GE schema序列化夹具复现24项shape缺失与无名端口；明确shape／dtype／属性／连接错误仍拒绝；缺权重、活动锁和不完整接口不能完成导出；目录移动失败恢复原公共AIR，之后可重试，成功移动保持AIR／权重字节和runtime不变。该验证仅覆盖检查与恢复逻辑，未读取用户完整AIR，也未运行OMG或设备。

## 加载与完整请求

| 项目 | 耗时 |
|---|---:|
| Prefill加载 | 10.300 s |
| Prefill加载＋运行＋关闭 | 11.337 s |
| Decode加载（一次） | 9.672 s |
| Draft加载（一次，含预读） | 3.954 s |
| 其中：顺序预读／SDK Load | 1.576 s／2.324 s |
| Decode外层运行合计 | 1.638 s |
| Draft外层运行合计 | 4.862 s |
| Tokenizer加载编码／输出解码 | 89.310 ms／0.063 ms |
| 常驻关闭 | 0.421 s |
| **完整请求墙钟** | **32.569 s** |

计时项存在嵌套，不应全部相加。此前无预读时Draft加载约206.640s，顺序预读后多次观察到3–4s；这一加载改进不等于模型计算加速。

## 精度与完成范围

| 项目 | 当前结果 |
|---|---|
| 接受／考虑候选 | 0／7，接受率0% |
| 纠正／Bonus | 7／0 |
| 生成Token | `[28,18,6,0,28,4,18,4]` |
| 解码文字 | `=3'!=%3%` |
| 增量Draft编译、精度策略与ABI检查 | C16／cap192／K15通过；不是数值验收 |
| 增量Draft完整板端流程 | 17输入／14输出接通，KV提交完成 |
| Draft独立数值精度 | 38次完成；旧逐元素规则未通过，当前作为阶段性诊断基线 |
| Target精度与语言质量 | 调试中，不据此判断Draft精度 |
| 新增量OMC逐算子性能 | 用户回传：Conv70.968ms；同轮整模型518.990ms |
| 相对普通Decode的配对加速 | 未测得 |

## 原S64 Profiling参考

用户回传旧S64工具PID45599的CSV；不是新增量模型的结果。

| 算子 | CSV设备 | 记录耗时 |
|---|---|---:|
| `__ge/v20/full_dot_conv/output(Convolution)` | npu_aicore | 282.430 ms |
| `__ge/draft_logits/flat_output(MatMul)` | npu_aicore | 19.912 ms |
| `draft_top1(ArgMaxExt2)` | cpu | 13.058 ms |

`v20`对应入口特征投影。新C16图沿用Conv2D表达，当前记录70.968ms；两个算子记录约相差4倍，不等于端到端4倍加速。后续以新CSV检查投影、全词表head／Top1与CPU/NPU调度；不把跨stream算子时间直接求和，也不据4B比例承诺50ms。
