# 当前结果

当前开发集中推进PyTorch Draft，手写GE／AddCustom优化按用户要求暂缓。现有公共GE产物继续作为已接通的运行基线，PyTorch尚未替换它。

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

## PyTorch当前开发路线（权重Transpose等价改写待复编）

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
