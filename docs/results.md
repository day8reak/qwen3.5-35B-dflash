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

## PyTorch当前开发路线（AIR已导出）

新增六层增量Draft模块，复用71份prepared权重和17输入／14输出的语义合同；使用Target CPro的`npu_rms_norm → RmsNorm`接口。RMS gamma不加1，候选在FP32乘gamma后转FP16，舍入位置不同于旧GE原语链。入口仍是完整dot的4×4 Conv2D表达，避免直接重试已被拒绝的K=16384 MatMul。

主机独立参考使用相同fused RMS公式、FP16 QK/PV和FP32 score/Softmax。缩小hidden等宽度但保留6层、8tap、block16拓扑，K15／7／3／1共37步：有效Top1全部一致、KV最大绝对差0、hidden最大绝对差0.001953125，预先设定atol=rtol=0.002；包含零新增行、连续增长、拒绝换anchor、重复、无效padding和容量边界。`torch.export`捕获32个测试RMS节点、17输入／14输出，无控制张量转Python标量，4组不同控制输入与eager一致。

单独用真实全宽FP16 `[2048,16384]`权重检查4×4投影，相对`F.linear`最大绝对差0.00048828125、L2为2.27e-5。完整模型5项主机检查、AIR／编译接线6项模拟边界检查，以及AddCustom工具7项检查通过。本地测试代码未发布。

随后用户在权重服务器的`qwen3_5_cpro`环境完成真实TorchAir导出，回传`PYTORCH_DRAFT_AIR_EXPORTED`。PyTorch为`2.8.0+cpu`，TorchAir版本未提供；源AIR为C16／cap192／K15，17输入、14输出、32个RmsNorm，schema为`ge_ir.proto / ge.proto`。导出保存80个外置常量文件，日志列出的参数与buffer包括71份模型权重及9份位置、索引、掩码和RoPE常量。deprecated和GMM tiling冲突warning未阻止本次导出完成，不能据此评价后续OMG或板端行为。

此结果是用户回传的AIR导出和源图检查成功；未独立读取服务器产物，也不证明数值精度、编译内部FP32或NPU性能。成功AIR归入公共`draft/air/`，迁移只移动文件且保留prepared源权重和当前OMC。

随后用户在DDK服务器从公共AIR编译PyTorch OMC，选择81个输入文件、2807233674字节，进入OMG后以退出码1失败。命令使用`kirinx90`、外置权重及`use_origin_format=true`；当前回传仅含`CalledProcessError`，不足以归因于RmsNorm、磁盘空间、ABI或工具链。下一步读取保留的`compilation/report.json`和`compile.log`，无需重编。入口已增加错误摘要和只读`--inspect-compile`，输入快照进度改为带编号的抽样输出。当前OMC和runtime未替换，50ms目标仍未达到。

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
