# 当前结果

2026-11-04，用户回传最后内置Bitcast批次：payload在OMG NPU预构建的tokens_bits失败，当前注册为INT32→INT4，与FP16入口不匹配；offset65500和full两项SKIPPED/NOT_RUN。没有有效新OMC、落点或设备结果，停止扩展内置候选。此结果满足用户“再试一次，不行就写算子”的授权条件。

已建立 `targets/cpro/ops/draft-top1`，固定有限FP16[1,15,248320]→精确INT32[1,15]。主机run `20261104T075532Z-9847-483675`：35项检查通过，七组完整尺寸fixture由NumPy/PyTorch直接ArgMax与独立C++ binary16整数排序交叉验证；70个复制输出仅用于比较器损坏/缺失/填充/伪造golden拒绝测试。只读DDK收集器验证了源码哈希、内部链接去重、外部链接拒绝、预算及不覆盖保护。源码与参考已落地，NPU kernel/tiling/注册尚未实现；当前缺选定内部DDK的接口与构建证据，本地profile也不可用。一次收集命令和bundle见[当前进展](current-status.md)。未改变完整Draft，未运行设备，也未给出性能收益。

2026-11-04，按用户“你再尝试一下？实在不行那我们就写算子”实现最后一轮内置候选：用两个FP16存储字携带INT32原始位，比较／选择树只搬运索引片段，最后GE Bitcast还原INT32。三项固定顺序为全ID双分支位模式、65500偏移的15×485奇数尾归约、完整词表Top1。前置编译或纯NPU检查失败会跳过后续编译；完整图只有全部NPU任务且无未归属主图计算才是候选。设备准备要求三项一起验证并重新检查落点。该路径有意避开旧INT32 Select／Mul／Add及BOOL→INT32 Cast，禁止对载荷字做浮点计算或转换；载荷中的NaN位模式也必须逐字节保留。

主机run `20261104T073410Z-8724-fa9f44`：248320索引位表示往返、全部65536种低半字、三图回放与两图PyTorch直接ArgMax、19种宽度57次并列／首末／奇数尾树回放通过；原38图manifest／发射C++／fixture完全相同。实际比较入口处理三项各10次伪设备快照，最后一次高位索引差1被拒绝；伪设备不能充当真实设备证据。6种模拟编译流程覆盖前置编译失败、CPU落点、未归属主图Cast、小树失败、完整图CPU及三项通过，验证依赖跳过逻辑。公开GE Bitcast原型不等于所选DDK的支持证明；本地没有执行所选DDK、设备或完整Draft。当前命令见[当前进展](current-status.md)，新方案失败后的自定义Top1授权已记录，残差AddCustom继续暂停。

2026-11-04，用户回传12项内置Top1替代批次：10项编译成功、2项树失败，`npu_only_cases`和`npu_only_full_top1_candidates`均为空。4个独立TopK全为CPUCL1；直接完整TopK为NPUCL1／CPUCL1，tiled与split15均为NPUCL2／CPUCL2，CPU任务明确包含`TopKD`，split15的15个局部TopK也全在CPU。Select控制为NPUCL1／CPUCL1，INT32 Select在CPU；GatherV2D和INT32 Mul／Add控制均为CPUCL1。两棵树在NPUCL编译`CastT tree_0_index`时报`kernelId not registered: 196620`；源节点是BOOL→INT32，失败图没有可用落点。来源为用户粘贴，未提供本批OMC哈希；设备和数值均NOT_RUN，完整Draft未改。本批不安排重复运行。

用户随后要求评估是否有必要自写算子。结论：针对纯NPU及精确INT32 ABI，值得做范围固定的自定义Top1可行性原型；首要风险是所选X90 DDK编译／注册／运行链路，而非归约数学。评估覆盖核内分块值／索引归约、同值最小全局索引、INT32输出、完整词表等价基线及整模型回归。主机穷举确认0–248319中单FP16仅7168个索引能精确表示，首个失真2049；双FP16分量可数学精确还原，但纯NPU INT32重建未证明。此结果不授权降精度或修改输出ABI。本轮没有恢复AddCustom、编写自定义内核或执行新设备探针；详细判断见[当前进展](current-status.md)。评估run为20261104T072307Z-8454-abb9a6。

2026-11-04，继续实现内置纯NPU Top1候选批次：4个TopK值／索引原语、3个INT32选择／Gather／合成控制、2个比较选择树、3个完整词表TopK组合，共12例。`--top1-alternatives`一次编译并自动读取选定DDK前端声明、CPU任务和主图边界，单列未归属任务的主图计算；只有图层面NPU任务独占且没有其他计算节点时才列为纯NPU候选，数值与收益另验。

有限FP16输入合同独立记录，原61例NaN合同与完整Draft失败不变。主机通过12图NumPy回放、9个logits图的PyTorch直接ArgMax／选值核对、20种树宽度的奇数尾项／并列／边界检查、12图公共ABI输出重排与错误dtype拒绝、12例各10次伪设备输出比较和最后一次INT32损坏拒绝；伪编译流程验证单例失败后继续、自动报告及无设备执行。原26个探针的manifest、发射C++和fixture逐字节相同。完整词表树273个构图节点、68字节显式常量，不建立全词表索引常量；导出辅助常量另计。主机run为20261104T065646Z-7503-f04a85。该主机阶段未运行所选DDK或设备；后续用户编译结果已记录于本页开头。

2026-11-04，用户回传现有9个OMC的Top1设备验证最终结果：61例全部完成，每例预热3次、测量10次，共610次测量调用；native／archive及全部case退出码均为0，输出重复逐位稳定。原完整合同下45例通过、16例失败，状态保留NUMERICAL_FAILED。已知完成后无本任务持有锁，不再重复启动／回收。证据为用户粘贴，未独立取回服务器文件；9个OMC的SHA256、失败索引及原fixture时延记录于specs/dflash-execution-state.json的top1_builtin_probe.device_semantics.observation。

| 模型／场景 | 精确结果 | 原fixture API均值ms |
|---|---|---:|
| 小NPU ArgMax [1,512] | 11／14；首／末／多NaN失败 | 0.4993 |
| 小NPU ArgMax [15,485] | 11／14；首／末／多NaN失败 | 0.6210 |
| 小NPU ArgMax [15,512] | 11／14；首／末／多NaN失败 | 0.6964 |
| 小NPU ArgMax [485,512] | 11／14；首／末／多NaN失败 | 0.5957 |
| CPU ArgMax [7275,512] | 原有限fixture通过 | 7.4633 |
| 完整tiled | 13／15行通过；NaN两行失败 | 19.8443 |
| 完整tiled_flat | 同上 | 19.5646 |
| 完整tiled_transposed | 同上 | 40.0270 |
| 完整tiled_485 | 同上 | 19.6726 |

4个小NPU模型在混合NaN场景均选择了fixture中的有限最大值位置，表现与忽略NaN一致，不能推广为所有硬件指令的NaN规则。全NaN返回0仅通过该退化场景；非NaN场景（含有限并列、signed zero、Inf及子正规数）均通过。完整组合每次仅row12 nan_across_tiles（期望512，实际0）和row14 last_nan（期望248319，实际0）不符；具体丢失NaN优先级的阶段未隔离。

用户指出正常推理不应有NaN，并要求网上查找其他纯NPU Top1。这里的NaN是人为诊断输入，并非真实Draft异常证据；正常logits预期有限，不能单凭此项排除小NPU原语。通用PyTorch Top1代码有hidden／logits有限性守卫，但当前设备增量图的真实logits有限性覆盖尚未建立。原golden、整数精确字节门槛和失败报告保持，不宣称完整合同或完整Draft通过。4个完整组合仍有全量CPU局部归约；其19.56–40.03ms为同步API墙钟，CPU对照7.4633ms只做7275个局部归约，两者不是等价功能基准，也不能直接扣减完整模型103.720ms。

调研收敛为分块TopK(k=1)值／索引联合输出、比较／选择树，以及需自定义接入的Ascend C联合归约三类。官方[TopK](https://asc.gitcode.com/api/SIMD-API/adv_api/sort_operations/TopK.html)与[ReduceMax](https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/reduction_compute/ReduceMax.html)均列出X90支持；这是内核API证据，不是选定CPro GE算子注册／落点保证。候选数学结构、优先级与限制见[当前进展](current-status.md)。本轮仅记录已有结果和调研，没有新编译／设备执行、没有改变完整模型或恢复AddCustom。记录run为20261104T064450Z-6893-0209ce。

2026-11-04，用户回传485a0d7对应的21项Top1矩阵摘要：全部COMPILED／PASSED，自动placement_inspection全部INSPECTED，设备与数值仍NOT_RUN。4个小ArgMax图[1,512]、[15,485]、[15,512]、[485,512]均仅NPUCL1且没有CPU操作；4个较大ArgMax图[7275,512]、[7680,485]、[15,485,512]及[512,7275]沿axis0均为CPUCL1，CPU节点明确为ArgMaxExt2。全部9个独立ReduceMax均在CPU，包含最小[1,512]与全宽[15,248320]。

4个完整词表组合（原3D、展平、转置、485宽）均保持NPUCL2／CPUCL2，CPU SubGraph_1仍同时包含局部ArgMaxExt2与ReduceMax，并接收完整3724800元素的FP32输入；逻辑14899200字节不等于测得搬运量。后续小NPU分区输出INT32[15]，末尾CPU完成Gather／INT32索引合成。该批已排除“只改rank、轴方向或交换因子就能移除全量CPU归约”的候选，也表明回退并非仅发生在组合图中；精确选择门槛和通用硬件支持范围未确定。来源为用户粘贴，未独立读取服务器OMC；紧凑摘要未含新OMC／批次哈希，不继承上一批哈希。

下一步复用9个已有OMC：4个小ArgMax各14种输入，1个大CPU ArgMax及4个完整组合沿用原fixture，共61例。小图新增52组输入覆盖首末、并列最小索引、负数、正负零、正负Inf、首／末／多处／全NaN和最小正FP16子正规数。新输入／golden仅写入独立执行目录，先保留原编译recipe、ABI与fixture核对，再绑定同一OMC。整数逐字节比较保持；汇总显示缺失覆盖、失败场景和具体索引，设备计时为每输入预热3次后10次同步API，不能直接相加推断完整Draft收益。

检查执行链发现旧tensor_probe_main.cpp把单维上限固定为65536，会在Load前拒绝完整词表248320。现将单维限制与16 MiB预算对齐，保留原合计IO字节、rank、槽位与dtype检查；build_hiai.py记录支持上限，Python在上传前拒绝旧运行器。需要重建native，OMC及plan V1无需更改。未改HIAI执行、计时或SDK失败后停止后续case的原规则。

主机run `20261104T062203Z-6272-3833d3`：315项检查通过，52组输入与直接PyTorch ArgMax逐字节一致，真实准备／扩展／比较／打包路径处理61例及610次模拟输出；原编译目录字节不变，损坏原fixture、旧运行器和伪设备证据均拒绝，NaN／零／子正规数及大索引差1可定位。实际C++计划解析器配合从源码提取的HiaiTensorBytes接受全部61份计划，拒绝零维、超单维、超乘积／合计预算、超rank和非法dtype6类输入；仅主机编译该解析器，没有链接HIAI或执行设备。完整Draft的103.720ms／NUMERICAL_FAILED不变，当前操作见[当前阶段](current-status.md)。

2026-11-04，用户回传6017594检查器对已编译三个Top1 OMC的结果，status=INSPECTED。direct的ArgMaxExt2在CPU SubGraph_0，输入FP32[1,15,248320]；flat的ArgMaxExt2在CPU SubGraph_1，输入FP32[15,248320]。tiled的CPU SubGraph_1同时包含local_argmax／ArgMaxExt2和local_value／ReduceMax，输入FP32[15,485,512]，输出INT32[7275]与FP32[15,485]；末尾CPU SubGraph_3执行索引Gather、INT32乘加与Reshape。因此局部全量归约仍在CPU，尚未消除完整词表CPU处理。该输入逻辑14899200字节，不是测得的搬运量或耗时。

| 用户回传OMC | 字节数 | SHA256 |
|---|---:|---|
| direct | 7897 | `20a37a9e45348f5164c320fb683ea3ba30574146766b1261e2a14308cccdcb04` |
| flat | 16042 | `3d1482284527c8e269b3a55b82f1c632328a13e991ef2183f42727fc3152762e` |
| tiled | 42159 | `ba5780097d6f31a836c7a64515cd386bd512ea5b12f7208d02df8a98f1835400` |

tiled的NPU SubGraph_2边界确为FP16[15,485]→INT32[15]，内部仍未解码。局部最大值经FP32→FP16转换；输入来自FP16且最大值选择原值，因此仅凭这一Cast不能认定有限值精度损失，NaN与并列行为仍需设备验证。源批次SHA256为`6c1c75b66fe79aa78ac527f3d72be2305622e3dc4ae55aedbd094dc206907ab0`。上述来自用户粘贴，未独立取回服务器文件；schema、逐例报告／主分区／CPU切片哈希见`specs/dflash-execution-state.json`的top1_builtin_probe.compiled_placement_inspection.observation。

按用户“尽量全面，避免多次单点探测”的要求，本轮提供固定21项矩阵：17个独立ArgMaxExt2／ReduceMax形状对照，4个完整词表组合。独立对照覆盖行数、485／512／248320归约长度、相同字节的2D／3D输入、精确转置的axis0输入；组合覆盖原3D控制、局部展平、局部转置、512×485分块。21图一次编译后自动汇总成功例的CPU算子和实际边界，无需第二轮读取命令。原编译result.json保持，独立placement-inspection.json绑定其哈希；失败例的诊断及未验证状态保留。

主机run `20261104T032933Z-4366-5ee867`：450项检查通过，57次语义图回放含40次完整词表组合，直接NumPy与PyTorch oracle一致；210次模拟native输出验证实际ABI准备和比较入口。原六图的manifest、发射C++及fixture逐字节保持。有限值原语覆盖最小正FP16子正规数；ReduceMax输出新增精确字节门槛，注入一个FP32 ULP会失败，原通用浮点容差保持。所有模拟均标注host-fake，未运行所选DDK、设备或新时延测量。完整Draft仍为103.720ms／NUMERICAL_FAILED，当前命令见[当前阶段](current-status.md)。

2026-11-04，用户回传283da71的三个修正后原生Top1探针全部COMPILED／model_compilation=PASSED，device_execution及numerical_validation均NOT_RUN。直接版CPUCL1、无NPU图，边界FP32→INT32；展平版NPUCL1＋CPUCL1，分别FP16→FP16、FP32→INT32；分块版NPUCL2＋CPUCL2，依次NPU FP16→FP16、CPU FP32→[INT32,FP32]、NPU FP16→INT32、CPU [INT32,INT32]→INT32，并出现graphop_data_cast_1的FP32→FP16转换。此处由用户摘要记录，未独立获取OMC哈希或边界shape。

前端缺属性／缺第二输入的编译阻塞已经解决。NPU_GRAPH_PRESENT仅证明存在NPU分区，不证明完整Top1归约或昂贵的局部归约落NPU。分块的接口模式与局部索引和值留CPU、获胜块选择在NPU相符，但必须以CPU节点及边界shape核实，尚不能作为事实或性能收益。未重编完整Draft，也没有放松NaN／并列规则、整数精确比较或启动设备验证。

新增只读--inspect-top1-placement，重用选定schema的OMC／CPU切片读取逻辑，先以保存编译report绑定OMC路径、大小、SHA256、平台与DDK根，检查各任务范围、不重叠、CPU owner及保存的GraphOp数量；输出主图实际描述与连接、各CPU任务的全部节点摘要。数据常量及NPU二进制不展开；不根据源名、缺失CPU节点或FP16接口自动推断NPU内的操作。本地主机run `20261104T032006Z-4074-bb512d` 的70项检查通过，真实schema／容器读取器处理合成三种分区模式，验证完整Draft参考输出不变、只读保证、损坏／错配拒绝和CLI互斥。未调用厂商进程或设备，原探针与fixture未变。下一步命令见[当前阶段](current-status.md)。

2026-11-04，用户回传所选DDK前端头文件及完整Draft的CPU节点，原报告仍为PARTIAL、三个失败case的原dump目录均NO_DUMPS。math_defs.h的HIAI_REG_OP声明明确给出ArgMaxExt2(x, axis)和ReduceMax(x, axes)，轴输入均INT32；数据端口分别允许FLOAT／INT32和FLOAT／INT32／UINT8，未列FLOAT16。array_defs.h声明GatherV2D(x, indices)、INT32 indices及整数axis属性。后端npu_generated_opinfo的一输入／轴属性定义不能继续用作这些前端节点的原型。此次已有声明足以修正探针，无需等失败dump出现。

用户读取的完整Draft OMC为1790918688字节，SHA256 `967e9b640915893890ef95425b0e1279c8245bff0cfc2833f0cbfa56eb023372`。CPU SubGraph_2的ArgMaxV2节点实际类型ArgMaxExt2，数据输入FP32[1,15,248320]，第二输入Const_483为INT32、空维度、4字节，输出INT32[1,15]；属性output_type=3、keep_dims=false、outmaxval=false、topk=1。Const实际数据未解码，不能据此声称读到其轴值；TensorDesc的INT32编号7与属性枚举3属于不同编码。CPUCL任务owner证明这里是CPU示例，描述中的device_type=NPU不能覆盖owner判定。

据此修正三个原生探针：公共输入仍FP16，在图内先扩展到FP32；ArgMaxExt2连接显式INT32标量轴，保留上述四个属性。分块ReduceMax连接INT32向量[2]，输出FP32[15,485]，GatherV2D与全局索引乘加保持INT32。导出守卫检查端口、类型和属性，移除上一版本对三个case的临时整例禁编。直接版只是对照，现有CPU例已说明改算子名不能保证NPU落点；分块的落点、NaN传播和并列规则仍需编译／设备验证。前端检查器也识别完整HIAI_REG_OP／HIAI_OP_END，优先实际op头文件、跳过TIK／AscendC内核目录，避免内核ReduceMax同名片段耗尽前端摘录预算。

主机run `20261104T030633Z-3506-c58c14`：243项检查通过，含六图54次完整形状PyTorch回放、直接NumPy交叉检查、C++端口／属性发射、模拟厂商元数据的真实ABI绑定及十次模拟回收，仍对高位token差1判失败。另46项检查通过：原三图manifest／C++与全部fixture字节保持、原错误单输入图拒绝、错误轴类型／端口／输出枚举拒绝、全部65536种FP16位模式的主机扩展（非NaN逐位往返、NaN分类保持），以及合成HIAI／GE头文件与内核噪声下的只读摘录。没有所选DDK执行，模拟结果不能充当设备证据。完整Draft仍103.720ms／NUMERICAL_FAILED，完整模型和暂停的AddCustom路径均未修改。下一步重编命令见[当前阶段](current-status.md)。

2026-11-04，用户回传8b80eec补output_type后的三个原生Top1探针仍全部编译失败、设备NOT_RUN。直接及展平版本在argmax_ext2、分块版本在winning_tile触发`Input size 1 is wrong, should be 2`，随后出现`Input[1] DataType FLOAT is wrong`。缺失第二输入后出现的dtype诊断不构成FP16 logits不支持的证据；没有已编译落点。上一版把NPU后端opinfo当成完整前端IR原型，接口假设错误；主机数学回放和发射检查未覆盖厂商前端合同。补output_type使校验走到输入合同，不证明完整属性语义已正确。

停止重复编译已知无效表达，新增只读--inspect-top1-ir：从所选DDK通用头文件／前端源码提取声明，按保存DDK路径和编译清单检查失败图，并可读取保留完整Draft的CPU ArgMaxExt2及其直接输入生产者。失败图仅尝试所选DDK的HIAI ModelDef schema；CPU切片按packing schema、owner、范围和不重叠约束读取。厂商库只作字节读取，不import／dlopen／执行，不写入输入目录。头文件、失败图和已编译CPU示例分别标注证据范围；没有猜第二输入端口名、轴值或ReduceMax空shape原因，也没有宣称解决NPU回落。

导出检查改为所选OMG已观察到的二输入数量要求；三个原生case在CLI替换测试槽前拒绝重试，保留失败证据。日志优先展示具体输入数量错误，避免次生dtype报错遮蔽原因。主机run `20261104T024333Z-2592-afd370`：69项检查通过，包括真实只读schema解码器处理合成ELF／IR／CPU OMC、通用头文件摘录、DDK及清单不匹配、损坏／重叠／owner错配的切片拒绝、链接与预算处理、CLI互斥、输入字节未变，以及六图／fixture保留和旧三图发射兼容。测试中的合成proto最初使用无presence的proto3零值，导致任务offset缺字段而被拒绝；改为有presence的合成字段后通过，未放松实际解析约束。未调用所选DDK或设备。完整Draft仍103.720ms／NUMERICAL_FAILED，下一条只读命令见[当前阶段](current-status.md)。

2026-11-04，用户回传三个原生Top1探针全部OMG编译失败、设备NOT_RUN。直接／展平版本在argmax_ext2节点、分块版本在winning_tile节点触发同一具体错误：`math_op_infershapes.cpp ArgMaxExt2Verify(476): AttrName output_type not exis.`。因此本轮没有验证出大轴或分块的实现限制，也没有可用于落点判断的OMC。分块日志显示Reshape到[15,485,512]和local_value／ReduceMax infershape success，随后winning_tile input shape为空；此现象尚未解释，不以“success”断言其shape正确或NPU可用。

探针上一版只使用NPU opinfo中的dimension属性，遗漏OMG IR校验要求的output_type。现为所有原生ArgMaxExt2补整数属性output_type=3（现用GE INT32枚举），不以输出描述代替属性、不使用私有protobuf dtype编号。发射前合同新增属性／输出一致性检查；已有旧图缺属性会在导出前拒绝。日志诊断将AttrName缺失信息置于一般Verify失败和可选kernel库提示之前，保留原始tail与有界读取。该设置的所选OMG接受情况仍待服务器重编，不宣称已解决NPU回落。

主机run `20261104T023316Z-2165-71799e`：233项检查通过，包含六图54次完整词表回放、整数属性C++发射、严格Top1比较和模拟ABI／native输出流程；另32项检查验证本次图差异仅output_type、fixture字节保持、三个旧原生图导出前拒绝、错误属性类型／枚举拒绝、具体缺属性错误位于raw tail之外仍可见、警告不升级和既有节点拒绝诊断保留。没有所选DDK或设备执行，完整Draft的103.720ms及NUMERICAL_FAILED基线不变。修复后重编命令见[当前阶段](current-status.md)。

2026-11-04，用户回传所选内部DDK的只读Top1检查：在`tools/tools_omg/master/lib64/config/npu_generated_opinfo.json`发现ArgMaxExt2（FP16 ND输入、INT32 ND输出、dimension属性、arg_max_d实现）、ReduceMax（axes／keep_dims属性、reduce_max_d实现）及ArgMin注册。TBE源码存在；这仍不是加载该源码、当前形状可用或NPU落点证明。KIRIN_CONF_JSON_PATH为空；部分公共布局目录及AscendC聚合库／bisheng路径不存在，而ccec存在。保留当前内部DDK和kirinx90目标，不据目录布局认定环境错误。报告PARTIAL包含12文件摘录及40候选限额，原检查器将池化／NMS与pyc混入候选，导致归约正文被挤出。

新增三种独立小图：直接ArgMaxExt2、二维[15,248320]归约，以及[15,485,512]局部ArgMaxExt2＋ReduceMax后再选获胜块。后一图使用单输入ReduceMax的axes属性，GatherV2D从展平局部INT32索引中取值，再以INT32乘加还原全局token。旧ArgMaxV2可能已映射同一实现，直接入口用于对照，不保证解决CPU回落。分块版本的NaN传播、并列规则和各段落点需实机核验；不因局部NPU或小逻辑中间量直接承诺时延改善。完整Draft／LM head、数值门槛及暂停的AddCustom路径未改动。

只读检查器改为按源码basename筛选、跳过字节码目录，注册解析完成后优先opFile实现，摘录参数检查正文并显式标注截断。主机run `20261104T022117Z-1605-76c12b` 的230项检查通过：新旧六图共54次完整形状PyTorch回放、NumPy直接argmax交叉检查、原生属性／端口发射、模拟厂商元数据下真实ABI绑定、FP16／INT32 plan、十次模拟回收和高位token差1必败。另22项检查覆盖读取器、噪声文件不耗尽摘录、参数守卫正文、DDK错配及只读保证；旧三图manifest和fixture逐字节保持。此处未调用所选DDK、未上板，新图编译／落点／数值／时延仍待用户执行。下一条命令见[当前阶段](current-status.md)。

2026-11-04，用户回传三种全词表Top1小图的编译结果：PARTIAL、device_execution=NOT_RUN。ArgMaxV2编译通过但仍为CPUCL1／NO_NPU_GRAPH，CPU子图输入FP32、输出INT32。直接ArgMaxWithValue和485×512分块版均在OMG的GetInferFunc／RunInferShape阶段失败，节点分别argmax_value及local_max；后者Reshape推导到[1,15,485,512]已成功。当前只能认定所选编译路径未找到ArgMaxWithValue的infershape函数，不能认定硬件没有最大值归约，也不能归因为词表容量或分块大小。

两份错误日志还包含libai_npucore_ascendc_kernel.so路径无效和TBE内部Ascend310B1字符串，它们未被确定为本次失败根因。既有普通Add成功编译日志也曾出现同类聚合库提示。保留kirinx90目标和共享DDK，不盲目补库、改SOC或重启原AddCustom探针。当前没有可进入实机性能验证的NPU Top1候选；完整Draft及103.720ms／NUMERICAL_FAILED结果不变。

下一步只读本轮omg-environment.json、实际配置路径和归约注册。probe_draft_vector.py新增--inspect-top1，核对所选DDK根，识别配置目录／单个JSON，按有界文件数与字节数查ArgMax／ArgMin／TopK／ReduceMax／ReduceMin，附命名相关源码的有限片段。记录不等于实现、落点或形状支持证明；零匹配同样不是不支持结论。主机run `20261104T021319Z-1312-aa1ef2` 的17项检查通过，覆盖键式／列表式注册、外部配置路径、去重、子链接跳过、DDK错配、畸形／超限文件、源码只读不导入、CLI互斥和输入未变；仅合成文件，未读取用户DDK、未重编或上板。失败分支与下一条只读命令见[当前阶段](current-status.md)。

2026-11-04，用户追加本轮完整op表，合计73656us，与紧邻的model表103720us组成同轮粘贴对照。以下均按duration字段计算，未独立获取原CSV、哈希或回收receipt；与host-embedding OMC的关联来自会话连续性。

| 项目 | 上一双NPU轮 ms | 本轮单NPU／host-embedding ms |
|---|---:|---:|
| model inference | 184.568 | 103.720 |
| 算子duration合计 | 78.986 | 73.656 |
| 两者算术差额 | 105.582 | 30.064 |
| 末尾MatMul_30 | 19.797 | 19.960 |
| CPU ArgMax | 17.576 | 12.353 |
| Conv | 8.685 | 8.643 |
| 两次输入Transpose | 5.616 | 5.615 |

model inference减少80.848ms，算子合计减少5.330ms，算术差额减少75.518ms。当前CPU算子合计13.138ms（其中ArgMax 12.353ms），NPU合计60.518ms。不能把duration合计当关键路径，也不能把30.064ms归为实测拷贝、等待或可全部消除的固定开销。NPU任务start time集中于极短窗口，与各自duration及CPU时间不构成可直接相加的统一时间线；小数部分位数也不固定。本轮load 160.281ms、unload 4.859ms另计；原数值失败及host查表未计入API的限制保留。

下一步独立编译全词表Top1三种内置GE表达：ArgMaxV2对照、ArgMaxWithValue、485×512两级归约，最后GatherElements＋INT32乘加还原全局索引。输入7,449,600字节、最终输出60字节为逻辑尺寸，不是实测跨设备传输量。只输出token索引的图允许编译器正常消除未使用values；不假定换算子名一定落NPU。[固定版本TorchAir schema](https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/python/torchair/_ge_concrete_graph/auto_generated_ge_raw_ops.py)提供端口与属性定义，不构成所选X90 DDK实现支持证明。未修改完整模型、head权重、数值容差或暂停的AddCustom路径。

主机run `20261104T015950Z-753-3ad426`：128项检查通过，包括三种图27次完整形状CPU回放、C++发射、用模拟厂商元数据执行真实ABI绑定、混合FP16／INT32 plan生成、十次模拟回收及高位索引差1必败。15行诊断输入覆盖并列、块边界、大索引、全负数、signed zero、±Inf和NaN；参考为直接NumPy argmax并与原PyTorch交叉检查，独立于候选归约表达。验证器原通用路径的plan dtype从固定FP32改为已核验描述dtype，INT32输出强化为精确字节门槛，原浮点门槛不变。仅主机检查，所选DDK编译及设备执行NOT_RUN。当前只跑小图编译，命令见[当前阶段](current-status.md)。

2026-11-04，用户回传host-embedding候选完整accuracy摘要：EXECUTED_OUTPUTS_RETRIEVED，board_attempt `qwen35_draft_accuracy_2486123ea3e3411390d491ff6db887a0`，锁已释放。host_embedding_binding=RESTORED_FROM_VERIFIED_AIR_AND_PREPARED_WEIGHTS，原FP16表SHA256仍为 `ab1a0ddd89ba1cb895b58259f78c18080420595215b02234ee5e84996cf31e28`。这次已越过来源绑定和MODEL_LOAD并完成推理；对应先前1.79GB／CPUCL2／NPUCL1候选由会话连续性关联，本次粘贴摘要未包含OMC digest，尚未独立读取服务器fixture。后续profile入口按保留fixture核验实际模型。

| 同类同步API测量 | 上一双NPU轮 ms | 本轮 ms | 降幅 |
|---|---:|---:|---:|
| isolated catchup，11次 | 192.067 | 103.958 | 45.9% |
| isolated proposal，8次 | 191.469 | 103.998 | 45.7% |
| feedback catchup，11次 | 200.515 | 106.090 | 47.1% |
| feedback proposal，8次 | 189.147 | 104.994 | 44.5% |
| 38次加权均值 | 193.772 | 104.802 | 45.9% |

按用户报告的均值和次数计算，本轮减少88.970ms、约1.849倍提速；全组最小99.055ms、最大110.569ms。与50ms相比，本组均值还需减少约52.3%，但此处既不是固定输入预热稳态验收，也不是包含查表和提交的端到端时延。单NPU主体与host查表组合已经有用户测得的API收益，尚不能将全部收益归因于某条CPU边界，也不能将旧105.582ms差额直接算作已消除。

NUMERICAL_FAILED保持，isolated／feedback仍各19例全部未通过原严格门槛；Top1为82/86及83/86，7处差异的位置与token和上一轮一致。hidden相对L2范围分别0.013387271239375807～0.025755650596962287、0.014494312718133574～0.026116219967572855，新增KV范围均为0.004417372793104131～0.008283531899103903；其余KV范围和case计数也与旧摘要一致。首例hidden最大绝对误差0.15234375。两模式preservation_failed均0，identical-repeat与full-cache-repeat逐位稳定。没有观察到摘要层面的数值退化，不等同于跨版本所有输出逐位相同，也不构成精度通过。

本地run `20261104T015331Z-570-fc5e19`仅记录用户证据并计算对照，没有重新编译或上板。旧原始报告／数值门槛保留，生产resident仍未接入新ABI。当时安排的profile已有后续用户回传，见本页顶部；不再重复采集。

2026-11-04，host-embedding候选accuracy在reference阶段报candidate and source host embedding bindings differ，device_execution=NOT_RUN、无已知持有锁。用户只读两份manifest后确认：优化策略均为packed-spatial-host-embedding；AIR含原FP16表[248320,2048]、1017118720字节、SHA256 `ab1a0ddd89ba1cb895b58259f78c18080420595215b02234ee5e84996cf31e28`，mask_token_id=248077、scale=1；OMC通过get读取host_embedding为null（未区分字段不存在与显式null）。当前源码编译流程会保留该字段，服务器生成时为何遗漏尚未确定；此报错不是模型加载或数值失败。

验证器现仅对该policy下缺失／null的OMC重复契约延后判断：原编译AIR快照、编译器与打包记录、语义ABI、原权重与embedding表字节、最终OMC哈希全部通过后，从AIR及prepared恢复诊断绑定，写入独立pytorch-abi.json并随fixture／结果摘要保留RESTORED_FROM_VERIFIED_AIR_AND_PREPARED_WEIGHTS。显式冲突、缺失AIR契约、错配原表和损坏文件仍拒绝，原始manifest及OMC不变。现有1.79GB候选无需重编译，本次Python修复无需重建runner；继续重试原accuracy命令。

主机run `20261104T013628Z-32299-efe976`：25项绑定检查通过，覆盖缺失／null恢复、原有相等记录、旧token-ID策略、明确冲突、AIR／编译AIR／OMC／原表损坏、编译DDK错配、修改快照哈希仍不能掩盖错误打包片段、错误block输入语义以及输入产物不变。使用合成protobuf、完整公共ABI与缩小权重，仅物理DDK接口检查为mock。另43项原字节查表／ABI／模拟参考fixture及结果传播检查通过，含prepared-only不访问板端；15项导出／编译流程检查通过并确认当前代码能保留host契约。均为主机证据，没有执行用户OMC；该候选加载、数值和时延仍待设备回传。

2026-11-04，用户回传host-embedding候选的编译OMC边界及依赖检查：模型1790918688字节，SHA256 `967e9b640915893890ef95425b0e1279c8245bff0cfc2833f0cbfa56eb023372`。较上一单NPU相位表OMC的2808047355字节减少1017128667字节，约1.79GB／1.668GiB。这里是完整OMC大小；本轮摘要没有给出各GraphOp权重大小或实际分配参数，不替代加载成功证据。检查工具的model_compilation=NOT_RUN表示本次只读已有编译产物。

CPUCL2／NPUCL1保留，中间CPU分区及输出列表均为空，unresolved=[]。前置CPU由4个标量输入变为3个，输出6条CPU→NPU路径；单NPU主体20输入／14输出属于内部边界，不是公共ABI数量。唯一NPU→CPU路径仍为float16 logits [1,15,248320]经Cast到float32后交给CPU ArgMax；对应逻辑尺寸7449600／14899200字节，Top1输出60字节。这些是描述尺寸，不是实测搬运量。报告绑定的边界JSON为244482字节、SHA256 `8c1b82a99f021b3c5288bee6df7113c6a9e1af96c53b1fc6a97a245cda49e330`；证据来源为用户粘贴，未独立读取服务器文件。

此结构已满足本轮避免中间CPU回流的目标；新OMC尚无设备加载、数值或时延结果，不能把旧184.568−78.986=105.582ms直接当作已消除的开销。下一步复用新OMC，重建支持accuracy plan v2的共享runner，执行原isolated／feedback验证与回收。查表在服务器fixture准备阶段完成，native API不含该开销；仍须保留原数值失败门槛、KV保持和后续端侧输入准备计时。命令已切换到[当前阶段](current-status.md)，无需重新导出或编译模型。

2026-11-04，host-embedding候选首次TorchAir导出停在air-validation。用户确认export.air为1790366439字节、weight/文件数0；尚未编译或上板。公开[TorchAir export_utils.py](https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/python/torchair/_utils/export_utils.py)按图与权重总量超过(2048−200)MiB才选择FileConstant，否则写内嵌Const，与减掉约1GB表后本次观察一致。64MiB错误来自本项目对外置权重AIR元数据的限制，不是NPU容量或数值失败。Const分支保留arg节点名但不写原参数文件名，不能依靠大小或位置猜出后续精度校验所需的权重绑定。

修复在导出CLI进程内临时固定原生外置权重分支；核对实际安装函数签名、转换器调用关系、单次转换、参数文件名无冲突、Data节点及连续非空张量，保留原used_weight_num和原生写盘实现。finally恢复原函数，记录安装源码哈希及各外置文件尺寸，并在导出、导入、恢复和编译入口保持检查。未修改TorchAir安装、数学、权重dtype、原精度门槛或64MiB限制。本次已有内嵌AIR需要重新导出，不再尝试直接拆包恢复。

主机run `20261104T010541Z-31378-5f4c04`：15项导出存储检查通过，执行固定上游版本的原决策／转换／写盘Python函数，GE构造器为stub；覆盖阈值两侧、FP16特殊位模式、FP32／INT32原字节、节点名字、异常恢复、错误接口、文件冲突及大小边界。另15项protobuf和模拟导出／编译／恢复检查通过，包含权重选择记录不能被finalize绕过。此处未安装用户所选TorchAir或DDK，实机导出、编译、加载、精度与时延待用户验证。下一步按[当前阶段](current-status.md)重导出后编译检查单个NPU权重块。

2026-11-04，用户回传显式limit_partition_size=true的结果：OMG编译通过，CPUCL2／NPUCL1，GraphOp权重仍为3892／2806299136／20字节。打包前检查在partition-weight-check拒绝超限候选，未发布packed model、未上板。所选OMG help.log明确true(default): limit by platform，未提供字节阈值；该选项不能解释为强制2GiB拆分。weight_merge描述IR权重合并，SC选项描述缓存用途，尚无证据证明它们能避开本次失败。

下一候选packed-spatial-host-embedding从NPU图去掉完整embedding常量，由调用前原FP16行复制提供[anchor, MASK×K]。Target特征用于context K/V，embedding用于候选分支初始hidden，两者不能互相替代。公共anchor:int32[1]改为block_embeddings:float16[1,K+1,2048]，仍17输入／14输出。K15输入65536字节；原表248320×2048×2=1017118720字节，旧单NPU块减表约1789180416字节，这是估计，不是新OMC观察。LM head完整保留，不量化、不缩词表、不改golden或容差，只允许原embedding scale=1。导出／编译清单绑定host表，独立ABI绑定核验原表哈希和layer0 RMS消费连接。

主机证据run `20261104T004345Z-29867-0cafbe`：六层缩小尺寸、C16／64、K1／3／7／15及滑窗开关的220次连续回放，全部14输出与相位表版逐字节相同，输入只读、未写KV保持。八档完整尺寸meta捕获（测试RMS）保留31线性、1Conv、12attention MatMul、6FP32 Softmax和完整head；参数59→58，不再包含embedding表。22项权重字节／语义ABI绑定、42项行复制／非法输入／完整ABI／模拟参考fixture和结果标注、14项protobuf及mock导出编译、3项保留输入诊断、22项C++实际plan解析／ABI检查通过。一次主机测试误把准备清单传给结果汇总，修正为实际compare返回值后通过，没有改变生产接受逻辑。本地缺少所选DDK／TorchAir，设备加载、数值和性能均未验证。

独立accuracy runner新增显式plan v2，旧v1及生产resident默认token-ID ABI严格保留。新fixture在服务器提前查表，native Process时延不含查表／输入准备，报告显式标注NOT_END_TO_END_BENCHMARK；不能据此宣称50ms端到端达标。编译并确认权重和无中间CPU依赖后，需重建runner再验证。相位表a045的已回收失败及原数值基线保留。当前命令见[当前阶段](current-status.md)。

2026-11-04，用户回传相位表OMC `a045f7629920fd962d21c8ce694b1ee694a3be61942c4cac6a556f00cc631027` 的accuracy执行失败。板端尝试cc7399ff087d40be8e647bcdd2f46a7c已完成回收、exitcode=1、NO_OWNED_LOCK_AFTER_KNOWN_COMPLETION；失败阶段MODEL_LOAD，尚未进入推理或数值比较。native.stderr中CLIENT_INIT、版本查询、MMAP、357485us预读和SetModelBuffer完成，Load在266499us后失败；这些是加载阶段时间，不是推理时延。

HiLog候选PID52903、设备日期03-17、23:54:25.537～23:54:26.236：SubGraph_0／1／2权重分别3892／2806299136／20字节；remote_c_mem_allocator.cpp Copy的memcpy_s失败后，AllocateWeightMemory、InitMemory、ModelManagerInit和Load依次失败。候选完整日志SHA256 `41c2c85dc7b6d33ec074aae7aaac6e3ccf21cca4caa34be2aa0966d888fed634`，共享快照SHA256 `5ffd08a8d89bbd570a6dd90c31597bb130b09d63749324474ee656c26309db5b`。没有记录本次native PID，候选与本次执行关联及实际dstSize／srcSize／memcpy_s返回码仍未证实。早期库搜索错误之后程序继续进入权重初始化，不能单凭第一条error归因。

公开OpenHarmony [securectype.h](https://raw.githubusercontent.com/openharmony/third_party_bounds_checking_function/master/include/securectype.h)定义默认SECUREC_MEM_MAX_LEN=0x7fffffff；[memcpy_s.c](https://raw.githubusercontent.com/openharmony/third_party_bounds_checking_function/master/src/memcpy_s.c)对超限destMax返回ERANGE。合并后的单块2806299136字节（2.614 GiB）超过此值，单块拷贝上限是优先假设；公开源码不证明设备800.631.120.010实际加载库的宏配置，也不证明本次失败是总内存不足。之前的无中间CPU结构结论保留，但该模型目前不可加载。

新增PyTorch编译显式参数--limit-partition-size true／false；默认省略，原AIR、相位表、训练权重和运行API不变。true请求所选OMG限制分区，并在打包前检查每个GraphOp的graphop_weight_size均可读且≤2147483647；未知或超限停止在partition-weight-check，保留编译和接口报告。报告包含每块owner／大小、max_known_bytes及超限名单；这些是编译描述值，不是实际拷贝参数。允许多个NPU分区，下一步仍检查无中间CPU依赖，不强求单NPU块。主机证据run `20261104T003321Z-29312-8cfae7` 的28项protobuf／模拟编译／CLI检查通过，包括默认选项不变、true／false传递、边界值、2.806GB单块阻止打包、总量超2GiB但各块受控、缺失／非法大小拒绝及原静态预处理回归。后续用户已确认true未改变单块大小，见本页最新记录；设备数值及性能仍未验证。

2026-11-04，用户回传相位表候选的编译产物检查：OMC 2808047355字节，SHA256 `a045f7629920fd962d21c8ce694b1ee694a3be61942c4cac6a556f00cc631027`。CPUCL3／NPUCL2变为CPUCL2／NPUCL1，前置CPU控制节点类别计数保持，中间CPU的两个频率Mul不再出现，两个NPU分区合并为一个。新SubGraph_2是末尾ArgMax，并非旧的频率子图；其FP32[1,15,248320]输入逻辑14899200字节、INT32输出60字节保持。NPU主体20输入／14输出是内部分区接口，不代表公共ABI扩为20输入。

依赖分析ANALYZED、unresolved为空、intermediate_cpu_partitions和intermediate_cpu_outputs均为空；CPU→NPU7条，NPU→CPU仅1条，即FP16 logits经insert_cast_0成为FP32后进入末尾CPU ArgMax。边界JSON 255447字节，SHA256 `7e75e9a23c78f7b748f8de2e08111bbc2699a46f155a9390a1cdbfe924f4a51c`。这是用户粘贴的静态元数据，未独立取回文件；accuracy_binding=NOT_PROVIDED、source_numerical_validation=NOT_EVALUATED，尚无此模型的设备数值、同步API或算子时延。结构目标达成，不将旧模型105.582ms差额直接记为收益；下一步使用当前OMC完成19步isolated／feedback设备比较及API统计，原golden和门槛不变。

2026-11-04，用户回传 `packed-spatial-controls` 的只读分区与依赖结果：OMC 2808227325字节，SHA256 `63b94d9f05ec42a9ef8521fcc03fc800cec9b155f011251850aca28f7a9dcb6b`。原BOOL[1,208]的中间NPU→CPU回流消失，mask逻辑转入前置CPU；中间CPU SubGraph_2从6输入／4输出缩为2输入／2输出，逻辑输出221184→8192字节，节点只剩Data1、Const2、Mul2、NetOutput1。两条浮点回流仍在：NPU SubGraph_1端口1／2的FP16[1,16,1]分别经netoutput_cast_0／1进入CPU的FP32输入；CPU产出两个FP32[1,16,64]，经graphop_data_cast_1／3回到NPU SubGraph_3端口3／4。CPUCL3／NPUCL2未合并，末尾CPU ArgMax的FP32[1,15,248320]接口不变。

本轮依赖分析status=ANALYZED、unresolved为空，CPU→NPU11条、NPU→CPU3条（含末尾logits）、NPU→NPU9条。完整边界JSON为306455字节，SHA256 `55c1b8c88d2a2335067532e55b2e7ba05b671f4989fdd78ca29f344c1caf92ac`；accuracy_binding=NOT_PROVIDED、source_numerical_validation=NOT_EVALUATED。这是用户粘贴的编译元数据，未独立取回文件，新模型无设备精度／时延，不继承4d的184.568ms或将逻辑输出减少换算为收益。前两次shell续行是缺少末尾双引号，最后一次依赖命令已成功。

下一候选 `packed-spatial-rope-table` 从运行图移除两次位置乘inv_freq：用原FP32公式预计算相位表，推理时INT32位置查表后沿用原Sin／Cos。表行数为capacity+max(C,K+1)，覆盖无写入满缓存时仍计算的填充context行；C16／cap192／K15为208×64 FP32，53248字节。只预计算相位，没有预计算或替换三角函数；原训练权重、17／14接口、KV区间和数值门槛保持。候选显式选择，旧policy仍可复现。需要编译后的分区及设备验证，不能宣称已消除同步或达到50ms。

证据run `20261103T235522Z-28355-7b79bf`：六层缩小宽度、C16／64、K15／7／3／1及滑窗开关的220次回放，全部14输出与controls版逐字节一致，输入未修改，未写KV保持。实际head_dim128、容量192／512／4096的24组几何覆盖全部表项，FP32相位及FP32／FP16 Sin／Cos结果与原公式逐字节相同；594组边界控制通过，原越界拒绝保持。八档完整尺寸meta图（测试RMS）通过：两次FP32相位表Gather、INT32索引、原2组Sin／Cos，31线性／12attention MatMul／6FP32 Softmax／32RMS及17／14接口保持。18项权重／ABI检查及14项protobuf／模拟导出编译检查通过。本地未运行所选CPro DDK、TorchAir或设备；下一步导出、编译并只读检查是否还存在中间CPU的两次Mul。

2026-11-03，用户在确认“最差hidden相对L2约2.61%、Top1共7/172不一致”后明确回复：**“接受本轮结果，先优化性能”**。当前组合版作为性能优化基线，暂停继续追查数值差异。此决定针对已观测的OMC与用例，原报告的NUMERICAL_FAILED、逐元素门槛及golden保留；后续候选仍须报告相对本基线的数值变化，不能由此次接受推出任意额外近似均已获准。生产未提升。

最新用户回传 `packed-spatial-4d` 的完整accuracy摘要、边界及profile表。边界所绑定OMC为2808242141字节，SHA256 `c7e8007bcceb37141e2d667040ed64cc88afbc532173043deb4ee6ff4719a5b2`；accuracy尝试 `qwen35_draft_accuracy_c2b8c8ed4df542daa4981af458b95903` 已完成并回收，未残留其持有锁。仍为NUMERICAL_FAILED：Top1 isolated82/86、feedback83/86，7处差异的位置/token及hidden／KV误差范围与已接受基线一致；未写KV保持，两组反馈重复逐位相等。摘要一致不证明两版全部输出逐位相同。

| 同类测量 | packed-spatial ms | packed-spatial-4d ms | 下降 |
|---|---:|---:|---:|
| isolated catchup API均值，11次 | 289.482 | 192.067 | 33.7% |
| isolated proposal API均值，8次 | 286.236 | 191.469 | 33.1% |
| feedback catchup API均值，11次 | 279.693 | 200.515 | 28.3% |
| feedback proposal API均值，8次 | 298.002 | 189.147 | 36.5% |
| 单次厂商model inference | 245.989 | 184.568 | 25.0% |

新profile算子汇总78.986ms，与184.568ms推理的算术差额为105.582ms（57.2%）。Load234.684ms、Unload7.560ms不在推理内。末尾MatMul19.797ms、CPU ArgMax17.576ms、Conv8.685ms；特征重排现在是两个NPU Permute，分别5.255ms和0.361ms。NPU model ID为62／63、stream478／479，CPU model ID为1。此处按用户表格记录；该新profile的回收receipt、PID、CSV哈希尚未提供，不继承上一轮的身份信息。单次profile和混合accuracy均值均不是50ms预热稳态验收。

新边界仍为CPUCL3／NPUCL2，但CPU0只接收四个INT32控制量共16字节，已经没有特征Permute；特征FP16[1,16,16384]直接输入NPU。CPU2从14输入／8输出降为6输入／4输出，逻辑输出647168→221184字节；其中BOOL[1,208]输入明确来自NPU1:2，另有两个FP32[1,16,1]输入经netoutput_cast_0／1，尚需追上游。末尾CPU ArgMax仍接收14899200字节FP32 logits。逻辑描述尺寸不代表实际传输量。

用户明确要求优先解决105.582ms差额，故暂缓Top1和MatMul改写。源码复核：native API计时仅包住同步Process，加载、输入拷贝、结果落盘及KV提交都在计时外。中间CPU子图的NPU输入依赖是下一项对照重点，但不能直接把差额归因于这条边界。新表中CPU ArgMax起点12:01:49.365758早于末尾MatMul起点49.366910，后者加时长还超过model结束时间49.383778；因此不以这些start time拼出CPU/NPU统一执行时间线或断言具体空等时长。

新增 `profile_draft.py --analyze-boundaries`，只读取已保存的完整边界JSON，穿过主图Cast／Reshape等显式输入边追踪GraphOp端口。列出NPU→CPU连接及有后续NPU消费者的中间CPU分区；不按src_name／device_type猜执行归属，不推断GraphOp内部输入输出关系。11组主机检查通过，覆盖直接及双Cast回流、动态shape分支、共享生产者、跨设备终端logits、缺失/错位引用、多输出不确定性、环、实际CLI及原证据不变。run `20261103T124343Z-21225-c999d4`，检查为合成编译元数据，不是用户OMC或设备测量。

用户随后回传依赖分析，status=ANALYZED、unresolved为空；输入完整边界JSON为333337字节，SHA256 `3f66adf078481b028ffa38ba76b2ef95c3099247d8dc095afe74ca1deed552a4`，继承上述4d OMC身份。确认SubGraph_1端口12／11的FP16[1,16,1]分别经netoutput_cast_1／0变为FP32，进入CPU SubGraph_2端口3／4；SubGraph_1端口2的BOOL[1,208]进入同一CPU端口5。这个CPU子图向SubGraph_3输出两个FP32[1,16,64]（经Cast成FP16）及两个BOOL[1,8,64,208]。两条位置形状路径与源码的RoPE位置展开／乘频率相符，这是源图对应推断，未解码NPU内部任务。三个NPU生产端逻辑尺寸合计272字节，CPU消费端合计336字节，不能由此给105.582ms差额分配具体拷贝或同步时间。

新增显式候选 `packed-spatial-controls`，仅改两个控制表达：位置先做INT32 Unsqueeze再Cast为FP32；cache／block有效性先各转精确0/1 INT32，再Concat及原有Gather扩展。保持FP32频率乘法、三角函数、完整词表、FP16权重和17／14接口，尝试将这些控制计算放在NPU主体之前。没有强制编译分配，源码改写不能证明中间CPU子图已消失。

主机证据run `20261103T125807Z-21638-b356b9`：六层缩小宽度、C16／C64、K15／7／3／1、滑窗开关的220次连续回放，全部14输出与4d版本逐位相等，输入未修改，未写KV保持。八档实际模型尺寸meta捕获（测试RMS）通过，确认位置INT32视图先于FP32 Cast、有效性不存在BOOL Concat；31线性、12attention MatMul、6FP32 Softmax、32RMS和17／14接口保持。18项权重与ABI检查、14项protobuf夹具／模拟导出编译检查通过。CLI支持新policy。本地无所选CPro DDK／TorchAir，候选未实际编译或上板；需要用户侧编译、完整数值回收、同模型边界检查及相同输入的同步API对照。

用户随后回传 `packed-spatial-controls` 编译成功摘要：C16／cap192／K15、17输入／14输出，优化字段INT32_UNSQUEEZE_BEFORE_FP32_CAST与INT32_BEFORE_QUERY_GATHER均确认。CPUCL3／NPUCL2仍未减少；static-preparation-v5为removed18、fill1、weight_transpose31、unit_scale1。没有新模型SHA、边界明细、数值或时延，compiled_abi=NOT_BOUND、compiled_precision=NOT_VALIDATED、device_execution=NOT_RUN。编译成功不能证明中间回流已移除，分区数量不变也不能证明三条具体连接完全未改。

为先核对结构再决定是否进行整模型板端实验，`profile_draft.py --inspect-boundaries`现允许不附accuracy结果，只读已完成编译的OMC并记录当前hash；仍验证打包记录中的DDK schema、任务范围与CPU图归属，并检查模型在读取过程中未变化。无accuracy时明确标记NOT_PROVIDED／NOT_EVALUATED，不借用旧模型精度；显式提供结果时保留所有模型／夹具／DDK一致性检查。run `20261103T132107Z-22364-d70b19`共32项主机检查通过，包括原18项回归和新模式的实际CLI、证据不变、读取期间模型变化、错配不降级、schema／输出路径保护、下游依赖分析。使用真实IMOD容器读取及JSON schema夹具，未运行所选DDK或设备。下一步检查当前OMC并追踪中间NPU→CPU→NPU连接，旧accuracy不绑定新模型。

用户另询问NPU ArgMax／流水。当前ArgMaxExt2被选到CPU不代表X90没有归约能力：[Ascend C ReduceMax文档](https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/reduction_compute/ReduceMax.html)明确支持X90 half／float、返回最大值索引及并列最小索引。248320词表需按UB约束分块；half块内索引不能超过65535，恢复全局索引应使用INT32，再按最大值优先、相等时最小索引合并。目标是NPU内完成完整词表Top1，只输出15个INT32共60字节，去掉末尾CPU的FP32 logits消费接口；当前14,899,200字节是逻辑尺寸，不是测得搬运量。公开API支持不证明已选DDK的注册、编译或本模型分配可用，新ArgMax尚未实现或测量。先做独立归约探针验证部署、边界值／并列值／NaN行为，再考虑整图接入；不默认TopK(k=1)在并列值时等价。单次推理内真正依赖NPU结果的CPU步骤不能仅靠异步调用掩盖；这些标量控制可前置，跨轮proposal仍依赖Target验证与新anchor，多请求吞吐流水不能替代单请求50ms时延目标。

以下为上一轮记录。2026-11-03，用户回传第一轮 `packed-spatial` 编译后精度与profiling结果，执行及回收成功。随后边界报告绑定优化OMC字节数2808368316、SHA256 `f178fc83a0a9981b501bd5a012e5dee33a54f11f28175dc84dd10625889a6a1e`，CPUCL=3／NPUCL=2；所回传精度摘要与上一版一致：isolated Top1 82/86、feedback 83/86，共7/172个相同位置和token的差异；hidden相对L2上限仍为2.61%，新增KV为0.44%～0.83%，未写KV均保持，两组重复调用逐位稳定。这些摘要不能证明两版全部输出逐位相同。原NUMERICAL_FAILED保留。

| 同类测量 | 旧组合版 ms | packed-spatial ms |
|---|---:|---:|
| isolated catchup API均值，11次 | 736.501 | 289.482 |
| isolated proposal API均值，8次 | 760.223 | 286.236 |
| feedback catchup API均值，11次 | 754.463 | 279.693 |
| feedback proposal API均值，8次 | 773.596 | 298.002 |
| 单次厂商model inference | 627.647 | 245.989 |
| Conv2D单算子 | 69.038 | 8.664 |
| 末尾MatMul单算子 | 19.975 | 19.798 |
| CPU ArgMax单算子 | 16.199 | 17.404 |

本轮profile模型Load237.132ms、Unload6.100ms另计。用户汇总算子约70ms，与模型推理相差约176ms；不是实测的拷贝或等待时间桶。混合输入API均值、单次厂商profile、固定输入预热基准仍须分开。Conv约8倍、完整单次profile约2.55倍的改善来自用户回传，不是本地重新测量，50ms目标未达到。

精度尝试为 `qwen35_draft_accuracy_2039853432e54f378e601bcebd7ee8dd`；profile尝试为 `qwen35_draft_profile_aee65f0c941d43548838915203005627`，工具PID34795，model/parser退出0，原CSV已回收且锁已释放。model CSV SHA256 `afe84bc612fc160afe483a6d97a62db0f952cf9353672dcb89c5de62dc5e3762`；op CSV SHA256 `4387fd276b3efc4d6a3b85b9ffc45a03dba190d950897ce8a23d0e46ae7e5b72`。此处根据用户粘贴摘要及表格记录，未独立读取服务器文件。

用户补充Target推理与算子汇总接近，因此不把Draft差额解释为所有模型普遍具有的统计现象。源码确认原Target厂商工具入口使用V2和外置权重，Draft入口使用V1和内嵌权重；Draft新op表仍有CPU Cast/Gather/Logical、ArgMax及两个NPU model ID。当前Target分区/完整对照CSV尚未提供，不宣称Target单NPU子图。V2对照未见收益，新Draft边界报告见下文，优先检查图内边界。

显式 `profile_draft.py --accuracy ... --modelmanager-itf V2`对照使用同一内嵌OMC、17/14 ABI、参考输入和单次profile/频率/优先级，只改为V2并从文件恢复。10组主机检查通过，证据run `20261103T113618Z-19197-f0ca6f`。用户随后回传模型表：V2 inference258.682ms、Load507.168ms、Unload10.904ms；对照V1的245.989／237.132／6.100ms，单次推理增加12.693ms（约5.2%），Load约2.14倍。没有观察到接口切换收益，默认继续V1。该轮未附op CSV、PID／模型hash和完整输出，不能用旧70ms算子和推算V2自身差额，也不能推广为V2一定更慢；输出正确性未验收，接口假设优先级降低。

同轮用户回传优化OMC边界报告，三个CPU task均解码；以下是编译描述，未独立取回服务器文件，不是实测拷贝或等待：

| 边界 | 当前观察 | 对下一轮的意义 |
|---|---|---|
| SubGraph_0 CPU | 接收FP32[1,16,16384]，1048576字节；含Permute1 | 结合源图确认Conv空间重排五维Permute落CPU，前后有编译器插入的FP16／FP32转换；现有op表该CPU Transpose仅0.404ms，不能据此解释全部差额 |
| SubGraph_1 NPU | Conv输入FP16[1,1024,16,16]，524288字节 | Conv仍采用已提速的完整K空间表达 |
| SubGraph_2 CPU | 14输入共36944字节、8输出共647168字节；Gather7、Cast14、LogicalAnd10、LogicalOr6 | 六层重复mask扩展仍有合并机会；最大输入列表只展示8项，不推断所有依赖或执行次序 |
| SubGraph_1→3 | FP16[8,208,128]，425984字节 | 前版对应32头1703936字节，GQA取消KV复制已反映在此接口 |
| SubGraph_4 CPU | FP32[1,15,248320]，14899200字节→15个INT32共60字节；ArgMaxExt2 | 完整词表Top1仍是CPU消费者，尚未消除 |

新候选 `packed-spatial-4d` 将原五维特征排列拆为两个四维Transpose，保持token到4×4 patch的精确坐标及完整点积；将层间相同的full／sliding visibility放在循环外，保留窗口、因果条件与inactive-query anchor witness。旧policies及manifest契约保持，增加显式候选契约。完整词表head／ArgMax未改，未新增量化或近似。

本轮主机证据run `20261103T115433Z-20069-553cbd`：两档C、四档K、滑窗开关的220次六层连续回放，新候选全部14输出与上一packed-spatial逐位一致；全部三种优化版对baseline共660个对照通过0.01／0.01，634逐位相同，最大绝对差0.005859375，Top1全部相同且未写KV保持。C16／C64的实际16384→2048随机FP16投影与原batch Conv逐位一致。11个完整配置meta图（测试RMS）通过，新候选八档的mask头扩展6→2、INT32 Gather19→15，12attention MatMul、6FP32 Softmax、17／14接口不变。25项权重／ABI绑定和14项protobuf／mock导出编译检查通过。本地缺少所选CPro DDK，尚未验证编译后的NPU放置、真实权重数值或时延；四维Transpose可能仍被改写或落CPU，不能由源图减少节点推断50ms达成。下一步命令见[当前阶段测试](current-status.md)。

第一轮图优化新增显式 `--optimization packed` 与 `packed-spatial`，默认baseline便于保留现有路径。packed把同层context/block拼行后合并K/V投影，合并MLP gate/up输出通道；query折叠到KV头组，避免复制KV；context/block各计算一次RoPE供六层复用。packed-spatial进一步将C16 token排为4×4个空间patch，在[1,1024,16,16]上以stride4执行原[2048,1024,4,4]完整点积权重；C64对应[1,1024,32,32]。没有拆K、量化、缩词表或改KV写入规则。

| 本地主机检查 | 结果与范围 |
|---|---|
| 六层缩小宽度、两档C、四档K、滑窗开关、连续反馈 | 440个优化版对照均通过0.01／0.01；427逐位一致，最大绝对差0.005859375；Top1全部一致，输入只读，未写KV逐位保持 |
| 实际16384→2048特征投影 | C16/C64的空间Conv与batch Conv逐位一致，随机合成FP16输入与权重 |
| 完整配置meta捕获 | 10图；55→31线性投影、sin/cos各18→2、31→19个INT32 Gather（删除12次KV头复制）；12attention MatMul、6FP32 Softmax、32测试RMS、17／14接口不变 |
| 权重组织 | 71→59参数文件、学习参数总字节数2806049792不变；逐段绑定原始prepared，交换K/V、损坏snapshot等拒绝，19项检查通过 |
| 导出／编译控制链 | 14项protobuf夹具与mock工具链检查通过，含packed-spatial元数据透传及可见融合attention拒绝；未调用真实OMG |
| CPU结果审计 | 原版和packed-spatial各14项检查通过，均保留旧门槛、失败状态和golden |

证据run `20261103T105358Z-18207-11e412`。以上为改动时主机检查，后续用户设备反馈见本页顶部；本地没有所选CPro DDK，未访问用户设备。新的17／14语义绑定支持Conv前空间Transpose，合并权重通过原K/V和gate/up各段hash核对。现有native runner、公共AIR/OMC、参考数学和验收门槛未替换。

2026-11-03，用户随后回传 `packed-spatial-4d` 编译成功摘要：C16／cap192／K15，17输入／14输出，CPUCL=3／NPUCL=2。optimization契约含TWO_RANK4_TRANSPOSES和shared_visibility_masks=true；source组合attention六组、无源融合节点，编译主图融合节点列表为空，但GraphOp内部未解码。v5准备removed18、fill1、weight_transpose31、unit_scale1。compiled_abi=NOT_BOUND、compiled_precision=NOT_VALIDATED、device_execution=NOT_RUN；尚未收到新OMC SHA、CPU节点细节或时延。数量仍为5不能证明重排仍在CPU或mask共享无效，也不能证明提速。下一步验证脚本绑定物理ABI并执行完整accuracy，再读取同模型边界和V1 profile，沿用原数值门槛；无需再次导出编译。该条是用户编译摘要，未独立取回产物。

50ms目标按C16/cap192/K15常驻同步API、同输入3次预热加10次测量判断。此前暂以5～7个工作日安排验证窗口，前提每天1～2轮编译／板端反馈；不是达标承诺。第一轮已见提速，但245.989ms与约70ms算子汇总间的差额仍未归因，不据算子接近70ms就缩短达标工期。NPU侧完整词表Top1尚未实现；图内边界候选已由用户编译，设备效果待验证。

当前基线OMC SHA256：`288644ea39913fc49466ed4b6d9329825b6ea819259c266c4601cba391b5e182`，来自用户回传的真实权重CPU审计。源数学在四个选定isolated case中通过原参考门槛，两种gamma舍入边界报告的指标相同；这不能证明所有中间张量逐位相同或编译后算术精度。

| case | CPU hidden相对L2 vs reference | CPU新增KV最大绝对差 | device hidden相对L2 vs CPU | device新增KV最大绝对差 |
|---|---:|---:|---:|---:|
| 0 empty-context | 0 | 无新增 | 0.0144943 | 无新增 |
| 1 initial-catchup | 不比较hidden | 0.015625 | 不比较hidden | 0.171875 |
| 2 partial-context | 0.00185118 | 0.015625 | 0.0158052 | 0.156250 |
| 17 capacity-boundary | 0.00181693 | 0.031250 | 0.0258057 | 0.140625 |

四例CPU Top1均与原参考一致；板端对CPU的Top1差异分别为1／0／0／2。非空context三例的12项新增K/V在设备侧全部未通过原门槛。完整保留结果中每层新增K、V各13例，isolated与feedback的逐层摘要误差相同，数值均有限。CPU最大绝对差大于0.01仍可能通过`atol+rtol*abs(reference)`逐元素门槛。以上缩小了参考表达差异的可能范围，没有确定Conv、RMS或其他具体算子根因，现按用户决定暂缓追查。

用户回传本次厂商profile的两张表，当前编译manifest仍为CPUCL=3／NPUCL=2。以下为粘贴数据，未独立取回原CSV、PID与模型身份文件。

| 计时项 | ms | 范围 |
|---|---:|---|
| Load | 231.187 | 本次新实例加载 |
| Model inference | 627.647 | 本次模型推理 |
| Unload | 7.652 | 本次卸载 |
| 三项合计 | 866.486 | 不能当作单次常驻推理 |
| 记录的CPU＋NPU算子耗时加和 | 134.940 | 不能当作模型墙钟 |
| 推理减算子加和 | 492.707 | 尚未解释的算术差额 |

模型inference begin显示`09:47:25.89418`，按未补齐六位的微秒整数解释为`.089418`，至`.717065`恰为627647us。算子表中CPU ArgMax start早于末尾MatMul start，NPU节点start又高度聚集，因此不能直接从这些start字段构建统一执行时间线。493ms差额不是已测得的CPU、IPC或拷贝耗时，也不应与另一组760ms混合输入均值相减推断开销。

算子热点为Conv2D 69.038ms、MatMul_54 19.975ms、CPU ArgMaxV2 16.199ms，合计105.212ms，占算子加和约78%；末尾MatMul的具体语义尚需图映射确认。六组QK/PV BatchMatMul与Softmax分别加和0.271ms和0.860ms，不含attention的Gather／Transpose等工作。只优化Conv不足以解释或消除整次627.647ms。现有精度runner已一次Load、多次Process，请求HIGH档位；CopyOutput、输出快照及CommitInputSlices均在Process计时外，尚未发现外层重复Load或文件IO混入API计时的证据。

新增`profile_draft.py --inspect-boundaries`，只读hash绑定的当前OMC、夹具manifest和所选DDK schema，输出GraphOp边界描述、明确同图producer及CPU节点清单。逻辑尺寸按端口统计，未知shape不猜测；NPU task不解码，CPU task按范围及归属图名检查，没有匹配OMG JSON交叉验证。该报告不能作为实际搬运量或等待归因。18组主机检查通过，包含真实IMOD容器读取／hash绑定及四项CLI拒绝；protobuf解码用JSON schema stub替代，未验证真实DDK schema或设备。证据run `20261103T102947Z-16488-f050e3`。用户已回传实际DDK读取结果，见下文；当前未改模型图，未声称提速。

2026-11-03，用户回传真实DDK的当前OMC边界检查，状态INSPECTED、模型hash与上述基线一致，三个CPU task均已解码。SubGraph_0／2／4输入逻辑字节分别为16／37968／14899200；中间CPU子图输出712704字节，末尾CPU子图只有ArgMaxExt2，接收FP32[1,15,248320]（约14.21MiB），输出15个INT32，共60字节。SubGraph_3的最大已列输入为来自SubGraph_1的FP16[32,208,128]，逻辑1703936字节。此证据确定了Top1的CPU消费边界，未测量实际拷贝、同步或493ms差额；不能把边界尺寸直接换算为耗时。所有数值来自用户回传JSON，未独立取回服务器文件。

用户要求全面借鉴4B分支，已只读检出 `day8reak/qwen3.5-4B-dflash` 的 `feature/gdr-chunk-verify`，固定提交 `90186732d915e8b058b2a053d205991d805eae76`，对照当前35B代码；没有把外部代码接入生产。参考实现使用Ascend310P／ATC／ACL，当前平台是X90／OMG／HIAI。

| 可借鉴项 | 参考分支实际实现 | 对当前35B的判断 |
|---|---|---|
| 投影合并 | PackedDraftLayer离线合并K/V、gate/up输出通道；DraftGraph先拼context与block行再做一次KV投影 | 每层9次线性投影可改为5次；六层加head为55→31次，不含12次attention BMM与特征Conv。数学相同但M/N改变可能影响FP16舍入，需编译及数值对照，不能按节点减少比例推断加速 |
| GQA不复制KV | query从[B,Hkv,G,Q,D]折成[B,Hkv,G*Q,D]，与原始KV做BMM，再恢复Hq | 当前32/8头可删除4倍KV展开；score元素数与attention计算量不变。X90已有BOOL Broadcast限制，mask表达须保留已验证路径 |
| 公共控制量复用 | context、block的RoPE sin/cos及位置距离在层循环外计算 | 当前18组RoPE可表达为2组并共享，另按full/sliding类型复用可见性mask；后者是基于其组织方式提出的35B扩展，参考代码仍在层内组合mask |
| 按行更新KV | BHSD直接展平为[-1,D]并用唯一行索引ScatterNdUpdate | 值得对照当前整bank Gather/Select，但4B会写物理padding暂存行，当前35B要求[start,end)外逐位保持；零有效行、尾部与容量边界均不能直接照搬 |
| 常驻特征与预绑定KV | ACL内Target输出features与Draft输入共享池，KV两套dataset预绑定，提交交换current/next角色 | 借鉴跨模型缓冲区所有权；35B已有一次Load、多次Process、固定tensor，不能重复计入收益。HIAI是否支持相应外部buffer绑定需验证，不能直接套ACL API |
| C16/C64与末块调度 | 非末尾prompt块C64建立KV，末块由第一次有效proposal消费，后续按有效行数选C16 | 35B已有两档能力；集成时避免额外catchup调用或用C64承载所有小增量。减少物理行数才可能减少计算，仅valid_rows变小不代表省算 |
| 纯context入口 | DraftContextGraph只做FC/norm、各层context K/V与cache更新 | 可用于35B纯catchup，删除无消费者的候选attention/MLP/head；参考生产导出仍使用统一DraftGraph，context-only生产快路在优化文档中仍待开发。增加OM及常驻权重成本需一并测量 |
| 权重与产物复用 | 合并后consume_source删除原参数；按来源/ABI/编译配置校验后复用公共图；W8离线NZ可逐字节逆变换 | 35B可借鉴逐对处理、避免同时保留整套原始与合并权重、仅重编变化的Draft。NZ是特定310P INT8布局，不能给X90 FP16权重只改format标签 |
| 同输入与交错回放 | 冻结实际输入，isolated与interleaved_prefill回放，校验输入只读、KV区域、指针及输出 | 补充当前重复用例，尤其用于检查复用buffer/共享workspace造成的跨模型污染；读回与hash会扰动执行，诊断结果不作性能成绩 |
| 完整生成评估 | 按相同已提交token前缀比较轮次，记录proposed/accepted/fallback；分别统计Prefill、Decode、各图与C档位 | 优化后除了单图ms还需接受率与端到端decode；不能把轮号相同、前缀内容不同的两轮直接比较，也不能只凭较低Draft L2断言加速 |

源码入口：[投影合并、context图与统一Draft图](https://github.com/day8reak/qwen3.5-4B-dflash/blob/90186732d915e8b058b2a053d205991d805eae76/framework/python/qwen35_dflash/ascend310p/incremental.py#L454)、[GQA与实际Top1](https://github.com/day8reak/qwen3.5-4B-dflash/blob/90186732d915e8b058b2a053d205991d805eae76/framework/python/qwen35_dflash/ascend310p/quant_factory.py#L290)、[常驻池、预绑定与回放](https://github.com/day8reak/qwen3.5-4B-dflash/blob/90186732d915e8b058b2a053d205991d805eae76/framework/runtime/cpp/src/acl_chunk.cpp#L502)、[不可变产物复用](https://github.com/day8reak/qwen3.5-4B-dflash/blob/90186732d915e8b058b2a053d205991d805eae76/framework/python/qwen35_dflash/ascend310p/common_reuse.py)、[按相同前缀比较](https://github.com/day8reak/qwen3.5-4B-dflash/blob/90186732d915e8b058b2a053d205991d805eae76/framework/python/qwen35_dflash/ascend310p/compare_rounds.py)。

本地直接提取该固定提交的AirDFlashOps类，使用35B当前Q32／KV8／Q16／KV208／D128、非连续query布局与FP16 QK／FP32 softmax／FP16 PV表达，对照当前显式KV复制路径。两组随机种子×五种mask（全208、仅anchor、部分context、sliding32、inactive witness），10组CPU合成输出全部逐位相同。只验证这些输入下的表达对齐，未测试完整权重、所选DDK编译或设备；参考仓自身也明确GEMM分块变化不承诺FP16逐位一致。证据run `20261103T104208Z-17166-836288`。

不直接借用的结论：参考生产Top1仍是Linear→ArgMax，融合完整词表head+Top1仅为[待开发接口](https://github.com/day8reak/qwen3.5-4B-dflash/blob/90186732d915e8b058b2a053d205991d805eae76/docs/OPTIMIZATION.md#L115)。特征FC使用普通Linear，未提供X90上K=16384的Conv替代方案。[参考实测](https://github.com/day8reak/qwen3.5-4B-dflash/blob/90186732d915e8b058b2a053d205991d805eae76/docs/RESULTS.md#L27)为310P FP16约20.22ms、W8 runtime63.52ms／离线NZ47.89ms、W4 runtime94.68ms；量化checkpoint层数不同，且来自不同预热配置，不能视为同模型纯bitwidth实验或X90速度承诺。

FC宽K对照：4B FP16为8×2560=20480，输出2560；4B W4/W8为5×2560=12800；当前35B为8×2048=16384，输出2048。4B FP16的K反而更大，不能按Target总参数量判断Draft投影宽度。参考源码直接调用Linear，使用310P/ATC；本地此前用户日志明确在matmul_nd_check_support.cc的CheckBatchMatMulCubeDim拒绝K>8000，并绑定draft.fc对应的__ge/v20/flat_output。该限制只属于已观察的所选OMG实现路径，不是X90所有矩阵乘的硬件上限。现有69.038ms Conv是该FC的完整点积等价表达，并非Target GatedDeltaNet时序卷积；4B源码没有同一Conv路径。历史拒绝记录见[完整Draft K限制](history/validation.md#完整-draft-k-限制与完整点积卷积表达)，4B尺寸见[配置](https://github.com/day8reak/qwen3.5-4B-dflash/blob/90186732d915e8b058b2a053d205991d805eae76/models/dflash_v1/dflash_config.py#L211)。

实施顺序：先做保持既有接口/精度边界的投影合并、GQA与RoPE共享候选；保留Conv空间排布与NPU侧完整词表Top1作为现有profile热点实验。随后评估严格valid-row KV更新、纯catchup图和Target→Draft常驻buffer集成。每项分别检查编译放置、相对已接受基线的hidden/Top1/KV变化及完整同步调用时间。该段为实施前评审；第一轮候选现已接入，主机结果与剩余工作见页首。原始prepared权重与旧数值报告保留。

以下为本轮原始数值验证及此前调查记录。

2026-11-03，用户回传组合attention完整Draft结果，attempt `/data/local/tmp/qwen35_draft_accuracy_d47d8f927c5e4a6fa14797efd0e23e80`：38次调用全部执行回收、锁释放，NUMERICAL_FAILED，未提升生产。以下数字来自用户摘要，未独立读取设备文件；随后CPU审计给出的OMC SHA为`288644ea39913fc49466ed4b6d9329825b6ea819259c266c4601cba391b5e182`。导出为六组QK MatMul → FP32缩放／mask／Softmax → FP16概率 → PV MatMul，完整208行逻辑KV、GQA INT32 Gather；学习权重、mask、17／14接口、特征Conv投影和原门槛保持。

| 当前组合版指标 | isolated | feedback |
|---|---:|---:|
| 整例通过 | 0/19 | 0/19 |
| 有效候选Top1匹配 | 82/86 | 83/86 |
| 有Top1差异的case数 | 3 | 3 |
| hidden失败case数 | 8 | 8 |
| 新增KV相对L2范围 | 0.004417～0.008284 | 0.004417～0.008284 |
| hidden相对L2范围 | 0.013387～0.025756 | 0.014494～0.026116 |
| KV失败case数／未写区失败 | 13／0 | 18／0 |
| proposal同步API均值 | 760.223 ms | 773.596 ms |
| catchup同步API均值 | 736.501 ms | 754.463 ms |

首例空context的第10号候选为220、参考13；hidden最大绝对差0.15234375、相对L2为0.0144943。其余Top1差异位于case17／18的容量边界与重复；两个feedback重复均逐位稳定。相对旧融合版Top1各0/86和hidden约1.35～1.39的相对L2，候选数值明显改善，但逐元素及Top1门槛仍失败。新增KV误差范围与旧融合版一致，其计算支路不经过候选attention；空context仍有误差说明单独修context Conv不足以闭合全部精度。

用户随后读取当前编译manifest：`PYTORCH_DRAFT_OMC_COMPILED`、`composed-attention-v1`、CPUCL=3／NPUCL=2，共5个子图。组合版与上一融合版的静态分区数量相同，不能将当前耗时归因于子图数量增加，也不能把53→5的旧结构改善当作已实现加速。用户随后提供当前逐算子及模型耗时表，详见上方记录；算子热点已见，但模型墙钟与算子加和之间的差额来源仍未知。单次厂商工具时延与常驻SDK统计分开解释。

新增 `validate_incremental_draft.py --audit-reference`：在服务器CPU复用四个代表case、hash绑定的prepared权重及原参考，比较源表达、板端和golden；另在CPU对齐RMS先转FP16再乘gamma的原参考边界，不更改生产模型。报告按层拆分新增K/V。14项主机检查通过，含缩小宽度六层实际计算、反转物理输入槽、已知hidden故障定位、原数据不变、失败状态保持、固定报告复用及身份／权重／AIR／控制量拒绝；测试的完整宽度ABI入口用mock替代，未运行用户真实权重或板端。证据run `20261103T101101Z-15574-ff6859`，用户已执行真实权重四例审计，结果见上方；当前性能诊断命令见[当前阶段](current-status.md)。

本地缩小宽度、保留六层与八个Target taps的32组序列／440次调用覆盖C16／C64、K15／7／3／1、滑窗开关、独立输入与连续缓存反馈。全部14输出与旧原语实现逐位相同，输入及未写KV保持。独立完整前缀oracle的hidden／KV通过原0.01／0.01门槛，最大绝对差0.0068359375；容量边界和重复反馈共有4个Top1差异，旧原语路径同样存在，该oracle检查记为FAILED。四档真实宽度meta捕获为12 MatMul、6 FP32 Softmax、32测试RMS、17／14接口、INT32 Gather、无融合attention。14组源图及mock导出编译检查、7组旧矩阵回归通过。以上均为主机证据，临时测试未发布。

融合矩阵补测来自用户回传，未独立取回设备文件。尝试 /data/local/tmp/qwen35_vector_bf2f47fe1c8548f18eaea37c8db6d7cb，50／50请求完成、native与归档退出均0，45失败、5通过，重复稳定。与首批79例去重后为127／129例比较、114失败、13通过；KV1执行失败，KV15未执行。补测INCOMPLETE表示只覆盖总矩阵子集，不是这50例执行中断。

| 补测组 | 比较数 | 通过数 | 行为 |
|---|---:|---:|---|
| Q32，Draft头配置 | 10 | 1 | 对应prefix32；key16通过，key32／192失败 |
| Target prefill头配置，Q64 | 10 | 2 | 对应prefix64；key16／32通过，key192失败 |
| Target decode头配置，Q1 | 10 | 1 | uniform／weighted精确匹配prefix1；选择key0的mask_rows通过 |
| constant_mask | 1 | 0 | 精确匹配declared_mask_prefix16 |
| runtime，含2个重复基线 | 15 | 1 | KV16通过；KV17至208已测长度仍与KV224基线逐字节相同 |
| observed | 4 | 0 | 仅attention失败；辅助读回通过，对应runtime attention逐字节相同 |

有限mask响应匹配带mask的受限前缀。页表rotate／reverse改变输出且匹配按页表映射的prefix16；补偿存储与页表置换后逐字节保持。Q16／32／64／1对应相同长度的有效KV前缀，但Q物理长度、actual_seq_lengths_q和all_seq_lengths_q同时变化，尚不能确定哪个量决定范围或物理原因。Target组仅为相同头部几何的合成图。该调查暂缓，先验证组合版完整Draft。

以下保留旧融合版完整模型与矩阵的历史实测记录。


当前开发集中推进PyTorch Draft，手写GE／AddCustom优化按用户要求暂缓。现有公共GE产物继续作为已接通的运行基线，PyTorch尚未替换它。

## 最新完整PyTorch复测：非零hidden，anchor仍未影响共同输出行

2026-11-03，用户回传新模型的recover与diagnose：OMC SHA256 `52cf888e66f8144cbb1a4e7a71ddd221b22a115cce80e5508d1d89c92a6d5eee`，板端尝试 `/data/local/tmp/qwen35_draft_accuracy_4375459a4d7e45a1a3a60f55a34ccd64`。两组各19例执行并回收，锁释放，原门槛下仍为 `NUMERICAL_FAILED`，Top1各0/86，生产未提升。以下数字来自用户JSON，未独立读取原始张量；本次未附编译准备receipt。

单位缩放旁路后的复测中，空context的30720个hidden元素已无零值，但15行逐位相同，Top1全16，hidden相对L2为1.357387893。context=18时有15个不同hidden行，Top1开始随位置变化；anchor改为32／34／38时，共同有效hidden前缀仍与前一proposal逐位相同，独立参考明显变化。context=192时输出也随位置变化，两次完全相同的请求逐位复现。控制文件hash和值匹配不等于设备输入回读。

| 指标 | isolated | feedback |
|---|---:|---:|
| 完整用例通过数 | 0/19 | 0/19 |
| Top1匹配 | 0/86 | 0/86 |
| hidden相对L2范围 | 1.349196～1.392399 | 1.349708～1.392346 |
| 新增KV相对L2范围 | 0.004417～0.008284 | 0.004417～0.008284 |
| KV失败用例／未写区失败 | 13／0 | 18／0 |
| Proposal同步API均值 | 742.979 ms | 732.262 ms |
| Catchup同步API均值 | 736.038 ms | 715.850 ms |

新增context KV不经过候选Attention，其误差范围与前轮相同不能排除临时block KV、mask、Attention或残差路径错误。当前只支持“候选输出恢复非零、anchor敏感性仍异常”，不能据此确定融合内核、RoPE或任一具体节点为根因。同步API仍是混合输入诊断计时，50ms目标未达到。

新增 `probe_draft_attention.py`：从同一源AIR保留第一层融合Attention的属性、长度／block-table端口和mask构造，使用三项原控制输入，在原packed Q/K/V输入处切入已知数据。只复制不超过256KiB的小buffer，拒绝embedding等学习权重；一份小OMC覆盖空缓存、改变临时anchor值、部分缓存、较少候选、容量边界、加权QK、零候选七组输入。输出Attention、mask是否为零的谓词、packed V展平回读；当前谓词使用FP16 0/1，首次INT32版本的编译失败见下文。后两项逐字节比较，Attention沿用0.01／0.01逐元素门槛。独立ND参考采用显式坐标打包，padding写入哨兵值以暴露误读；probe不包含投影、RoPE、QKV打包算子及后续层，也不保存完整图分区。mask谓词不是原始负无穷字节或物理内核输入的读回。

12组主机合成／mock检查通过：原属性和非QKV端口保留、学习权重祖先排除、静态Fill／ZerosLike准备、拒绝错误长度／shape／控制依赖、七组mask解释执行和独立SDPA比较、打包坐标、重排公共ABI、绑定文件篡改拒绝、mask／Attention故障与NaN诊断。原前缀探针23项回归通过。该轮主机检查不证明DDK编译或设备数值；完整模型精度门槛、权重、源码数学表达和v5准备规则未更改。

2026-11-03，用户回传上述Attention探针的实际OMG失败：DDK `100.520.020.010`、目标 `kirinx90`，8个输入快照共50951字节，未复制学习权重。源AIR为 `e976adece11fc82e0911fb82552cdf6fa9115ffc2c53e3221e4df05a1af53d95`，首次准备后的probe AIR为 `a9bf96121db566099735213f14bb22c202201b28f47d0dba0fd32c24907181c6`。明确终止链为 `probe_attention_visible` → CastT SelectKernel → `kernelId not registered: 196620` → `compileInfo is null` → SubGraph_3编译失败，OMG退出1。此前Cast_44／Cast_47推形和AdnFusedInferAttention建算子告警不能代替这条终止证据。本次未生成可验证成品、未上板，不能认定Attention本体数值错误。

修订为 `source-first-attention-mask-packed-qkv-v2`：只把新增回读支路的BOOL→INT32 Cast替换为Equal条件加SelectV2，选择完整形状的FP16常量1／0；不对负无穷mask做乘法或其他算术。mask golden改为同义FP16谓词，仍逐字节比较；Attention参考、0.01／0.01门槛、源Attention其他端口与mask构造保持。附加边仍可能改变编译分区，修订后的实机结果待回传。分析支持异常非二值／NaN回读并保留失败状态。

修订版12组合成／mock检查通过，包括七组mask解释执行与独立SDPA、原节点字节保留、无新增Cast的FP16谓词、物理ABI重排、原目录失败后重编、mask／Attention／NaN故障；19项目录重试检查通过。`--replace`限于直接draft/tests子目录内本探针已知失败的OMG编译，采用与验证器相同的源目录锁，拒绝成品、板端运行、未知退出、符号链接和仍占用目录的编译进程；确认后删除该失败目录，不保存备份。以上全部为主机检查，不能记作所选DDK编译通过或板端通过。

2026-11-03，用户回传Attention探针v2实际编译通过：probe AIR SHA256为 `9fccfba3296b3b07177b040a205c3692ff07101275e82e3fc681db073df8e183`，内嵌OMC为 `df4248dd95204cf882ad5d0f088eeb82697491f11b373768420d529ad35d09ff`，218998字节。它仍使用同一源AIR、原Attention属性及非QKV端口；FP16谓词回读消除了前次新增CastT的编译阻断。这只确立本探针编译通过。

板端尝试 `/data/local/tmp/qwen35_vector_612345cd94bd4d5b85fcf57d50673401` 在首例 `pytorch_attention_empty` 退出1，archive退出0，其余六例没有执行退出记录；恢复为FAILED_OUTPUTS_RETRIEVED、锁已释放、数值NOT_RUN。HIAI查询的输入形状依次为 `[1,256,16,16]`、两项 `[7,64,32,16]`、三项 `[1,1,1,1]`；输出为 `[1,256,16,16]`、`[1,1,16,224]`、`[1,7,64,512]`，format均2，dtype不可查询。原运行器在第0槽期望 `[1]`，因此在Process之前中止。Load和IO查询已完成；没有Attention数值或时延结果。输出前置单位维仍由原ND兼容规则检查，不因本次输入错误修改此规则。

现有小探针绑定按编译Data.index排序。用户日志证明该计划与此OMC的HIAI输入顺序至少在第0槽不一致；两项K/V和三项控制量同形，不能从维度推断它们各自的语义。新增 `probe_draft_attention.py --inspect-compiled` 只读当前OMC，报告GraphDef输入引用、Data节点序号／index／属性、NetOutput和现有绑定，明确运行顺序未解决；不执行自动重排。11项主机检查覆盖排列与index冲突、同形歧义、缺失图输入、错误绑定仍可查看、身份/hash拒绝、只读性与CLI互斥；12组原探针回归通过。真实编译元数据待用户回传，完整Draft精度状态不变。

2026-11-03，用户回传上述同一OMC的只读接口检查。GraphDef.input明确为Q、K、V、start_position、valid_rows、proposal_count；对应Data.index仍是3、4、5、0、1、2。此编译图边界声明与此前HIAI描述的形状序列一致，旧绑定器按Data.index排列为控制量在前，解释了第0槽在Process前退出。角色由明确的Data名引用确定，同形状K/V和三个控制量没有用shape猜配。

Attention探针现在单独启用 `COMPILED_GRAPHDEF_INPUT_REFERENCES` 绑定，计划runtime index按GraphDef.input生成、fixture_index保持原文件身份、data_index单独记录原属性。缺失／重复／未知引用、非Data／非0端口、描述变化均拒绝，不回退到节点排列或Data.index。原OMC与源夹具字节不变，不要求再次编译或构建native；完整Draft和其他探针的输入绑定方式没有随此修复改变。新的物理执行、mask及Attention数值验证仍待用户回传，不能由边界声明确认同形状输入已在设备正确消费。

本轮24项主机检查覆盖用户报告的顺序冲突、独立打乱节点与index、同形状角色引用重排、非法边拒绝、旧绑定兼容、未知尝试保留和已关闭失败尝试同目录重试。11项接口只读检查、12组合成／mock探针检查和23项前缀回归通过；其中七组实际生成的输入／输出文件逐字节匹配各自原夹具，OMC／编译目录hash保持，原数值失败门槛仍生效。它们不是实际DDK或板端数值结果。

同轮用户运行旧四组探针清理，候选／删除／保留列表均为空、logical_bytes_selected=0。本次未删除文件，没有可归因于本次调用的磁盘释放量，也不代表完整模型或当前Attention探针已被删除。

2026-11-03，用户回传修正输入计划后的七组Attention执行与离线分析，尝试 `/data/local/tmp/qwen35_vector_9178f47c526e444593a912330176d7b1`。同一OMC `df4248dd95204cf882ad5d0f088eeb82697491f11b373768420d529ad35d09ff` 完成全部调用，native、archive、七组case退出码均0，锁已释放；每组三次预热、十次输出逐位稳定。三项输出中mask可见性谓词、packed V逐字节匹配，Attention全部失败，数值均有限。相同OMC的输入顺序修复已经越过原先的Process前描述检查，但本次未验证完整Draft。

| 用例 | Attention最大绝对误差 | 相对L2 | 同步API均值 ms |
|---|---:|---:|---:|
| empty | 2.031250000 | 0.799853595 | 1.7280 |
| anchor | 1.968750000 | 0.898641129 | 1.7672 |
| partial | 0.516601562 | 0.540583581 | 1.6121 |
| short | 0.112548828 | 0.289828023 | 1.4768 |
| full | 0.104003906 | 0.183729832 | 1.5454 |
| weighted | 0.553222656 | 0.537972693 | 1.4305 |
| catchup | 0.119628906 | 0.303020848 | 1.7837 |

分图CPUCL=2、NPUCL=2；以上API墙钟包含同步调度，快照在调用之间、不计入API时间，不是纯NPU时间或完整Draft稳态性能。用户诊断显示各组mask可见key计数与参考相同、padding可见数0。六个非加权用例首头首通道的16个查询位置均为-0.03125，anchor增量由+2变为-2仍相同；加权用例均为-0.027435302734375。独立参考在空context的首个查询分别为+2、-2，后续查询也有预期差异。

本地主机按已有合成夹具计算，忽略mask并只取KV逻辑前16行，恰好复现上述非加权与加权的两个FP16样本。前32／192／208／224行给出不同的首通道值。同一假设复现全部七组最大绝对误差；六组非加权相对L2与用户结果相差至多1.11e-16，加权组预测0.537950372929622，对比用户0.537972692833318，差2.23199e-5。该结果是 `FIRST16_UNMASKED_SIGNATURE_HYPOTHESIS`：真实完整张量尚未独立读取，不能把样本与汇总指标匹配直接写成逐元素相等或具体长度端口的根因。当前源码构造的actual_seq_lengths_q／actual_seq_lengths_kv／all_seq_lengths_q为16／224／16，converter沿用原Target的端口转发；源码数字和辅助mask谓词不证明运行内核的实际读值或掩码解释。

扩展现有只读 `--analyze`：从hash绑定的保留Q/K/V按确切角色与槽位解包，对每例65536个Attention输出元素比较上述五种前缀假设；独立重算原可见性golden，报告全部query行的逐位差异，并比较正负anchor用例的整个输出。hypothesis是诊断结果，不替换golden、不参与原验收；即使命中也保持NUMERICAL_FAILED。14项主机检查包括随机完整宽度SDPA对照、显式NZ坐标、两项用户样本复现、七例故障注入、首通道外错误、NaN、部分选择、输入篡改、十次快照复查和只读性；12组探针回归通过。实际完整输出签名仍待用户在原result上重新分析，当前不新增编译或板端尝试。

2026-11-03，用户再次回传同一已回收尝试 `qwen35_vector_9178f47c526e444593a912330176d7b1` 的完整张量签名分析。六组非加权用例各65536元素与无mask前16行SDPA逐字节相同，最大误差、RMSE、逐位差异均0；加权组最大误差0.00048828125、相对L2为0.0003691992091269304、RMSE为0.0001091502326789014，21424元素逐位不同，原容差内无失败元素。加权实际query行有15种，不能把首通道相同推广为所有行相同。非加权前32行假设也在容差内，但65536元素均逐位不同、RMSE为0.001953125。七组原golden全部逐字节复现。

正负临时anchor的实际65536元素完全相同，参考全部元素不同、最大变化4、RMSE为1.2587414108669686。此证据支持“临时anchor未体现在该探针输出中，输出符合无mask前16行计算”的数值结论；不能单独证明某个长度端口或mask内核实现错误。原Attention七组数值失败、十次重复稳定和完整Draft失败状态保持。本轮用户只运行离线分析，没有新增设备执行。

新增可选 `--controls`，在同一小图中保留原Attention，另复制四个分支：仅看key0、仅看key15、无mask且actual_seq_lengths_kv=224、无mask且actual_seq_lengths_kv=32。修改限定于复制节点的mask／KV长度连接，原属性、QKV、block table、actual_seq_lengths_q=16、all_seq_lengths_q=16保留；query长度FileConstant必须绑定已验证的INT64源sidecar。源mask回读仍只观察原动态mask辅助支路。新增参考独立于实际输出，不使用前16行假设替换验收；单key参考为对应V行，另两项为指定范围的SDPA。不同输出shape只做原NZ字节视图，方便唯一ABI绑定。

14组主机合成／mock检查通过：原分支字节保留、四项修改范围、固定mask内容、长度32和224、所有query长度保持、七输出唯一描述、单key直接V参考、独立PyTorch SDPA、乱序输入／输出绑定、十次比较、全部分支错误截到16行时仍判失败、单分支故障定位及不合法query长度来源拒绝。原探针12组回归和签名14项检查通过。新对照的真实DDK编译和板端执行均NOT_RUN，完整Draft未改动或提升。


2026-11-03，用户回传同图mask／长度对照的执行及完整分析。OMC SHA256 `856090b9aaa37a5d578ba297da0229eb2ef9325aa3a8e12ab166f9301e3884a0`，尝试 `/data/local/tmp/qwen35_vector_f4e9e86b2f17408386e313b393d378af`。native、archive及七组case退出码均0，锁释放；三次预热、十次输出均逐位稳定。此处证据来自用户JSON，未独立读取板端张量。mask谓词和packed V全部逐字节正确，原Attention七组仍失败。

固定mask只看key0／key15的两个分支，在全部七组十次调用中逐字节匹配参考；首头首通道分别为−0.5／0.4375，整个65536元素的分支差最大值与RMSE均0.9375，与参考一致。这证明在本图中固定mask能选择这两个key，不能再用“所有mask都被忽略”解释结果。

无mask且actual_seq_lengths_kv=224的分支、无mask且该长度=32的分支，与原Attention的全部65536输出元素逐字节相同，包括加权组。六组非加权结果逐字节符合前16行SDPA；KV32参考虽在原容差内，仍全部元素逐位不同、最大误差和RMSE为0.001953125。加权KV32分支有13696个失败元素、最大误差0.0146484375、相对L2为0.032013517179236735。非加权“通过”不证明长度32生效。加权前16行签名仍只有0.00048828125最大差，原Attention在加分支后仍保留此前异常特征。

以上支持“固定mask有效，而此次224→32的长度端口改动没有可见输出变化”的观察；尚不能区分长度端口语义、编译传递、常量处理或内核实现。动态mask辅助回读并非物理输入回读，原空context中前16个key都被屏蔽也需结合内核处理解释，不能仅凭均值断言mask未消费。七组五个Attention分支的所有golden均由独立参考逐字节复现。整例汇总误差现在可能来自无mask对照，应使用各分支指标判断原Attention；该诊断图的API均值2.873～3.500ms包含多个Attention，不能充作完整Draft性能。

新增只读 `probe_draft_attention.py --inspect-lengths`：校验既有OMC和所选OMG的hash，读取保留编译日志／文本元数据，按所选DDK的限定算子目录及Attention文件名寻找可读实现／注册候选，输出长度相关行号和hash。已有IR检查扩展到32个端口，完整保留Attention的18个输入及第16端口生产者。源码候选不是已加载实现证明；缺少IR／源码、不透明NPU任务和读取限额均明确报告。不执行厂商代码、不改图、权重、fixture、DDK或数值门槛。15组新增主机检查覆盖只读性、准确行号、hash命名编译元数据、端口16、DDK／OMG／日志／元数据篡改拒绝、路径与符号链接边界、读取限额及CLI互斥；11组接口、14组对照和12组原探针回归通过。真实DDK长度检查尚待用户执行，完整Draft仍NUMERICAL_FAILED，未提升。


2026-11-03，用户回传 `--inspect-lengths`：同一controls OMC为341503字节，hash仍为 `856090b9aaa37a5d578ba297da0229eb2ef9325aa3a8e12ab166f9301e3884a0`。所选OMG hash为 `f199b5cdc78eb694d78c952b6a15feada16a6f6196f39358e739670eae932aa0`；编译report hash为 `a11e69b49651d19f848e5bb36cd7a5ce88ae139e28a9de6e1e891c9271a925fa`。主图没有匹配Attention节点，IR为NO_DUMPS；限定SDK目录扫描3801项、未触限，未找到可读实现候选。这不证明整个安装包没有实现。

compile.log为985893字节／8862行，hash `b270f2ba1161e9f20d86f57ab2564afcfb8dbf8f02d44b23b4b8f1345c1e65bd`。摘录出现ACTUAL_SEQ_LENGTHS_Q_ATTR读取失败后返回NOT_CHANGED、KV32常量跳过INT64→INT32转换、ATB参数／mask检查成功、`seqlenQ=16`、tiling key 1329990976、blockdim1及992字节参数容量。唯一保留的Attention kernel JSON为602字节，hash `fa90f1f9361b8032e5d1ccf764a4c8222299cb29abb2a57fbfeba9276ca60a30`，声明AiCore与opParaSize992。query16与Q形状一致，摘录未提供有效KV长度或完整参数映射；NOT_CHANGED不是本次成功编译的终止错误，单个kernel文件也不证明分支错误合并。此轮只读检查没有新增编译或设备执行。

本地再次核对原Target的export调用：`allQLen=torch.tensor([seq_len], dtype=torch.int64)` 作为 `dynamo_export` 的运行时输入，转发到GE端口16；当前Draft同值来自FileConstant。这是已识别的结构差异，尚不是实机根因。新增 `--runtime-lengths` 同图对照保留原Attention、源mask及V回读，另加四分支：仅端口16改运行时16、仅端口6改运行时224、两端口同时改、两端口同时改且mask全0。两个INT32公开输入经Cast精确扩展到INT64，所有query16／KV224值、QKV／block table／Attention属性保持，8输入7输出沿用现有native runner。图内Cast和新增分支可能改变分区；没有用未知端口解释扩大Q长度，也没有修改完整模型。

7组新主机合成／mock检查通过，覆盖GraphDef／ModelDef原节点字节保持、声明外端口不变、8／7 ABI及唯一输出视图、INT32→INT64精确解释、全部七组原golden与前三个新分支一致、独立PyTorch SDPA、编译失败／重试与准备流程、乱序物理槽位、十次比较、全分支错误截到前16行时仍失败、单分支故障、hash有效但值错误的runtime长度拒绝、CLI互斥。14组mask／长度对照和12组原探针回归通过；真实DDK编译与板端执行均待用户完成。原数值门槛及完整Draft NUMERICAL_FAILED、未提升状态保持。

2026-11-03，用户回传运行时长度对照编译失败（DDK 100.520.020.010，OMG退出1）。提取AIR hash为 `5a7ad0a80fac17cfdfc5505c59592672f2f72a099583ab1f9f731aa284d3301f`，预处理AIR hash为 `8d81d0a39f0b1fe59b6867d80284189d94f90dd3a8a92e71271c4162b1b4697a`。编译尾部明确为 `probe_attention_kv_length_i64` 的INT32→INT64 CastT报 `kernelId not registered: 589827`、`SelectKernel failed`、`compileInfo is null`，随后SubGraph_1编译失败。该新增转换未获得当前NPUCL实现，runtime长度假设没有得到设备检验；原Attention数值问题仍未闭合。

修复将两个长度输入直接声明为INT64[1]、各8字节，连接原第6／16端口，移除新增Cast，保留原图、16／224值、四分支参考及数值容差。提取策略为 `source-first-attention-runtime-lengths-v2`。现有HIAI执行器的INT64分配／字节长度路径无需改动；探针计划解析、准备时ABI字节计算和native构建能力记录补充INT64，启动前检查实际所需dtype并在HDC前拒绝旧runner。INT64输出比较另要求逐字节相等，防止FP64不能区分大整数相邻值导致假通过。

7组图／夹具／绑定／比较检查、14组原controls和12组原探针回归通过；另4组检查通过：以真实探针main和头文件构建主机mock，验证高32位保留、3预热＋10次快照和错误字节数／容量拒绝；旧runner在设备访问前被拒绝；2^53与2^53+1比较失败；构建记录声明INT64且绑定探针二进制。这些均为主机证据，所选DDK的直接INT64图编译与OHOS runner重编、设备执行仍待用户运行。复用原失败目录，未生成版本目录或修改共享DDK。



2026-11-03，用户回传直接INT64长度对照的保留输出分析，OMC SHA256 `cb1c94b0b0703a4fdb9825a236ac8dca38baf50e2864b5bc70e293b210d69471`，尝试 `/data/local/tmp/qwen35_vector_03fa34f46ce34ac890dbd979dd964b36`。七例十次逐位稳定，mask可见性和packed V精确通过，原Attention与四个runtime分支均不满足原参考。每例四分支与原Attention的全部65536元素逐字节相同，包括加权例；改变端口6／16来源并没有修复结果。六例仍精确匹配无mask前16行，weighted最大差0.00048828125。empty／anchor实际输出相同，参考最大差4。CPUCL=2、NPUCL=2。此次收到的是分析JSON，未另收到编译报告或native重编记录；不补造锁状态和构建hash。完整Draft仍未复测。

前几轮局部对照不足以区分端口处理、分页布局与shape相关实现。新增独立8模型矩阵，共129例，详见当前阶段页。保留原源图7例；其余122例覆盖逐query mask、有限与负无穷mask、Q/K扰动、13个KV边界脉冲和单key选择、12个运行时KV长度、页表置换与补偿置换、辅助回读影响、Draft Q32和Target Q64／Q1的head几何。Target来自配置的16／2／256参数与Draft 32／8／128不同，仍使用合成KV224；不是实际Target AIR的数值验证。若某形状失败，只能先记录该组合失败，不能把编译成功当作数学支持或把一个形状通过当作整模型通过。

7组矩阵主机检查通过，包含全部122例的独立Torch FP64 SDPA、序列化图解释执行、受控编译失败后完成8组尝试、乱序ABI、全部十次模拟快照和错误前16行签名识别；原7／14／12组检查通过。矩阵的设备数值／shape支持仍待用户执行，未更改权重、完整模型或数值容差。


2026-11-03，用户启动矩阵后长时间无输出，按Ctrl-Z后shell显示`Stopped`。请求`qwen35_vector_297df57991e34b2a8ad76aed1118f437`第一次recover返回`VECTOR_VALIDATION_PENDING / RETAINED_OR_UNCERTAIN`，后两次因`another local model job is running`拒绝。没有该矩阵的数值输出；不能把这次等待归为Attention shape不支持或内核卡死。源码确认旧启动逐文件HDC发送／校验且没有进度，request在上传前创建；缺少complete就标记running并不证明已发起模型调用。第一次回收目录不存在，因此跳过复用检查；第二次复用检查扫描整个AI_RUN_DIR，挂起的启动脚本也会触发拒绝。

修复增加只读本地`--status`、持久阶段记录和30秒等待提示；同一源请求的独立vector回收目录只检查引用该回收目录的活动任务，原执行目录复用仍保留全局活动任务检查。新请求打包上传并核验归档和每个成员SHA，板端记录当前例／归档阶段并打包日志回收；旧请求兼容逐文件回收。未完成请求报告`completion-unconfirmed`和证据标记，不宣称进程仍存活，也不重启模型或强制解锁。7组新主机检查包含真实SIGSTOP进程、重复回收、只读状态、冲突写入保护、真实shell／tar流程、损坏拒绝、失败停止及十次host-fake输出比较；24组原ABI／目录生命周期和7组矩阵检查通过。助手未连接DDK服务器或板子，本次旧请求实际停留阶段仍需用户status／recover回传。

随后用户回传本地状态：PID68999为`T (stopped)`，已有1172条通信记录，最后一次为成功的`vector-upload`。`fg %1`后旧进程在发送`attention_matrix_runtime__key_64/plan.txt`时报180秒超时，结果为`FAILED / NOT_RUN / NO_KNOWN_OWNED_LOCK`；未进入模型执行。用新版重启后129例准备通过、无准备失败，1021文件共146871831字节，在`pack`阶段报`invalid or duplicate probe payload destination`，仍未访问设备执行。

本地确认新增路径正则遗漏`+`，正常部署的`lib/libc++_shared.so`即可复现同一错误；上轮传输测试将运行库清单模拟为空，漏测了这一项。修复复用原部署`_relative`合同，保留绝对路径／父目录／控制字符／反斜杠／重复目的地／保留名拒绝；错误包含具体成员与位置，并保留`payload-manifest.json`。新增3组检查含实际`_build_payloads → discover_libraries → SDK C++库发现 → start → pack → tar解包 → sha256sum`链路，使用明确标记的合成AArch64 ELF和构建记录、129例实际夹具生成逻辑，共1021成员逐文件校验通过；原7组启动／回收检查补入C++库名后通过。合成ELF从未执行，未访问DDK服务器或板子；Attention数值结论未更新。

2026-11-03，用户回传矩阵分析与完整recover摘要。8／8模型编译成功，129例请求仅79例完成比较，71例失败、8例通过；全部已比较例十次逐位稳定。尝试 `/data/local/tmp/qwen35_vector_a883382ee2d74125aeb24bb88fead246`，native退出1、输出归档退出0，回收 `outputs.tar` 为136690176字节，状态 `PARTIAL_OUTPUTS_RETRIEVED / NO_OWNED_LOCK_AFTER_KNOWN_COMPLETION`。第80例 `attention_matrix_runtime__length_1`（Q16、运行时KV1）输入／输出描述检查及ND尾部补1接受后，`HIAI Process(sync) failed: 1`；不是编译或传输失败，也不是本轮记录的超时。后49例没有退出记录，脚本按原生失败停止；因此Q32／Target等尚无数值结果。

| 组 | OMC SHA256 | 已比较／计划 | 数值失败 |
|---|---|---:|---:|
| source | `d58f04ba408de19f2cc0983cc435725386f764aa0faed4788de68ef18307495d` | 7／7 | 7 |
| constant | `f75b15e3f2415aeefb6cf054e2d81ea716ba27280e4e5aae85cdfcfd320707fb` | 36／36 | 32 |
| runtime | `0032c80ce23dde92ca8588127f0f828a99bf302cc4a844cc037181419bdf5147` | 36／51 | 32 |
| constant_mask | `759bb6ca1e4c4e683c66e401e82d7ae9c30ed7355540830767d3512311578171` | 0／1 | 未比较 |
| q32 | `f669dadba7b8fd76a1f7d72dfeacdb9e640232f0033522aae89037bcf9548bea` | 0／10 | 未比较 |
| target_prefill | `0f46a4de4a9445e17c7557851e5a56134c3ff0d8d30ce1b75141b08dda587558` | 0／10 | 未比较 |
| target_decode | `b952125f37604e7f6c873d088bbffebc31222bf68765c803acb0984b5b6c36a7` | 0／10 | 未比较 |
| observed | `2acb338d37943da6fafd49112e27ac39de58f3ef37d2a63067bbf8e705abc0c5` | 0／4 | 未比较 |

constant／runtime完成的36对输出逐字节相同。两组均只有前缀因果mask、zero_v、key0、key15通过；原source七例仍仅Attention失败，mask可见性／packed V辅助读回通过。两组V幅值8的行0／15脉冲均使输出变化0.5，完整224行参考变化0.03570556640625；行16／31／32／63／64／127／191／192／207／208／223的脉冲与zero_v输出逐字节相同。该组独立扰动联合支持有效KV范围局限于前16行的现象；单个零输出可能匹配多个前缀，不能单凭closest_signature=prefix1认定只读1行。weighted改变Q使实际输出变化4.513671875；仅反转K第32～63行则输出不变，参考应变化0.3421783447265625。两种大负mask值产生相同结果，前缀因果mask通过及mask_bias与无mask签名不符，不支持“所有mask完全被忽略”的概括。

这一结果把已复现故障集中到Q16／KV224的融合Attention接口执行路径；仍不能单独确定是端口／布局／分页、编译处理还是内核实现。runtime KV1发生Process错误，说明不同长度值会改变执行结果，不能断言所有运行时长度输入被忽略。8组编译日志的tilingkey均为1329990976，blockdim1，分别记录seqlenQ16／32／64／1；相同key不能证明其他tiling字段相同或这些形状数值正确。Target组尚未执行，完整Draft／Target精度与性能结论保持未通过／未验证。

分析器此前丢失执行诊断，现保留原恢复的状态、首个原生失败、退出码和锁状态，补充分组覆盖及首个未比较例；未比较不自动等同于执行失败。新增保留原加性mask的KV前缀假设，明确不模拟全负无穷行fallback，显示多签名歧义，并标明K扰动区间及mask响应。这是保留输出的诊断增强，未更改夹具、OMC、执行停止策略或验收门槛。5组新增主机检查及7组矩阵回归通过。后续使用已编译模型补48例并带2个runtime配对基线、共50例，形状先行，原79例保留；KV<Q的length1／15单列为后续边界调查，不标为通过。

随后用户回传同一尝试、同一批79例的新版只读分析，没有新增设备执行。constant／runtime两组的 `mask_rows` 均与 `declared_mask_prefix_16` 逐字节相同，`mask_bias` 均与该签名在原门槛内匹配（最大误差0.0078125），两者在被检查的候选中各只匹配该签名。uniform→mask_rows的实际输出最大变化8，weighted→mask_bias最大变化8.3212890625，直接证明这些mask扰动对输出有响应。结合此前Q/K/V响应，有限mask实验支持 `softmax(Q K[:16]^T / sqrt(D) + M[:,:16]) V[:16]` 的输出行为；原KV224参考仍失败，不能把诊断签名匹配当成修复或数值验收。

仅选择key16及之后位置、或temporary causal的有限mask，在前16行全部为相同大负值；这些零QK夹具的截断参考因此为均匀加权，同时匹配有／无mask前缀。这个歧义不支持“完全未读取mask”的推断。`mask_rows_inf`与有限mask输出相同，但截断后全负无穷query行没有在诊断参考中规定fallback，不能据此认定厂商做了clamp或采用某种softmax策略。Q32、Target及长度／页表补测仍未回传；不提升完整模型，也不从输出签名宣称具体内核根因。

## 此前PyTorch独立执行：加载成功，数值失败

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

2026-11-03，用户继续运行组合attention精度验证，同时要求研究更快的特征投影。当前图和测试命令保持；以下是主机布局检查与待测方案，不是新板端性能结果。

优先候选是把token从Conv的batch维搬到空间维。C16原输入为`[16,1024,4,4]`，每个batch输出一个位置；候选把16个token排成4×4个互不重叠的patch，输入为`[1,1024,16,16]`，原权重仍为`[2048,1024,4,4]`，stride改为4×4、无padding，输出`[1,2048,4,4]`，再还原为`[1,16,2048]`。具体映射为`packed[0,c,4*i+h,4*j+w]=X[4*i+j,16*c+4*h+w]`，必须实际重排输入，单纯reshape不成立。每个输出仍归约完整16384项，不增加FP16部分和边界；是否改善权重复用和tiling仍需所选DDK实测。

主机使用确定性合成全宽FP16权重，M=1／2／16／64各测随机输入和边界脉冲，共8例；空间网格／横向条带与原Conv全部输出逐位一致，相对独立`F.linear`最大绝对差0.00048828125，原逐元素门槛均通过。另对四种M的全部输入地址检查重排与恢复逐元素一致、窗口无重叠，权重字节保持。这些不是训练权重或X90数值验证。证据在run `20261103T095423Z-15096-6621a2` 的 `projection-layout-checks.json` 和 `projection-review.json`，测试程序仅在run内。

C16的逻辑输入0.5MiB、输出64KiB、权重64MiB、运算量约1.074GFLOP。单独物化输入重排和输出恢复至少增加1.125MiB读写；实际格式转换、padding及重复权重流量未知。后续小图应对照原batch、空间条带、空间网格三种表达，计入完整pack→Conv→unpack的时间与分区，以同一权重／输入、3次预热和10次输出比较，再通过完整Draft精度及同步API门槛。不能从历史70.968ms或主机结果承诺收益。

第二方向是仅计算有效context行。小C档可减少填充行的fc和六层新K/V计算，但当前ABI／调度仅支持C16、C64；C1／C2等尚未实现。零追加专用图在`valid_rows=0`时可删除fc→hidden_norm→各层新K/V路径，因为所有缓存写入Select都选择旧KV；正常推进轮通常仍要追加新行，不应增加一次append调用再把proposal的零Conv当作整轮提速。拆K为4×4096或8×2048属于后备方案：FP16部分和会增加舍入，之后转FP32累加也不能恢复，尚未实现或验收。

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

## 2026-11-04 通用 Top1 源码交付

[StableArgMax](../targets/cpro/ops/draft-top1/README.md) 已含 AscendC kernel、host注册/tiling、Torch schema/Meta/GE converter、所选msopgen工程准备。支持合同扩展为静态rank1–8的有限FP16最后维Top1，Qwen为一个验收配置。源码控制流/真实Torch export63项、原参考工具35项、准备工具11项、文本回传9项主机检查通过；这些不证明厂商编译或NPU行为。用户负责复制和在所选DDK编译安装，真实AIR/OMG/设备/性能待验证。TopK与310P适配未实现，完整Draft不变。
