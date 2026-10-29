# 当前更新与执行命令

[首页](README.md) · [环境配置](environment-setup.md) · [验证记录](validation.md)

更新日期：2026-10-29。当前优先级按用户要求调整为：**两套完整模型、真实 target features、多轮 DFlash 端到端执行**。精度和性能另行验收；本页是当前命令的唯一入口，下方折叠命令属于历史。

## 用户现在执行什么

### draft：复用已执行且本例 Top1 全匹配的 residual-fix

用户最新板端回传：完整六层 `full-draft/residual-fix` 已 EXECUTED，165/165 编译精度检查通过，原 Top1/hidden 两路类型保持，**15/15 Top1 与同输入 PyTorch 参考完全一致**。hidden relative L2=0.0126796675、cosine=0.999919614、max abs=0.08984375，非有限值差异为0。它是完整6层、block16、248320词表模型，输出15候选；12处残差加法在FP32计算后回到原FP16边界，没有裁层或缩词表。

这只解决当前固定合成 target-feature fixture 的 Top1，不代表真实特征、多轮或语言质量精度已经通过。`production_equivalence=DIFFERENT` 比较的是旧失败设备模型，不能误读成这次 Top1 失败。保留 `residual-fix/{result.json,compiled-output-abi.json,board/comparison.json,bundle/}`，端到端使用这份模型。**当前不重跑 draft 导出、OMG、打包或残差候选。**

### target：A 原PyTorch 与 B 直接GE并行

用户最新要求两条方案并行，防止GE路线出问题。**A在原权重机器按原PyTorch/TorchAir导出；B在当前服务器对齐原Cpro算子、量化权重和paged/NZ/RefData接口，走直接GE。** 两条均为完整40层、原量化权重、完整248320词表及真实8层features，各自保留独立构建和设备证据，共用上方完整residual-fix draft。

完整命令、边界和状态见[双路线执行](target-parallel-tracks.md)。A原机入口为`prepare_target_pytorch.py`，先只读配置/分片头和小源码准备，缺原参数即停止；资源就绪后才在原机导出。用户已完成B原AIR检查：6361049字节、原文件保持，HIAI schema不匹配（`OPAQUE`）。随后当前服务器已找到两份真正的TorchAir生成文件；两者提取同一7381字节GE descriptor，SHA256为`5a1eac27746a7bb5109e819ce6dd7813a3ffcaab04bd0ab2ca3c2cdf1e4084ea`。选择更小的`3_19`文件继续，不需要从原机再传schema或权重。

**当前执行这一条**，使用实际GE schema只读重解析；完整图写到新JSON，终端打印小摘要，旧OPAQUE报告保留：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/inspect_original_target_air.py" \
  --air /home/l00886453/qwen3_5_moe/glm_export_air_cpro_40block_55_quant/export.air \
  --torchair-pb2 /root/anaconda3/envs/custom_pypto/lib/python3.11/site-packages/torch_npu/dynamo/torchair/_ge_concrete_graph/ge_ir_by_protoc_3_19_pb2.py \
  --full --output "$AI_RUN_DIR/target-ge-source-air-ge.json"
```

schema查找已完成，但是否与此AIR全部字段及89/81边界匹配，仍以这次解析结果为准；`FOUND`不代表图已解析或features已绑定。

同名输出若已存在，工具拒绝覆盖；保留已有报告并使用其路径，不删除旧证据。B后续绑定入口为`prepare_original_ge_target.py`；实际图/来源/算子证据不全时保持待绑定，不把现有functional GE target当成原Cpro等价实现。原源码的norm表达式、量化pack/QPARAM、自定义算子及80状态都要逐项核实。

原权重机器目前只有PyTorch/TorchAir，没有同版本Cpro DDK；OMG继续使用当前服务器原环境，大输入快照、scratch和输出放共享挂载或独立大盘。HF和quant检查点留原机，当前服务器仅需约485.95MiB的原INT8 embedding+FP32逐词scale用于连续输入。新AIR复用源sidecar必须逐项核对；新OMC分区/packing权重仍需重新编译，10GB不作为足够空间的保证。[详细空间安排](original-target-features.md#当前低磁盘安排)。

实际控制输入仍用只读检查器核对，不能用导出dummy值推进连续状态：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/inspect_original_target_inputs.py" \
  --input-dir /home/w00949577/run_35B/chunksize_64_data --sequence-length 64
```

### 端到端第一轮的验证范围

先采用gear1串行verify：完整draft每轮产生15候选，与target已知下一token逐个比较；只把接受的候选或target纠正token送入target，拒绝候选不进入状态。这样可验证多轮提议、接受/拒绝、80状态接续及真实features历史；本阶段不声称并行verify、回滚或加速已经通过。主机调度器已有回归，真实HIAI适配和连续设备证据尚未完成。

服务器约剩10GB，当前只新增小型源码、接口和报告；不再构建任何新draft模型。需要整理旧编译输入快照时可执行以下已有工具；省略 `--apply` 为只读计划：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/compact_draft_storage.py" --apply
```

它仅合并同文件系统、大小/SHA/权限/属主均匹配的只读编译输入快照，保留所有路径和报告，不删除AIR/OMC/bundle或修改prepared权限。实际释放空间取决于文件系统共享情况；原target新导出/编译的空间需求仍需独立核对。

本周最终目标仍包括真实输入多轮正确性、状态处理与同条件至少2x实测；没有因本轮“先执行、可乱码”而改成通过。

## 历史 draft 排障记录（不作为当前执行命令）

<details>
<summary>展开较早的 direct、packed、精度锁和加载诊断记录</summary>

以下保留当时的观察与命令，时间顺序和“现在执行”等措辞属于历史上下文；以页首完整模型端到端计划为准。

最新direct结果已在用户内部DDK通过153/153计算精度检查，16个实际输出类型均与源码一致（`SOURCE_DTYPES_PRESERVED`），打包完成；runner在Load前错误拒绝ABI。原因是全局regex把嵌套 `direct_output_binding.artifact_kind` 也计入，报 `draft diagnostic wire requires its explicit ABI kind`。现修为只校验JSON顶层kind，重复顶层键/转义重复键/错误JSON仍拒绝。更新后继续下方 `run_draft_direct.py` 命令，复用reference、AIR、OMC、observed-fixture和bundle，只重建runner并重跑板端；失败board证据保留在 `direct-trace/board-attempts/<resultSHA>`。

用户明确本周交付仅剩3天。完整正确性、原target普通解码、DFlash多步状态/接受拒绝与至少2x实测尚未闭环；不能承诺在该时限全部完成。target已安排另一窗口同步推进；用户说明旧full-target产物已删除，按没有已验证产物处理，先确认原AIR、量化权重及活跃进程。入口现支持校验后复用完整任务的 `--resume`，不自动重编译失败/中断任务。原89输入/80RefData、64/1动态档和外置权重还缺原接口native适配，不能直接套现有draft或显式状态runner。

最新PID1611的完整packed trace已通过331/331精度检查、证明140条跨分区路径并完成设备执行，但观察数据内部矛盾：同源 `layer_0_output/layer_1_input` 不同，`context_fc` 显示全零而下游归一化非零，QK探针与其缩放后score不一致。诊断hidden与生产hidden相对L2=1.402770、Top1全不同。因此不能将 `first_nonexact=context_fc` 当原生产图首次错误，也不能把探针的3876个非有限差异直接归因于模型算术。

现在使用原值直接输出入口；它先只读复核已有packed结果，再在固定 `full-draft/direct-trace` 构建一份16端口诊断图。原594个计算节点不增不改，无额外观察Cast/Reshape/Concat；Top1/hidden保留，重点观察FC、首层attention/MLP及第0→1层边界，同源语义共享物理端口。原CPU探针参考校验后复用，不重新准备权重或golden；旧生产及packed产物保留。

```bash
git pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/run_draft_direct.py" \
  --tbe-diagnosis "$AI_RUN_DIR/fused-all-f6b81ead7ae8/diagnosis"
```

此入口需要新诊断图的一次编译与上板。打包前静态读取实际OMC输出dtype/shape，不按源码猜buffer大小。本地公开DDK小图出现3个附加FP16输出被编成FP32；用户DDK需独立实查。只允许附加端口的这类提升用于明确标记的诊断读取，记录 `SOURCE_DTYPE_CHANGED` 并保持数值验收FAILED，不把它当源混合精度通过。固定Top1/hidden类型、shape或未知类型改变均停止上传。实际数据、比较、一致性及生产端对照报告位于 `direct-trace/board/comparison.json`、`direct-trace/result.json` 和 `compiled-output-abi.json`。

如只需只读复核旧packed数据，可运行 `python3 "$REPO_DIR/targets/cpro/scripts/run_draft_trace_review.py"`；不编译、不写文件、不访问设备。不再重复运行原packed trace入口以定位首次算术错误。

最新接口明细已确认：升级后的检查为320/331，20条跨CPU路径已证明；剩余失败源于检查器还比较了其它端口的描述符。用户回传的可见差异位于四维CPU NetOutput输入，dtype/shape相同，仅CPU标NCHW、main标ND；本次追溯的是三维FP32 flatten探针。现将描述符检查精确限定在所追溯路径的每个端口；CPU归属、唯一Data/NetOutput、全接口端口数量和原生产者身份仍严格检查。所选路径若出现FP16、NCHW、shape或来源变化，仍停止上传。没有把ND/NCHW普遍视为相同。

继续使用下方原命令，复用现成OMC。新版错误会列出具体owner/node/port和两侧描述符，并合并重复原因；完整错误仍保存在 `layer-trace/precision-check.json`。

最新trace回传：OMG已编译完成，331个精度要求中319项匹配，12项失败全部是11个诊断分组Concat及最终Concat的 `FP32_REQUIRED_EDGE_CHANGED`，实际生产者为 `SubGraph_*_data`。没有缺失节点或dtype/shape错误；仍未打包或上板。检查器此前只支持同一CPU子图的直接连接，现增加严格跨CPUCL分区来源证明：有序Data/NetOutput端口必须与main GraphOp完全对应，逐段FP32、ND、相同shape，并精确追溯回原生产者。额外Cast、NPU、不同来源或不明端口仍失败。

更新后继续下方同一条 `run_draft_trace.py` 命令即可。仅检查/控制代码变化时会复用既有reference、fixture、AIR、OMC，保持旧owner及恢复标记；构建依赖、编译选项、输入和DDK仍校验。重新检查通过后才继续打包及上板。无需删除 `layer-trace` 或重新准备权重。失败时终端直接打印明细；可选 `--inspect-only --include-boundaries` 只读查看已保留的分区接口。

2026-10-29 新回传（PID57427）：`fp32-cpu` 编译检查153/153通过，设备执行及输入回读通过。hidden L2恢复为175.270776（参考175.270586），但相对L2仍为1.389480，Top1为15/15不一致。最终归一化幅度已恢复，完整draft数值仍失败；不能据此确认所有设备运算正确。新OMC SHA256为 `bd9f826bccf2b741a50be8d1afb7e9c95b5ef15aa31e79d3feae42160ae32705`。以下为用户日志证据，原远端文件尚未在本地重新核验。

现在使用一次完整逐层对照，复用固定目录的prepared及原fixture，不重复原生产执行：

```bash
git pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/run_draft_trace.py" \
  --tbe-diagnosis "$AI_RUN_DIR/fused-all-f6b81ead7ae8/diagnosis"
```

它在固定 `full-draft/layer-trace` 中重放参考、导出及编译一份诊断图、检查编译精度、流式打包并上板一次。166个探针覆盖输入FC/embedding、六层Q/K/V、mask、QK、softmax、PV、残差和MLP，FP32打包输出35,429,052字节。原生产图、golden和board证据保留。参考hidden/Top1必须逐字节复现原golden；编译精度检查失败时停止上传。终端逐节点打印误差，完整结果在 `layer-trace/board/comparison.json` 和 `layer-trace/result.json`。`first_nonexact`仅表示首次非逐位一致；不等于首次数值错误。还会比较诊断图与原生产图设备hidden/Top1，只有 `production_equivalence.status=EXACT` 才能把该trace直接用于解释原运行。输出探针可能改变编译分区，因此诊断本身不能替代生产图精度验收。

以下保留正常生产入口，当前无需再重复这一轮：

**编译后的RMS链出现了源码未要求的FP16边界。现在重新编译时锁定FP32区域，并在上传前检查实际OMC：**

```bash
git pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/run_full_draft.py" \
  --weight-storage embedded-stream --precision-policy fp32-cpu \
  --tbe-diagnosis "$AI_RUN_DIR/fused-all-f6b81ead7ae8/diagnosis"
```

正常入口默认采用 `fp32-cpu`；已有明确策略则继续继承。它从现有AIR的哈希绑定recipe生成逐节点编译配置，保留prepared、fixture与AIR，只重建compilation和bundle。全部RMS的平方、均值、epsilon、rsqrt、归一化乘积，以及attention score/softmax、RoPE的FP32区域，必须在编译后的CPU子图中保持FP32。源码规定的FP16矩阵乘和Cast边界不变。该策略先建立正确性基线；速度需在数值通过后重新测量。

`compilation/precision-check.json` 检查每个受保护算子的CPUCL归属、端口shape/dtype和FP32连线；缺失节点、未识别的融合或额外降精度都会阻止打包/上传。配置使用DDK已存在的分组选择接口排除默认FP16实现，implementation编号7代表custom AscendC，并非CPU编号；最终以OMC实际CPU归属作为硬门槛。若所选DDK没有按该要求编译，不能用配置被接受或OMG退出码0代替检查通过。

本地真实编译已验证两组图：双RMS的10个FP32运算全部通过检查，中间的FP16运算仍保留在NPU；两层小draft的61个受保护节点也全部通过，覆盖RMS、attention和RoPE。小draft为适配本地公开DDK使用了记录在案的等价接口替换，没有修改生产AIR导出器。这些是本地同族DDK的编译检查，尚不是用户服务器完整35B draft的板端数值结果。当前最新设备数值状态仍为FAILED。

此前用户回传的PID54002运行已使用v6 AIR和v3参考，设备退出码0、输入回读一致、输出均有限；15个Top1仍有12个不同。参考hidden L2为175.270586，设备为27883.188584，相对L2为159.063547，设备范围为[-258.75, 258.75]。这是明显数值错误，不能用正常FP16舍入解释。以下远端结果来自用户回传，尚未在本地主机重新核验原文件。

用户随后回传逐行诊断：全部15行RMS约150–162，余弦相似度约-0.026至0.136；±257.75分别出现1019/1031次。执行bundle与AIR/编译报告绑定一致，但没有 `dump_graph`。上一版入口仅查IR dump，未检查现成OMC；现已补上这一缺口，同一条命令改为直接检查已保留的小external OMC的ModelDef、明确归属的CPU GraphDef及模型输出描述。

该入口通过packing receipt核对外置源OMC与已执行内嵌OMC的图/任务分区字节身份，读取所选DDK的静态schema，不加载执行库。NPU任务体保持opaque，只展示可验证的GraphOp边界；相关编译日志按清单哈希核验后提取最终norm节点上下文。同时检查原生输出分配字节数、FP16奇偶半字及同字节FP32假设视图，帮助排除输出解释问题。这些观察不改变原FP16比较，也不把±257.75直接认定为芯片饱和。

终端输出关键结果，完整报告固定写入 `full-draft/numerical-diagnostic/report.json`。它不重编译、不上传、不执行设备、不复制大权重，也不修改golden或放宽比较标准。仍有数值差异时退出码为1；报告中的INSPECTED只表示诊断完成。schema、分区或任务格式不支持时明确显示原因，不猜测类型编号。

此前离线重放逐字节复现了旧golden：首次塌缩位于 `draft.layers.0.post_attention_layernorm`，输入绝对最大值340、FP16平方出现27个非有限元素，9行变成全零。仅将RMS改为FP32后所有26个norm均无新零行，hidden恢复有限非零；与旧板端仍有12个Top1差异，hidden相对L2=1.9395076956574235。

最终norm的2,048个scale均为1。正确FP32归一化后的30,720个hidden元素L2约为sqrt(30720)=175.271218；新参考175.271140符合，旧板端357.492527明显不符合。这确认旧设备最终输出不满足该归一化约束；具体设备FP16溢出如何处理尚未观察，不将饱和机制猜测当作事实。

对齐依据为4B `feature/gdr-chunk-verify` 的固定提交 `90186732d915e8b058b2a053d205991d805eae76`：全部26处RMS在FP32中归一化、转FP16后乘原scale；attention保持FP16 QK/PV矩阵乘结果边界，score缩放、mask和softmax使用FP32，概率转FP16后进入PV；RoPE角度和cos/sin使用FP32，完成后转FP16。内部矩阵乘累加精度不能仅从Cast推断。35B的权重、六层拓扑、输入和完整词表保持原样。

与35B原版的区别已单独核验：[官方固定版本](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash/blob/52cb554b4995dede3e2e1bdb129cdb1f3529332b/README.md)为BF16、SGLang/FA4；本次是FP16部署候选。RMS/RoPE的计算顺序符合作者参考源码，但不能将显式FP16 QK/PV边界视为SDPA/FA4内部数值复现，源模型质量验证仍需独立完成。

此前v6源码精度升级会在固定 `full-draft` 目录中保留prepared权重，替换旧fixture、AIR、编译与bundle，并验证输入字节哈希不变。用户已完成该升级；本次编译精度锁只重建编译与bundle，采用已经执行成功的embedded-stream封装，无需原始target checkpoint。旧 `reference-replay` 小报告保留供诊断。

GE导出契约仍为 `cpro-fileconstant-relative-int-fc-conv2d-mask-gather-air4b-precision-v6`，参考契约仍为 `pytorch-fp16-air4b-precision-v3`；本次只增加编译约束，不重新生成golden。用户回传OMC已确认FP32 ReduceMean前后存在FP16边界，输出hidden的FP16类型和61,440字节分配相符。具体芯片内部饱和机制仍未确认。`--compare-only`仍可复核当前板端结果；`--diagnose-numerics`保留只读诊断功能，不必再重复本轮检查。

[独立 draft 指南](draft-standalone.md)保留完整模型入口。它只读取draft全部权重及target的embedding/LM head，不构造或运行40层target。默认完整6层、block16、248320词表，输出15个候选ID和pre-head hidden诊断值。使用固定合成target features，因此这一步不证明语言质量、接受率或加速。

最新内嵌重试 **`full-draft-6599025a0b74`** 在 `Compile end` 后保存模型失败：`SaveToBuffer/CreateCompiledModelBuffer` 中的 `LargeMemoryOps::Copy` 报 `memcpy_s` 失败，经 `SetModelHeader/CopyDataToBuffer` 返回保存失败。它没有进入板端执行；目前只能确认大缓冲区序列化失败，不能确定是某个固定大小上限或其他原因，也不会自动改回external重跑。

按用户要求，入口现在默认使用固定 **`$AI_RUN_DIR/full-draft`**。此前一次迁移命令如下；已经完成迁移时不再执行它。它从732迁入已完成权重、参考、v4 AIR，以及已经成功编译的external OMC和bundle，不编译或上板：

```bash
git pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/run_full_draft.py" \
  --resume "$AI_RUN_DIR/full-draft-732b26335347" \
  --prepare-only
```

迁移采用同一文件系统内的目录重命名，不复制大权重；保留原清单字节及哈希，并将组件目录改为固定目录下的路径。从732继承external模式，保留现有成功编译和bundle；6599的embedded保存失败仍作为最新故障记录。成功后后续命令省略 `--resume`，自动从固定目录继续。成功阶段复用；失败或失效阶段及旧board输出会在原位置清理重建，最新日志和报告覆盖旧版，不再默认生成新批次目录。并发运行会被锁拒绝，中断的迁移通过日志恢复。迁入完成且组件均已位于固定目录后，旧批次目录可按用户选择清理；入口不自动删除整个旧目录，也不删除独立TBE诊断目录。固定目录仍需容纳一次编译的输入暂存和输出，不能替代释放已满磁盘空间。

此前 **`full-draft-732b26335347`** 的external模式已完成v4 AIR编译和打包。PID46426日志显示CPUCL初始化成功，越过 `v501/BroadcastTo` 障碍；随后NPU `SubGraph_1` 在 `InitRuntimeParams` 报 `weightsMemBaseSize=16`、`modelWeightMemorySize=1151486976`。尚无Process或数值结果。6599继续复用该v4 AIR和 `d7b4db958850` prepared/fixture，本次迁移不改变图、权重、参考或加载行为。

运行时给15个不同子图打印了同一个 `weight hash:0x1dabbe2881ec5472`，但它不是文件SHA256，不能据此断言15份权重内容相同或确定错配原因。最新报告已回传实际文件SHA256清单。本仓库只读审查未发现按编号重排或丢失目录：编译到bundle再到上传均按完整相对路径处理；这仍不能证明设备HIAI打开了这些路径。

用户随后列出的编译文件中，`SubGraph_1.weight` 是 **1,151,486,976字节**，与模型要求一致，`model.omc` 是2,905,990字节；因此这份编译文件本身不是16字节。文件大小不能替代SHA256或证明运行时选中了正确文件。runner的“路径→buffer”调用会丢失路径，已由本次getter观察确认；外置权重文件的实际选择和加载契约仍待核对，见[验证记录](validation.md#外置模型加载描述符诊断入口)。此前成功tiny没有开启外置权重选项，不能直接验证这条加载契约；V2文件恢复也已复现该故障，目前没有已确认的运行时修复。

原生失败保存在 `board/results/result.json`、`board/comparison.json` 与 `board/draft.stderr`。错误回传已修正：保留原生失败原因，单独记录传输/证据问题，运行失败时不要求成功输出bin。此前 `a091f16d13d8` 的嵌套OMC输出路径问题已修复。

4B和35B draft均为六层DFlash、block16、前五层sliding attention加最后一层full attention，但checkpoint配置不同：hidden为2560/2048，MLP为9216/6144，target feature宽度为20480/16384，embedding/head分别共享/不共享。不能替换35B的训练宽度或权重。使用同一小配置、权重及公共运算测试两套结构，三组prefix的完整六层hidden逐字节相同。4B分支记录20.22 ms/次（310P、C16/C64混合均值），同时标注PASS_WITH_DIFFERENCES；这个速度不证明全部数值一致。合并投影与增量缓存属于后续优化，Cpro性能需重新测量。

## target 并行窗口：原量化模型生成与验证

target窗口直接推进原40层量化AIR、89输入/80 RefData及64/1动态档，目标是实际普通prefill→decode与数值比较。draft由另一窗口继续。用户已确认原AIR及同级权重仍在、旧日志已删除；64档原始输入位于服务器 `/home/w00949577/run_35B/chunksize_64_data`，目前没有golden。代码在 `target/cpro-dflash-target` 分支，基于 `main` 的 `54f9ffd`；后续与draft分支合并。

更新后的入口在启动前检查同源target入口及有输入清单绑定的OMG进程，并按原AIR路径加锁；不会阻止另一份draft编译。完整编译后从所选DDK的schema静态解码小OMC根图，核对真实Data/RefData名称、形状、类型、序列化index及输出生产者，另保留原工程输入/状态构造与所选HIAI头文件的本地摘录。不会导入原工程、重做量化或传送内部源码/库。

已按用户原运行脚本接入板端 `model_run_tool_internal`，使用HIAI V2、`--weight_dir`、`--times=1` 和实际编译接口。原脚本的部分arg名称来自较早导出（例如`arg112_1/arg106_1`），不能原样套用本次`arg113_1/arg107_1`；先检查实际schema、索引、shape及dtype，再构造参数。原状态模板在板端展开成80份独立输入文件，保留原字节，不清零或改写原文件。

原服务器在已配置环境中执行以下命令；使用独立源码worktree以保留draft窗口当前checkout。若`${REPO_DIR}-target`已经建立，复用它，不重复添加；`full-target`必须是新任务目录。入口发现同源target编译任务会拒绝重复启动：

```bash
git -C "$REPO_DIR" fetch origin target/cpro-dflash-target
git -C "$REPO_DIR" worktree add --detach "${REPO_DIR}-target" origin/target/cpro-dflash-target
python3 "${REPO_DIR}-target/targets/cpro/scripts/compile_full_target.py" \
  --air /home/l00886453/qwen3_5_moe/glm_export_air_cpro_40block_55_quant/export.air \
  --source-root /home/l00886453/qwen3_5_moe \
  --tbe-diagnosis "$AI_RUN_DIR/fused-all-f6b81ead7ae8/diagnosis" \
  --output "$AI_RUN_DIR/full-target" \
  --run-board --sequence-length 64 \
  --input-dir /home/w00949577/run_35B/chunksize_64_data
```

关键产物为 `full-target/result.json`、`compilation/compile.log`、`target-contract/report.json`、`runtime-sources/report.txt`、`board/result.json`、`board/tool.stdout`、`board/tool.stderr` 和 `board/outputs/`。实际OMC接口或schema不支持时单独报告UNAVAILABLE，编译结果仍独立保留。`MATCHED_VISIBLE_BOUNDARIES`仅表示根图可见名称/类型/形状符合声明；序列化index不等于已验证的HIAI输入顺序，当前源文件也不能单独证明与当初导出AIR同版。板端入口校验输入字节数、dtype/档位及81个输出描述；完整全局index若与用户有序映射矛盾则停止。RefData单独编号或index缺失时保留原观察，使用用户脚本明确给出的有序参数，不将这些序列化index当作HIAI runtime槽位；最终仍需同输入数值参考确认。

上传保留外置权重层级，按编译身份建立模型缓存，只重传哈希不同或缺失的文件；每次执行使用独立目录，保留输出、exitcode和工具日志。状态输入为各槽位独立副本，不推断底层RefData内存绑定已正确。相同模型执行有板端互斥锁；HDC等待超时不代表远端已取消，锁未释放时拒绝重叠执行。没有golden时状态为`EXECUTED_UNVALIDATED / NOT_RUN_NO_REFERENCE`，不会把执行、有限值或输出文件数量当精度通过。工具内部warmup/实际Run次数尚未核验，不作速度结论。

用户原输出契约只有`logits + 80 RefData`，没有8层、宽16384的DFlash target features。该普通target是首轮基线；后续仍需补目标特征导出、同输入数值参考、64→1状态接续及DFlash验证/回滚。

已有**完整COMPILED任务**后，继续检查用下面一条命令。它流式核对原文件、输入快照、全部生成文件、OMG及日志SHA256，保留外置文件层级，不复制权重、不重新编译模型；新增小报告放在 `continuations/`，原报告不覆盖：

```bash
python3 "${REPO_DIR}-target/targets/cpro/scripts/compile_full_target.py" \
  --resume "$AI_RUN_DIR/full-target" --inspect-only \
  --source-root /home/l00886453/qwen3_5_moe
```

编译完成后只重试板端时，将上面的`--inspect-only`换为`--run-board --sequence-length 64 --input-dir /home/w00949577/run_35B/chunksize_64_data`，仍用`--resume`，不重启整模编译。

失败/中断编译、源权重改变、DDK/参数改变、外置文件缺失/改写/新增均拒绝复用，不自动删除或重编译。不要把残留OMC当成已成功任务。编译成功但板端失败时仍保留COMPILED阶段，`board_status`单独为FAILED、入口退出码1；可复用已验证编译后处理该板端失败。主机验证见 `specs/original-target-workflow.json`；真实公开DDK小OMC读取也通过，并正确识别其不符合89输入声明。这些不是内部DDK整模编译或板端通过证据。target仍待原服务器首轮结果。

以下为保留的历史编译问题与诊断说明：

以下为原编译入口及既有诊断的历史记录。用户最新已说明旧 full-target 产物删除，并另开窗口推进 target；不能据本节早期“正在编译”的描述推断当前进程或产物仍在。

**已经在编译的任务先保留，用另一个终端观察。** 旧脚本在 `Compiling existing AIR` 后将输出写入 `compilation/compile.log`，直到 OMG 退出才刷新终端；默认7200秒只是超时上限，不是预计完成时间。十几分钟没有终端输出不能单独判定卡死。查看本次目录里的日志尾部、文件更新时间以及 OMG/编译子进程的 CPU 累计耗时、内存和等待状态；日志静默也可能来自缓冲。先定位当前阶段，不重复启动另一个整模编译。

新入口等待期间每30秒显示已用时间、日志字节数、最近写入时间距今秒数及有长度限制的最后一行。这是等待观察，不是完成百分比或编译正常的判定；监控失败不会重启、终止或改写编译结果。它只影响以后启动的任务，正在运行的旧进程继续通过日志观察。

**上一轮外置权重暂存问题已修正。** `full-target-compile-e019e4a7ad53` 已通过输出参数校验并开始解析 AIR，在 `arg3_1` 的 `ConvertFileConstant` 处找不到 `./glm_export_air_cpro_40block_55_quant/model_model_layers_0_input_layernorm_weight`。用户确认这些权重就在原 `export.air` 同级目录；旧收集逻辑漏掉无扩展名文件，暂存布局也缺少引用里的导出目录前缀。

入口现在收集同级无扩展名的普通文件，保持 AIR 位于编译工作目录，将权重快照放在其下的原导出目录名中。默认 `--input-layout export-directory`；其它 AIR 若使用不带导出目录名的相对路径，可显式选择 `--input-layout air-directory`。不改写 AIR，不借用原目录链接，也不重新转换 YAML 指定的量化权重。以下保留启动命令供后续需要时使用；已有编译运行时不要重复启动：

```bash
git pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/compile_full_target.py" \
  --air /home/l00886453/qwen3_5_moe/glm_export_air_cpro_40block_55_quant/export.air \
  --tbe-diagnosis "$AI_RUN_DIR/fused-all-f6b81ead7ae8/diagnosis"
```

脚本内置用户提供的89个输入形状、80个`RefData`状态输入和5个动态轴，使用 `--dynamic_dims="64,64,64,64,64;1,1,1,1,1"`，保留 `--limit_partition_size=false` 和 `--save_weights_as_external_data=true`。原命令中的 `--d11ynamic_dims` 按拼写错误处理；执行前记录所选OMG帮助。帮助未列出的内部参数仍按用户命令原样交给实际编译器验证，不偷偷删参数。默认调用当前DDK的 `tools/tools_omg/master/omg` 实际程序，沿用已经成功的隔离环境，避免外层脚本修改共享工具目录；不添加FP16权重强制转换或探针的布局选项。

前一批次 `full-target-compile-771648b65da5` 的 `CheckOutputPathValid` 错误已通过短名称 `--output=model` 绕过，最新日志确认已进入 AIR 解析。本次权重修复继续保持短输出名。编译前会显示所选文件数量、总字节和权重相对前缀；`input-inventory.json` 记录快照哈希，输入文件不会被当作新生成 OMC 外置数据。文件收集仍有目录/数量限制，完整依赖尚需实际编译验证。

失败时入口直接打印有长度限制的日志摘要；完整异常和参数保留在 `result.json`、`compilation/compile-command.json`。帮助阶段失败显示 `help.log`，无需为了查看错误重新运行工具。

每次输出到新的 `full-target-compile-*`，保留 `compilation/compile.log`、完整命令、输入快照、OMC及生成的外置数据文件。编译成功后尝试读取编译图接口，报告在 `interface/report.txt`；工具不支持或转换失败单独报告，不改写编译成功状态。文件名和用户形状声明不等于实际AIR验证。原AIR保持原图，**本仓库基础Conv/RMS替代不会自动修改这份AIR**；以真实编译及图检查结果确认节点、状态和量化接口。

这条入口开始整模编译，不自动套用探针runner。原AIR的动态档位、输入顺序/dtype、RefData持久状态和外置权重装载还需与实际HIAI接口对齐后才能上板。

**三填充诊断已完成，不需要再续跑或重跑这批探针。** 用户回传 `gdn-diagnostic-138b60d3eba7` 为 `DIAGNOSIS_COMPLETE`：原 `gdn-diagnostic-c0cbac5ac9cf` 的零填充结果复核完成，另两轮执行完成，三轮host输入回读SHA256一致。保留这两个诊断目录和原 `fused-all-f6b81ead7ae8` 批次，后续直接分析已有证据。

同一OMC/输入的GDN输出随预填充值变化。零填充时prefill/decode核心relative L2约为0.00392/0.0564，末态约0.146/0.155；A5填充时核心升至3.44/11.19；5A填充时出现约203.25的误差，与FP16位模式 `0x5A5A` 的值一致。有效token区域已排除padding。当前证据不能区分部分写、输出累加和缓存可见性；清零不是已验证修复，数值验收仍未通过。上述结果来自用户报告，原始远端文件尚未在本地重新核验。

target当前里程碑为OMC编译与一次普通prefill/decode，与独立draft同时推进。现成AIR路线绕过当前直接GE的W4A8重建缺口，但不能据此宣称整模已可上板。作为备选，新增加的权重准备器可流式生成完整target的逻辑INT4/INT8/FP16缓存，支持中断后校验复用；其约18.1GB数据尚非I4ZN/QPARAM，也不是板端峰值内存。原量化权重的 PyTorch/GE 路径现已接入，详见[原量化模型接入](original-quantization.md)；真实DDK量化绑定和QPARAM转换仍需实际环境提供。FP16约69.3GB不适合32GB板端。`--expert-dispatch selected`只减少计算，不减少常驻权重。现有自建stateful图的 `past_length` 静态；原AIR的动态状态契约需要独立检查。

target后续工作：按用户原转换契约完成现有AIR编译、读取实际图接口、接通状态执行；继续分析GDN读写问题。暂不增加Conv/RMS候选、重复平台诊断或新的性能矩阵。需要实机验证时给单一入口命令。

模型范围仍保留GDN、CacheUpdate、Attention三类六例，Conv/RMS使用基础算子。只有确需重新执行全部六例时才使用 `run_fused_probes.py --model-ops`；旧九例 `--all-ops` 保留为可选诊断，原Conv/RMS失败不改为通过。

原工程中 `LinearAttentionConvstateUpdate` 用于Cpro，`AdnRmsNorm` 用于ND，Cpro的RMS调用内置 `npu_rms_norm`。本仓库当前选择现有基础算子实现Conv历史拼接/截取和RMS，生产GE只启用 `--fused-ops gdn`。CacheUpdate/Attention的独立探针通过尚不等于已接入生产GE模型；完整模型及数值验证继续保留。

若再次进入Docker，恢复conda/DDK后执行 `source targets/cpro/scripts/env.sh` 即可；配置含自动生成的HIAI头文件及三库路径，详见[环境恢复指南](environment-setup.md#docker-重进后的配置恢复)。上述命令要求保留当前批次，保留权重无需重新准备。

## 当前状态

| 项目 | 状态 |
|---|---|
| PyTorch完整结构 | 原量化 target 按 Cpro forward 对齐 FP32 residual/门控/RoPE；新增原风格 embedding/位置选择入口，功能式状态接续通过主机测试 |
| GE完整结构 | 原W4量化target完整40层64/1构图；同步 FP32 计算边界与指定位置完整 logits，实际DDK量化绑定和设备验证待完成 |
| 自定义算子探针 | 当前选择GDN/CacheUpdate/Attention三类6例，用户回传导出、编译、加载及执行通过；原Conv/RMS三例失败保留为可选诊断 |
| 真实权重AIR导出 | 旧双图支持完整FP16；新GE target支持显式缓存 |
| 完整 draft | 原 direct 已执行但数值失败；当前 residual-fix 候选待165项CPU检查及板端验证 |
| 原工程完整AIR | 旧 full-target 产物已删除，另一窗口推进原量化40层编译与普通prefill/decode；尚无本页确认的目标设备数值通过 |
| 原量化缓存 | 按原YAML读取已有W4及scale，I4ZN逐字节参考对照通过；专家QPARAM缺失时明确报告pending，不重新量化 |
| 状态执行 | 主机fake prefill→decode接续通过；真实板端待验证 |
| 尚未完成 | 真实DDK量化契约/目标执行、常驻缓存与native DFlash调度、完整模型设备精度和性能 |

## 已确认与剩余问题

| 项目 | 结论 |
|---|---|
| GDN两例 | 三填充诊断已完成，输出依赖预填充值；零填充改善但末态relL2仍约0.146/0.155，未通过精度验收，原因未定位 |
| 编译目标 | 始终为 `kirinx90`；`Ascend310B1` 是DDK初始化后返回的内部名称，`Ascend310P3` 是原工程平台 |
| GDN状态 | 安装版初态/末态为FP16；本次runner修正了末态只接受FP32的旧校验 |
| conv | 当前走基础算子的历史拼接/截取路径；原Cpro自定义Conv两例在rank-4 y2维推后格式配置失败，历史结论不变 |
| RMS | 当前走基础算子RMS路径；原ND用的 `AdnRmsNorm` 探针缺维推函数，保留失败；不再尝试其他融合RMS候选 |
| CacheUpdate | prefill/decode调用通过，比较仅为SHAPE_ONLY_UNVERIFIED_CONTRACT，无数值精度通过结论 |
| Attention | prefill/decode调用通过，比较仅为SHAPE_ONLY_UNVERIFIED_CONTRACT；先前导出器问题不再阻止调用，精度仍未验证 |
| 尚未验证 | 当前六例的正确数值及NPU执行位置；CacheUpdate/Attention生产GE接入、基础Conv/RMS的完整模型目标验证与连续状态验证仍待完成 |

新增调用源码的[核对与勘误](qwen35-moe-operators-and-quantization-analysis.md#调用源码核对与勘误2026-10-27)确认：Attention converter把Python `pse_shift`实参接到GE端口16，端口3留空，当前探针方向正确；Conv的输出rank、GDN的Meta/安装版状态dtype和Cache的Functionalize调用名仍有差异，不能凭前端声明判定安装版行为。

## 完整模型构图（按需）

需要完整模型结构时选择下面一条路径；`sequence-length=8`只限定输入长度，模型层数、专家和词表均为完整尺寸。

**PyTorch：** 生成完整meta模型和层级FX结构，不需要检查点或TorchAir。

```bash
python3 -m qwen35_moe_dflash.cli build-full-pytorch \
  --sequence-length 8 --output "$AI_RUN_DIR/torch-full-$(date +%Y%m%d-%H%M%S)"
```

**GE：** 生成完整target/draft构图源码，仅选择GDN融合，不需要检查点或GE库。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/direct_model_air.py" build-full-structure \
  --sequence-length 8 --expert-dispatch selected --fused-ops gdn \
  --bindings "$AI_RUN_DIR/fused-all-f6b81ead7ae8/bindings.json" \
  --output "$AI_RUN_DIR/ge-full-$(date +%Y%m%d-%H%M%S)"
```

这些是未绑定真实权重的结构产物。当前生产GE融合接口支持 `gdn,conv_state,rms` 三个角色，本次只选择 `gdn`，Conv状态和RMS保留基础算子。CacheUpdate/Attention尚未通过该接口接入生产图。实际导出AIR：按 [PyTorch](pytorch-guide.md)或 [GE](ge-guide.md)指南继续；无需重跑旧探针。

## 文档融合接口

本次fresh批次的 `bindings.json` 可复用于GE融合构图，需要时在构图命令前设置其路径。绑定包含文档端口与安装源码的GDN状态声明，不代表全部kernel已安装或已在目标芯片验证；融合构图命令见[GE指南](ge-guide.md)。

完整W4A8 + FP16 DFlash先按24–28GiB可用RAM规划；这不是实测峰值，也不适用于当前FP16权重导出，见[内存预算](environment-setup.md#内存预算文档-w4a8--dflash)。
