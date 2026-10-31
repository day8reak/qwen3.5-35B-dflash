<a id="resident-mixed-first-device-completion"></a>

> 历史记录：保留当时的观察、失败和候选命令。当前操作请从[文档首页](../README.md)或[当前状态](../current-status.md)进入；不要直接重复本页旧请求。


## 2026-10-31：混合接口常驻完整板端请求执行成功

用户回传 `dflash-board-resident-mixed8-recovery`，PID5008、prefill1／decode8／draft7、target/draft各加载一次并正常关闭、锁释放，token为 `[4,0,28,18,2,2,2,0]`，接受0／7，总墙钟88.680秒。单次功能及常驻执行通过；源码/模型原始artifact未由助手连接设备重取，compact不包含完整状态链重放。精度、重复稳定性、独立权重释放和配对算法性能未验收。

与旧逐次加载总墙钟305.788秒的单次记录比为3.448，但phase计时范围及token路径不同，不作为DFlash算法加速结论。新旧第2个token为0／12；源码调度证明该值在首次draft Process之前由首decode算出，优先隔离prefill输出/首decode输入与target执行差异。draft已加载，共存影响仍可能需要单独对照。详情及原始内存遥测限制见[结果](board-resident-mixed-first-completion.md)和[结构化观察](../../specs/board-resident-mixed-first-completion.json)。

<a id="resident-mixed-candidate"></a>

## 2026-10-31：用户授权 V2 target／V1 draft 混合常驻候选

在当前HCL的V1 path-only失败及工具头文件调查之后，用户明确回复“允许混合接口，优先推进常驻”。新 `--resident` 准备独立 `resident-mixed` 请求：本进程V2 prefill64执行一次并关闭，随后V2 decode1与V1完整embedded draft各加载一次并反复交替运行。绑定三份原OMC、权重和独立prefill／decode named ABI；未编译模型或改变算法／控制。旧V1、V2、逐次进程请求仍按原策略恢复，失败不会自动换接口。

V2初始化保持已有设备Init→InitWeights均返回0的顺序。新请求在加载前固定 `ManagerDeinitOnly` 关闭策略，DeInit后销毁manager，再释放built model和tensor；它不调用单独DeinitWeights，不将未调用的返回码伪造成0，也不吞掉历史1／3错误。DeInit的called／returned分别记录；异常仍清理并失败，不重复关闭。官方[V2模型管家索引](https://developer.huawei.com/consumer/cn/doc/hiai-References/model-manager-class-0000001139115907)只确认DeInit是公开接口，未取得证明私有InitWeights资源所有权的正文。因此保留 `external_weight_release=NOT_INDEPENDENTLY_VALIDATED_MANAGER_DEINIT_ONLY`，不宣称外置权重释放已修复或实机验证。

新native目标和每模型API随manifest／config／返回摘要绑定；compact及dump均校验实际API计数和关闭事实。新增prefill加载前后及关闭后的RSS/HWM／MemAvailable快照，均不是专用NPU显存或释放证明。主机fake验证持有期、调度、跨模型buffer重叠拒绝和失败传播，不能证明实际DDK加载、模型数值或性能。具体测试及源码快照见[混合候选验证](../../specs/resident-mixed-host-validation.json)。

本轮186项Python单元测试通过，18组native CTest通过；完整C++ mixed runner生成的7个唯一场景已通过主机验证。首次联测有两项在新增恢复测试夹具中缺AI_RUN_DIR，补齐后仅补跑对应两个场景并通过，原5通过／2失败的XML与补跑2通过的XML均保留。两项恢复实际执行本地文件系统上的shell锁／PID／哈希检查，未使用设备HDC；成功回收及生成完成后关闭失败均不重启或上传。实际内部SDK编译、加载和双模型常驻留待用户完整板端请求。

<a id="resident-v1-hcl-file-load-failure"></a>

## 2026-10-31：原40层 V1 FILE_ONLY 在 HCL Load 被空buffer拒绝

用户回传 `dflash-board-resident-v1-file8-recovery`：运行库 `800.631.120.010`、PID63072、阶段 `prefill-v1-load`、SDK Load返回1；prefill／decode／draft Process计数全0，token为空，锁已释放。板端attempt为 `/data/local/tmp/qwen35_board_serial_60724b33b72e47aab3558ae0b1ec2d09`。进程stdout显示主OMC启动哈希已完成，随后进入prefill加载；HiLog显示 `DDK pipe is HCL`，`CreateLocalBuffer` 要求 `data != nullptr && size > 0`，随后报 `CreateLocalBuffer return null, load failed`。

因此，本次选定HCL分支确实拒绝了path-only描述符的空buffer／零长度，失败发生在OMC恢复、外置权重解析与推理之前。不能归因为输入mask、80状态、精度或常驻内存不足；`libffrt_acc.so` warning不是直接报出的失败点。此结论仅针对本次运行库分支，不宣称所有V1实现均不支持文件或外置权重。设备日志时钟为用户原样提供的03-13，不改写为主机记录日期。证据为用户粘贴，助手未连接设备独立重取。

此前4B path-only示例和可接受文件描述符的fake SDK只能支持实验和调度测试，不能替代所选运行库验证。官方旧buffer示例不包含当前外置权重合同；当前没有证据支持通过添加builder、交换互斥setter或猜测后端开关修复。需要核对实际选中头文件／受支持入口。原18.8GB旁置权重不能直接塞进现有UINT32长度的主模型buffer，但这个限制本身不证明所有V1扩展均不可用。失败候选命令已从当前计划撤下，模型加载代码尚未改动。

本轮修正恢复摘要，直接输出验证后的 `prefill_lifecycle`／`prefill_failure_stage`；未加载resident会话的正常 `CLOSED` 不再冒充失败阶段，真实resident失败和清理错误继续保留。新增独立只读日志脚本，绑定同一已完成失败请求及其原设备、配置和退出回执，收集stdout／stderr末尾各64KiB，可选当前PID过滤HiLog快照；不加载模型、不改锁、不读权重。HiLog允许轮转和PID复用，只作诊断快照。本次HiLog已经足够，无需用户再采集。

132项transport测试和26项日志收集测试通过，包含真实shell的tail失败传播检查；均为主机测试，没有实机加载／推理、模型编译或新精度结论。源哈希、JUnit及本次设备观察见[本轮记录](../../specs/resident-v1-hcl-load-failure.json)。旧板端原prefill完整循环成功继续保留，V1常驻仍未通过。

<a id="resident-v1-file-target-candidate"></a>

## 2026-10-30：显式 FILE_ONLY target／prefill 与 BUFFER_ONLY draft 候选

以下为实验前的历史决策；该候选已由上方2026-10-31用户回传判为失败，不再作为当前执行命令。

用户授权在不修改三个OMC及权重的条件下直接继续完整板端实验，不再插入load-only试验。本次依据其提供的4B V1 `model_run_tool` 源码，将原版外置权重target／prefill显式选择为FILE_ONLY：只设置并检查完整绝对模型路径，再调用V1 Load；embedded draft显式选择为BUFFER_ONLY，保留完整模型buffer及其生命周期。各分支只验证对应表示，不再要求path和buffer共存。失败停止，不换setter顺序、不自动改用buffer或V2、不重新编译模型。新配置策略为 `legacy-v1-file-target-buffer-draft-v1`，session API记为 `legacy-v1-mixed`；prefill／target分别记为 `legacy-file`，draft记为 `legacy-buffer`。FILE_ONLY要求getter中的path精确、buffer为空、长度0；不把这些描述符条件当成SDK加载通过。

这是一条已获授权但尚未实机验证的加载候选。历史 `bf545c2` 属于10月23日tiny target／draft bring-up，其path-only HCL空buffer错误不能直接证明当前原版40层prefill／decode OMC的同一结果；现有记录不足以断言两次运行库身份相同或不同。用户提供的path-only、OFFLINE（字面值1）代码支持选择这条实验路径，并不构成35B外置权重Load／Process成功证据。选定SDK头文件／实现可继续帮助解释失败，但不是开展本次完整实验的必要前置。

官方HiAI-Foundation的旧buffer示例、用户提供的V1工具，以及公开4B项目必须分开：后者在固定提交 `90186732d915e8b058b2a053d205991d805eae76` 的[README](https://github.com/day8reak/qwen3.5-4B-dflash/blob/90186732d915e8b058b2a053d205991d805eae76/README.md#L3)明确使用Ascend310P；其[实际C++加载器](https://github.com/day8reak/qwen3.5-4B-dflash/blob/90186732d915e8b058b2a053d205991d805eae76/framework/runtime/cpp/src/acl_executor.cpp#L146)调用AscendCL `aclmdlLoadFromFile`，没有建立所选Kirin SDK的V1外置权重加载合同。

新请求使用 `dflash-board-resident-v1-file8-run`，回收使用 `dflash-board-resident-v1-file8-recovery`；旧707d825／50184b4失败与测试记录不覆盖。按[当前命令](../current-status.md#用户现在执行什么)一次启动完整循环，由板端完成prefill、常驻decode／draft和串行verify。实际SDK加载、89／82运行时接口、完整状态链与Process结果决定执行结论；未回传结果前不标记Load PASS、数值通过或性能收益。

<a id="resident-v1-exclusive-setters-failure"></a>

## 2026-10-30：V1 prefill 在 SDK Load 前被互斥描述符阻断

用户回传 `707d825` 的完整板端请求在 `prefill-v1-load` 失败，原始错误为 `SetModelBuffer did not preserve exact model path`。SDK Load／Process 调用数均为0，生成token为0，锁已释放。该结果来自用户回传，未由本地重新连接设备核验；它证明加载前检查失败，不构成外置权重加载、推理、数值或常驻成功。旧版原prefill完整闭环的成功记录仍保留，但不能代替新V1验收。

这次遗漏的是已有实机证据，而不是新发现的SDK约束。本页此前的[外置模型加载描述符诊断](#外置模型加载描述符诊断入口)及 `specs/model-workflow.json` 已记录：选定SDK中path→buffer会清空path，buffer→path会清空buffer，四次setter返回值均为0。新V1代码仍要求设置buffer后保留同一绝对path；新增fake默认又允许二者共存，因此主机集成通过未暴露这个必然冲突。已有主机测试对调度、状态字节链和生命周期的覆盖仍有效，但其结果不能证明该SDK的外置权重加载方式可用。

`50184b4` 的修复范围是把内嵌权重draft恢复为buffer-only加载，并将fake默认行为改成互斥；需要模拟双字段的调度测试必须显式标注该假设。该版本仍保留外置target的路径保留检查，当时完整V1命令暂停，未形成新的实机通过记录。后续用户授权的显式FILE_ONLY候选见本页首节；旧失败、旧attempt和该版本测试记录保持。

官方来源对照采用固定提交 `83320362c10f2e85a2a1bbb283b3359db204ab2a`：[Huawei HiAI-Foundation 的 classify_sync_jni.cpp](https://github.com/huaweicodelabs/HiAI-Foundation/blob/83320362c10f2e85a2a1bbb283b3359db204ab2a/app/src/main/jni/classify_sync_jni.cpp#L61-L103)先由 `AiModelBuilder::InputMemBufferCreate(modelPath)` 创建buffer，再使用 `SetModelBuffer` 和 `HIAI_MODELTYPE_ONLINE` 调用Load。它没有示范同时保留path与buffer，也未建立当前35B外置权重OMC的解析合同。用户提供的4B代码则为path-only、`OFFLINE`（字面值1），调用方式不同；没有当前35B在选定SDK上的成功加载证据，不能将任一示例直接当作兼容性结论。

`50184b4` 主机验证135项元数据／恢复测试、2项实际C++fake集成以及7组native CTest通过，证据见[描述符回归复核](../../specs/resident-v1-descriptor-review.json)。默认测试复现实机互斥setter行为，验证内嵌buffer加载成功及外置target在SDK Load前明确拒绝；显式双字段模拟只覆盖调度，不代表所选SDK支持外置加载。未访问板端、重新编译OMC或变更权重。

文件／buffer分流候选完成主机验证：191项Python测试（158项准备恢复、24项构建/API、9项实际C++fake集成）及13组native CTest通过。完整debug和compact循环、文件Load失败、描述符路径丢失、旧策略恢复与禁止重启均覆盖；没有真实设备执行或模型重编译。源哈希和JUnit记录见[当前候选验证](../../specs/resident-v1-file-buffer-candidate.json)。

## 2026-10-29：完整 residual-fix 的单例 Top1 匹配

用户回传完整六层、block16、248320词表draft的 `full-draft/residual-fix` 板端报告：EXECUTED、165/165编译精度匹配，原两路输出dtype保持，15/15 Top1与同输入PyTorch参考完全一致。hidden非有限位置差异为0、relative L2=0.0126796675、cosine=0.999919614、max abs=0.08984375。`production_equivalence=DIFFERENT`指它与旧失败模型的设备输出不同，不是本次参考Top1失败。

证据范围是用户粘贴的远端结果，原远端文件尚未在本地重取核验；输入仍为固定合成target features。该例Top1已匹配，hidden尚非逐位一致；未据此宣布真实特征多轮、语言质量、完整模型数值或速度通过。按用户指示先集成真实完整target和这份draft，保留全部后续验收。以下为此前按时间记录的历史检查。

# 验证范围与保留结论

## direct runner 顶层ABI类型解析与原地重试

用户内部DDK的完整direct图编译153/153通过，16个实际输出dtype保持声明，bundle流式打包成功。runner退出1且没有原生结果；stderr为 `qwen35_dflash_hiai: draft diagnostic wire requires its explicit ABI kind`。原正则扫描整个JSON并要求唯一artifact_kind，合法ABI中的嵌套编译输出检查报告也有同名字段，因此在ReadStepPlan阶段、Load/Process之前被错误拒绝。本次未发生新的数值比较，不能据此更新模型精度结论。

改为有界JSON顶层成员读取；保留1MiB上限，深度32、token262144上限，拒绝重复键（含unicode转义后的相同键）、错误JSON、非字符串顶层类型及错误ABI类别。嵌套kind和字符串中的kind文本不影响顶层判断。原native测试只覆盖简化ABI，此次加入完整嵌套direct_output_binding回归。

本地直接使用原 `prepare_draft_bundle` 生成的完整ABI、模型和TSV做旧/新runner对照：ABI SHA256 `f773c126aafd9e3f52892b06d4654a1cce9924a86c0ed821d0cf4ddab0b43257` 不变，旧runner精确复现用户错误，新runner在host fake中EXECUTED并回收全部5个测试端口。证据run `20261029T014455Z-28341-8b01ec/generated-abi/{before,after}.json`。这是生成链路与native适配器的集成回归，不是手机模型数值证据。

控制入口重试只重建runner及board阶段；reference/fixture/AIR/compilation/observed-fixture/bundle均校验后复用。旧失败board先在固定目录内按结果SHA保存，再重试，owner及恢复标记不改。失败终端自动打印SHA已验证的stderr/stdout有界尾部，防止missing result摘要遮蔽真实原因。

本轮组合主机回归263项通过（同run `final-tests/tests.xml`），native CTest8/8通过（`native-json/Testing/Temporary/LastTest.log`）。包含完整direct失败后恢复到board的模拟集成：第二次仅核验编译/输出ABI并重跑board，六个上游manifest字节不变，不导出、不编译、不重打包。用户手机重试及direct数值比较仍待执行。

## packed观察矛盾与原值直接输出

用户回传PID1611：完整packed trace编译精度331/331、跨区证明140条、设备执行及回收通过；Top1/hidden仍失败。`first_nonexact=context_fc`，但该观察FC为零，而projected_context相对L2仅0.004969；层0QK相对L2为5.782891，scaled score却仅0.002022。最强矛盾是同一原Graph Value的 `layer_0_output` 和 `layer_1_input` 观察分别为relative L2=3.275297/174.521739，后者多587个非有限差异；其它层边界也类似。语义→源Value映射、spec offsets和切片主机审查未发现确定错误，这批观测无法作为原计算首次偏离证据。

packed诊断hidden与原生产hidden相对L2=1.402769704876514，Top1差异15，`production_equivalence=DIFFERENT`。诊断hidden自身对参考相对L2=1.3912042027820002。3876个非有限差异来自packed观察，不能直接当作原模型中间值非有限。原生产完整draft数值失败仍保留；没有性能结果。远端证据来自用户粘贴，原文件尚未在本地重取核验。

新增只读 `draft_trace_integrity.py`，验证同源观察、独立Top1端口、残差/score/mask和零输入归一化关系。微小算术舍入差异仅为诊断，不新增通过阈值；明确矛盾将阻止首次根因归因。直接输出共享同源端口时标记alias已去重，不伪装成独立快照。

新增direct路径以原Graph Value作为独立输出，不添加观察Cast、reshape、concat或未经证明的Identity复制。完整配置原594节点不变，默认16物理输出/17语义记录，源声明总1,650,748字节。保留原生产、packed图、CPU参考及其板端证据；在 `full-draft/direct-trace` 单独编译与执行，仍需检查输出一致性及生产终端差异。

本地真实两层小图direct AIR/OMG编译成功，61/61个FP32计算要求通过，但编译OMC把index4/11/14附加源FP16端口提升为FP32（layer0input/postattention/output），形状不变。OMC SHA256 `02ade73f971b825c614e9d4322bedaafb9502c096d39dce21debcb8cbb7717ed`，证据run `20261029T011736Z-26760-00e33d/direct-compiler-smoke/smoke-report.json` 和 `physical-output-abi.json`。这是公开同族DDK观察，不证明用户内部DDK同样变化，也不证明它是此前错误的确定根因。

新reader按所选schema读取实际有序输出，绑定小OMC、编译AIR和声明spec。固定Top1/hidden或shape/count改变均拒绝；仅附加FP16→FP32允许用于显式诊断，以真实dtype收回数据、把参考FP16精确转FP32用于读报，并记录 `SOURCE_DTYPE_CHANGED`，整体数值验收仍FAILED。未修改源数学、权重、golden或比较标准。

本轮组合主机回归460项通过（同run `final-tests/tests.xml`），native CTest8/8通过（`direct-board/native/Testing/Temporary/LastTest.log`）。覆盖原生产/packed ABI拒绝规则、原FP32精度门禁、直接图输出原节点保持、参考复用、实际输出dtype检查、观察fixture、运输/别名/格式及离线一致性。公开DDK小图实际编译61/61检查通过，不构成用户完整direct图或设备精度通过。

## 跨分区来源检查排除无关端口布局标记

用户复用完整trace OMC后为320/331，跨分区精确来源证明20条，仍有11个Concat被拒绝。只读解码进一步显示：返回的差异样例位于CPU NetOutput的 `input_desc`，main与CPU的dtype、shape相同，四维shape包括 `[1,32,16,80]`、`[1,16,32,128]`、`[1,80,8,128]`；main标ND，CPU标NCHW。没有回传Data接口差异，也没有三维flatten探针差异。具体数值dtype编号仅在所选DDK schema内解释，不作通用枚举推断。

上一版 `_interface` 的全端口描述符相等要求，会让与所追溯边无关的四维输出阻断三维FP32探针路径。修复仅删除这一无关范围检查，保留唯一CPUCL归属、唯一packed Data/NetOutput、完整接口端口数量、有序连接和精确源producer/output。对所选路径的Data input/output、GraphOp input/output、NetOutput input/output、最终生产者output逐一要求FP32、ND、相同shape。路径内NCHW/FP16仍拒绝；没有将ND/NCHW普遍等同，也不宣称四维算子或完整设备数值已通过。

新增错误details标明owner、节点、方向、端口、expected/actual及不同字段。终端合并重复原因，完整逐节点错误保留在原报告。更新只影响已绑定OMC的检查和终端输出，不修改AIR、参考、权重、编译配置或数值阈值。

本轮run `20261029T005905Z-26100-cc481a` 组合回归176项通过（`final-tests/tests.xml`），含62项路径/诊断测试。真实四维混合接口几何的回归验证了无关ND/NCHW异标可共存，路径7处各侧的dtype/shape/layout变更仍拒绝；原真实小OMC复核仍为13/13（`real-partition-recheck.json`）。用户完整331项重查及设备逐层结果仍待回传，不能从本地检查宣称完整模型精度通过。

## trace Concat跨CPU分区连接检查

用户完整诊断图已完成OMG编译，精度要求331项、匹配319项，设备未执行。实际12个未匹配项全部为 `__draft_trace/concat_0/{0,16,...,160}` 和 `draft_trace_values`，错误仅为 `FP32_REQUIRED_EDGE_CHANGED`，实际输入生产者为对应 `SubGraph_*_data`，没有CPU节点缺失及端口dtype/shape错误。该清单排除了先前“重复probe被公共子表达式消除”的猜测；没有为该猜测更改图或门禁。

新增 `draft_precision_edges.py` 解析实际main GraphOp及CPU GraphDef的唯一packed Data/NetOutput接口，按有序端口验证全部接口描述符，再逐段追溯至原required producer及精确输出端口。路径必须全部CPUCL、FP32、ND、相同shape；不同源、NPU、额外Cast、缺失/模糊接口、坏端口、循环均拒绝。原153项生产算术要求、331项trace要求、recipe、图和编译配置不变；这是一项来源证明扩展，不是忽略Data边界或删除失败要求。

本地真实双RMS加FP16 NPU桥的OMC复现同一种错误：旧同区检查12/13，新跨区证明13/13。OMC为71,929字节、SHA256 `ba817745729553f28c0cd8d98c1a34719362c2b54f16682bdb7c4b36cf3118d0`，未重编译而仅更新检查即可通过。证据run `20261029T004826Z-25470-5c600f` 的 `partition-smoke/before-check.json`、`final-check.json`，本地公开DDK静态编译证据不代表用户完整OMC已通过。用户内部DDK原OMC仍需原地重新检查，通过后继续诊断上板。

恢复逻辑只允许明确的控制/检查模块更新，并要求reference/fixture/AIR/compilation四阶段均完整、源job/spec/实际构建依赖及编译profile一致；旧owner字节不改。只读 `--inspect-only` 可在新源码下核验和查看旧失败证据，`--include-boundaries` 额外静态导出接口明细，无需构建、写文件或HDC。

本轮组合回归142项通过（同run `final-tests/tests.xml`）；最终补充进度输出后恢复相关32项再次通过（`final-tests/resume-final.xml`，为前142项子集）。覆盖跨CPU来源证明及负例、原153项精度拒绝规则、旧owner原地恢复、编译选项/DDK绑定及只读检查。未修改任何模型数学表达或数值比较标准。

## FP32锁定后的完整draft回传与逐层对照

2026-10-29用户回传PID57427：新完整OMC为2,852,155,819字节，SHA256 `bd9f826bccf2b741a50be8d1afb7e9c95b5ef15aa31e79d3feae42160ae32705`；编译精度153/153通过、native EXECUTED、退出0、输入回读一致。hidden参考L2=175.27058602513762，实际175.2707760915955，相对L2=1.3894797648159405，最大绝对误差19.7333984375；Top1差异15/15。范数恢复只说明最终归一化幅度改善，不能证明中间算术、布局或完整模型正确。无性能测量；远端原文件尚未在本地重取核验。

新增独立逐层诊断入口 `targets/cpro/scripts/run_draft_trace.py`，固定写入 `full-draft/layer-trace`，不替换生产图或golden。它绑定现有prepared及fixture，先重放v3参考并要求hidden/Top1逐字节复现，再输出166个中间探针并检查诊断OMC的精度。所有原GE节点名称、算术和顺序保留，只添加输出观察。每个探针报告相对L2、余弦和非有限位置；mask score合法的同号-inf单独匹配，未新增数值容忍阈值。

诊断图还必须与原设备运行比较终端hidden及Top1。`production_equivalence=DIFFERENT`表示额外输出改变了执行行为，不能将trace的首次非逐位一致解释为原图首次出错。源码审查尚无确定结构差异；draft.fc的Conv2D展开、KV重复、mask和RoPE在主机数学上匹配，真实设备的首次偏离仍待这轮输出。

本地run `20261029T003431Z-23724-5fc0f9` 的最终组合主机回归314项通过（`final-tests/verified.xml`），native CTest 8/8通过（`trace-board/native/Testing/Temporary/LastTest.log`）。两层小draft真实AIR/OMG编译严格检查128/128通过，含原计算61/61和新增trace节点67/67（`trace-compiler-smoke/smoke-report.json`）；六层小配置166个探针与独立参考在三种prefix长度下对照通过。公开同族DDK smoke仅对派生图记录了Select/ArgMax/权重basename接口适配，没有改生产导出器或原153节点精度门禁。这些均不构成完整35B板端数值通过。

[首页](../README.md) · [当前更新](../current-status.md) · [项目结构](../reference/project-structure.md)

本页记录当前实现能证明什么。安装、编译、加载、执行、数值一致、实际执行位置和性能是不同检查；主机测试不作为设备通过记录。

## 编译FP32区域锁定

2026-10-29，正常full-draft入口新增默认 `fp32-cpu`，契约为 `cpro-fp32-cpu-v1`。从已有v6 AIR绑定的recipe收集153个FP32计算节点，逐节点请求编译实现选择，并以实际OMC中的CPUCL归属、端口dtype/shape及FP32生产者连接作为上传前的必过检查。源码指定的FP16与Cast边界、权重、参考和六层结构不变；切换策略只使编译与bundle失效。私有配置快照及其SHA进入编译清单，复用编译也重查；bundle必须绑定当前编译、AIR和fixture清单。

配置使用所选DDK的 `cl_customization` 单组格式。implementation 7实际表示custom AscendC库，不是CPU设备编号；本地该库没有这些普通FP32算子实现，编译器因此采用CPUCL。不同DDK可能具有不同注册，配置被接受不等于锁定成功，必须经过编译产物检查。一个组包含所有受保护节点即可；多组且遗漏其他节点的格式在实际编译中被拒绝，没有纳入最终配置。

本轮run `20261028T235456Z-17998-116042` 的真实本地编译验证：

- 双RMS图：10/10个FP32节点通过严格检查，2个CPUCL区域，中间FP16计算仍位于1个NPUCL子图。证据 `out/precision-two-rms/policy-final-verification.json`，OMC SHA256为 `14dad3f4e1f3054f598f8ccad8474a19c005d15dff81ad984e9ca5dc56369648`。
- 两层小draft：61/61个受保护节点通过，覆盖10个RMS、2个attention及RoPE。证据 `out/precision-draft-smoke/smoke-report.json`。本地公开DDK需要basename权重路径、Select/ArgMax等价接口适配，并省略其不接受的limit_partition_size参数；全部仅用于此smoke，生产导出器和用户DDK参数未改。该小图全部位于CPUCL，不能作为完整模型的执行位置或速度证据。
- 组合主机回归246项通过，报告 `out/precision-final-tests.xml`，覆盖策略生成、真实编码OMC验证、类型/边界故障拒绝、配置快照及恢复入口。最后补齐旧bundle来源校验后，两个CLI测试文件合跑102项通过（工具session 69787，退出0，18.39秒，未另存JUnit）；其中新增3项，去重合计249项。

这些结果验证了本地编译控制与检查链，没有新增完整35B板端结果。用户最新的设备数值状态仍为FAILED；下一步是一条正常入口命令完成受控重编译、打包、上板与比较。没有放宽数值门槛，也没有把静态描述符当成核内累加精度或源BF16模型质量的证明。

## v6已执行OMC的FP32边界丢失

用户最新回传的无dump检查已绑定PID54002的AIR、编译OMC和embedded-stream图/任务分区：编译图包含27个CPUCL与26个NPUCL GraphOp。26个ReduceMean均保留FP32描述，但最终RMS的square所在SubGraph_49输出是FP16，经netoutput_cast_64转FP32后进入CPU SubGraph_50；其FP32均值又经graphop_data_cast_41转成FP16，送入执行后半段归一化的NPU SubGraph_51。源图要求整个平方到归一化乘积为FP32，编译后的这些边界与该策略不符。单独看到ReduceMean是FP32不能说明整段RMS保精度。

实际NetOutput的hidden端口为FP16 `[1,15,2048]`；原生分配前后均为61,440字节且buffer身份不变。奇偶半字均有大量大幅值，将同字节解释为FP32得到约1.57e17幅值，不符合“正常FP32归一化结果被当FP16读取”的简单解释。这些证据定位了编译精度边界问题，但不证明±257.75来自哪一种芯片内部机制，也不排除其他数值错误。

后续修正必须覆盖所有FP32计算区域，并在打包/上传前核对编译OMC的算子归属、端口dtype及FP32连接。只有完整板端对比通过后才能更新数值结论；原参考、权重、结构及比较门槛保持原契约。

## v6完整draft执行成功后的数值失败

用户回传PID54002的v6 AIR/v3参考流程：OMG编译通过，embedded-stream OMC为2,852,358,026字节，SHA256为 `717fc0c43a81a865cac05eb8626601e90ca5863bf15af91df02b1625c9d57a45`；原生EXECUTED、退出码0、输入回读MATCHED。Top1差异12/15；参考hidden L2=175.27058602513762，设备L2=27883.188584137377，相对L2=159.06354720130383。设备全部元素有限，但范围为[-258.75,258.75]，429个元素为零。没有性能测量，实际执行位置未确认。

最终scale全1的归一化约束与这一幅度严重不符。该症状优先指向需要检查最终RMS链及输出转换，但不能据此确定某个节点、编译器转换或芯片机制。53个子图及启动时动态库探测错误也不单独构成根因证据。当前source/recipe审查没有发现确定的数学结构错误，因此未继续盲目更改计算或重复编译。

新增 `--diagnose-numerics` 复用当前执行产物，在固定小报告内批量给出逐行hidden统计、常见FP16位模式和编译IR描述符。解码采用所选DDK库中的静态schema，不执行该库；同一dump只解码一次，省略dtype、融合改名、跨图同名和资源限额均明确标注。编译描述符仍不是运行时张量值，任何差异都需要结合节点连接解释。原golden、比较结果和设备证据不改写，数值通过阈值不变。

该远端运行结论基于用户粘贴记录，原始远端文件尚未在本地复核。本轮212项主机回归通过，报告为run `20261028T124757Z-10057-688e0b` 的 `out/final-numerics-tests.xml`，覆盖报告、绑定、边界检查、只读行为和现有CLI。它们不证明设备精度通过；本轮数值问题仍未解决。

后续用户诊断确认15行均严重违反单位scale约束：行RMS150.6758–162.2495，余弦-0.0255–0.1357，±257.75共2050个元素；没有保留IR dump。新增无dump分支直接解析已有external OMC，在packing收据核验后解码main ModelDef及CPU任务GraphDef，并报告实际编译IO描述。NPU任务内容仍不可见。输出字节解释检查只添加旁路统计，不更改原dtype或比较。源码到recipe、C++描述符/属性及AIR重载校验的审查未发现明确Cast编号、轴、epsilon或gamma缺陷；未据此修改生产数学表达。

无dump分支及相关回归去重后290项主机测试通过，证据为run `20261028T132516Z-11513-4a97e8` 的 `out/retained-omc-handoff.json`。覆盖真实编码的ModelDef/GraphDef、packing身份与分区范围、opaque NPU、编译后的CastT/Power摘要、输出端口和日志哈希；最终完整性检查失败会丢弃已解析边界及日志结论。真实服务器OMC的检查结果仍待更新入口回传。

## 4B AIR精度对齐与结构核对

用户完整服务器重放逐字节复现旧参考：第0层post_attention_layernorm输入最大340，FP16平方27个非有限值导致9行归一化输出为零。RMS单独改为FP32后，26处norm均无新零行，但与旧板端仍有12个Top1差异，hidden相对L2=1.9395076956574235。该RMS-only候选是历史诊断，不是本次完整精度对齐后的新参考。

最终gamma为全1时，有限正确RMS输出L2应约为sqrt(30720)=175.271218，末端FP16舍入仅有小变化。RMS-only参考175.271140符合，旧板端357.492527不符合，表明旧板端最终输出不满足该约束；设备平方溢出是否采用饱和仍未观察，不能认定具体机制。

按4B分支 `feature/gdr-chunk-verify`、提交 `90186732d915e8b058b2a053d205991d805eae76` 核对实际AIR路径后，GE与新CPU参考同步采用以下边界：

- RMS：输入、平方、归约、epsilon、rsqrt及归一化乘积FP32，转FP16后乘原始gamma。没有对draft权重加1。
- Attention：FP16 QK结果转FP32，再缩放、mask和softmax；概率转FP16后进入FP16 PV。Cast不证明硬件内部累加精度。
- RoPE：inverse frequency、position、角度、cos/sin为FP32，仅最终cos/sin转FP16。

源文件分别为 `models/dflash_v1/dflash_ascend310p_ops.py:124`、`framework/python/qwen35_dflash/ascend310p/quant_factory.py:237` 和 `models/dflash_v1/modeling_dflash.py:94`。固定源码快照位于本run的ref4b-source。4B的RESULTS.md标为PASS_WITH_DIFFERENCES，20.22ms不证明全部数值一致。

GE IR、C++描述符及recipe增加FP32类型，已有类型编码保持不变。参考契约v3与AIR契约v6分别使旧阶段失效，重新生成的输入须与旧fixture哈希一致。旧legacy和RMS-only replay保留；当前新参考通过正常board比较或compare-only验证。旧设备执行成功不能作为新图的精度证据。

两者训练结构都为六层DFlash、block16、32个Q头/8个KV头、head_dim128、前五层sliding加末层full attention。4B/35B的hidden分别为2560/2048，MLP为9216/6144，target feature宽度为20480/16384，target取层及embedding/head共享方式也不同。保留35B checkpoint配置。对固定4B源码及其padding adapter和当前35B实现做独立结构对照，在相同小配置、权重和公共运算下，prefix1/6/8、window4的三组六层hidden均逐字节相同。证据在run `20261028T120455Z-6764-8b00f2` 的 `out/structure-audit/report.json`，这不等于两个不同checkpoint的真实输出相同。

本次最终有效主机结果去重后488项通过、4项可选GE runtime检查跳过，未解决失败为0。覆盖GE/recipe精度边界、完整六层源图计算、类型读写、恢复失效及保留结果复核；早期旧FP16断言失败由更新后的用例覆盖。证据为本run的 `out/air4b-test-summary.json` 及其列出的JUnit报告。

当前本地缺少所选Cpro目标配置。后续用户回传已确认新图的实际DDK编译和设备执行完成，但数值失败，详见上节；精度与性能仍未通过。

## 与35B官方源码的精度关系

再次核验固定checkpoint `52cb554b4995dede3e2e1bdb129cdb1f3529332b`：config下载字节与本地 `specs/draft-config.json` 及锁文件SHA完全一致。配置为BF16；[固定README](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash/blob/52cb554b4995dede3e2e1bdb129cdb1f3529332b/README.md)记录SGLang、BF16和FA4 draft attention。该checkpoint目录没有dflash.py，也没有固定SGLang版本或作者代码commit。因此不能将4B AIR对齐或当前作者源码当作35B训练/服务数值行为的完整复现。

[作者当前源码快照](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py)使用Qwen3RMSNorm、Qwen3RotaryEmbedding和SDPA attention；RMS与RoPE的FP32计算及返回输入dtype顺序与本次修正一致。当前参数/主激活从BF16转换为FP16，且AIR显式物化FP16 QK/PV结果，SDPA/FA4内部累加和舍入不由这一表达证明。结论限定为FP16部署候选；原BF16质量及设备数值等价均未通过。35B结构/权重配置保持原样，4B只作为部署实现参考。

补充核对SGLang当前固定源码 `5cc9c933fc8acdffb5fc847d3b4824f082cd3297`：DFlash创建的普通RMSNorm默认 `cast_x_before_out_mul=False`，`layers/layernorm.py:873–917` 的native路径在FP32归一化后先乘gamma、最后转换输出dtype。这与作者Transformers及当前AIR的“先转换再乘gamma”有舍入边界差异。CUDA分支调用融合RMS，BF16 Q/K还可融合norm/RoPE；Python native顺序不证明所有融合核的逐位行为。该SGLang提交也不是官方benchmark固定版本。源码快照及哈希在本run的 `sglang-source/manifest.json`。因此“RMS使用FP32有源码依据”不能扩展成“当前RMS与官方SGLang所有精度边界相同”。

来源及文件哈希记录在本run的 `out/source35b-audit/report.json`；作者数学核对记录在 `out/original35b-source-audit`。未因源码核对更换checkpoint、回退mask修复或放宽验收阈值。

## 退化参考与RMS精度重放

用户回传compare-only报告确认：旧参考hidden全部30,720个元素为零，expected Top1为15个0；板端hidden全部finite，30,711个非零，L2=357.4925270667643，范围[-9.5,10.171875]。双方Top1差异15个。norm审计核验26份scale均有限且非全零；最终norm和hidden_norm的2,048个scale均为1。设备执行已通过，当前旧参考退化，不足以单独判断设备精度。

新增CPU重放保持原输入、权重、fixture、bundle和板端文件。先重放旧FP16 primitive并逐字节对照旧golden，然后仅对DFlashRMSNorm采用现有TorchDFlashOps的FP32归一化公式；其它FP16计算边界不变。逐节点trace记录输入/FP16平方/输出统计，区分原本为零的padding行与有限非零输入变全零的行。当时尚未观察完整重放；用户后续结果已确认这一参考退化路径，见上节。

RMS公式核对依据是[Transformers v5.7.0 Qwen3RMSNorm](https://github.com/huggingface/transformers/blob/6ffbb07f93d9e44457450d1150136309b0dc966b/src/transformers/models/qwen3/modeling_qwen3.py#L50)：归一化在FP32中完成，normalized转回输入dtype后乘直接scale。当前checkpoint源码锁未包含有效的35B dflash.py；因此这个对照不宣称复现完整历史上游BF16实现，尤其RoPE和attention仍保留旧参考FP16边界。

新参考生成增加全零hidden拒绝；finite-only检查不再把这种退化输出记为REFERENCE_COMPLETE。重放产物只用于诊断，原golden与现有数值阈值不覆盖、不放宽。

本次主机验证286项通过：包括六层小模型旧参考逐字节复现、受控FP16平方溢出与FP32恢复、输入/权重/输出哈希绑定、诊断文件复用及原始证据保持。证据位于run `20261028T114628Z-5767-f04adf` 的 `out/reference-replay-integration.xml`（260项）和 `out/draft-reference-replay-tests.xml`（26项）。该轮测试时完整服务器重放仍待运行；后续用户已完成重放，结果见上节，板端数值对齐仍未通过。

## 完整 draft 执行与保留输出复核

用户回传流式内嵌OMC已在PID14008执行，native为EXECUTED，退出码0，输出回收无错误。模型大小2,852,374,570字节，SHA256为 `dc1136c5fe9ce7c83157f2fa50504a89eb3f7f5c6487a7b654dd274bb17ae0b4`。原加载故障已越过；最终作业FAILED发生在主机hidden相对L2产生inf后写JSON，因此不能把作业FAILED解释为本次Process失败，也不能据EXECUTED宣称精度通过。

比较器现在对零参考范数返回relative_l2=null及明确原因，保留绝对误差、误差范数、双方有限/零/非零值和范围。非有限参考或实际值返回可序列化的失败结果；全零参考单独标记退化。无新hidden验收阈值。

`--compare-only`对原bundle、native结果、输出、退出状态、传输哈希和runtime身份重新核验后比较，原board文件不变，只更新固定comparison-review.json。它不执行HDC、编译器、加载或参考重放。小型norm权重统计只用于排查全零scale；不能代替中间激活诊断。

本轮 `20261028T091818Z-3980-265c4f/retained-comparison-regression.xml` 记录206项主机测试通过，覆盖零/非有限比较、严格回收证据、重复离线复核及固定工作区。测试中的device标记为模拟数据，不是新增实机证据。

历史FP16 primitive参考的RMS平方有明确溢出风险：FP16输入[300,1,-2,3]的平方包含inf，归一化结果可全零且finite。当时这个最小复现不能单独证明完整draft的根因；后续用户逐字节重放已定位旧参考塌缩，当前修改见上节。原生单步未计时，性能未评估。

## 外置编译结果流式封装

2026-10-28，最新用户报告PID6868确认所选DDK已构建V2 runner，文件恢复成功，CPUCL初始化成功，NPU初始化仍报 `weightsMemBaseSize=16 modelWeightMemorySize=1151486976`，尚未调用Run。不能再把文件恢复API存在或Restore成功当作外置权重加载通过。

新 `--weight-storage embedded-stream` 保留已有external编译结果，将权重直接写入bundle内的新OMC，绕开OMG已失败的大缓冲区保存入口。静态参考DDK的写入/读取路径确认IMOD头长256字节，分区表记录type/累积偏移/大小，type1为按GraphOp权重偏移寻址的连续分区。实现仅接受 `[0,1,2]`、原type1为空、表偏移吻合、单主图和每个GraphOp范围连续且不重叠的情况；使用所选编译器的protobuf描述解析，不加载vendor库。原图、任务和权重字节不修改。已知参考DDK与用户所选DDK不同，因此仍严格检查实际输入，不宣称参考二进制重现了用户的memcpy_s故障。

主机验证覆盖完整字节布局、哈希绑定、偏移与边界、原子发布、空间不足、源文件变化、bundle来源绑定和只重建必要阶段。报告中的EMBEDDED只指封装完成；设备执行和数值验证仍待实测。 本轮 `20261028T084216Z-2648-a906ed/embedded-stream-regression.xml` 记录253项通过；保留的参考DDK OMC经提取、重新封装后93,178字节逐字节相同，SHA256为 `c5582a919ace3de1f602c13bc21b60f3bda8176f214271575e59e5a1d4c0f98a`，证据为 `out/omc-embedded-reference-roundtrip/report.json`。参考文件回环不代表用户当前完整draft的兼容性验证。

## V2 文件加载与完整 draft 执行

2026-10-28，用户回传所选DDK的API检查：当前头文件声明 `IBuiltModel::RestoreFromFile`、`IModelManager::Init`、`Run`、`InitWeights`，`libhiai.so`有CreateBuiltModel/CreateModelManager等对应已定义符号。此前两个setter互相清空字段的观察保持有效。API存在尚不是实际外置权重加载通过。

新 `run_full_draft.py --model-load-api file-v2` 将文件恢复、初始化、ND张量核验、同步Run、输出回收及参考比较接在一次完整流程中，复用已有成功external编译和bundle。没有新图或权重转换，也没有推测调用InitWeights。legacy-buffer路径保留，target/fused入口仍使用原API。原生报告记录具体runtime_stage和加载API，Python要求file-v2结果身份匹配，旧runner结果不能冒充新路线。

失败HiLog流式选择记录，取消16MB整文件大小拒绝；新增同一服务线程在明确提及runner PID的失败前后两秒上下文，且限制在候选runner记录区间内。共享服务线程仍可能穿插其他任务，因此这些行是关联诊断，不是独占归因。SDK编译失败自动展示实际编译错误和ND头文件。

设备状态为用户已报告的external初始化失败，V2路线已由所选DDK构建并在设备复现16字节权重错误。本地缺少cpro-blue-pending目标配置，不报告远端通过。验证命令与结果见本轮工作流记录。

本轮 `20261028T081315Z-32283-07df46`：`v2-board-tests.xml` 记录134项Python流程测试通过；`diagnostics.xml` 记录59项日志/SDK构建配置测试通过；`native-file-v2-tests.xml` 记录8组CTest通过。覆盖完整V2调用、12种前置或推理故障、输出对象变更拒绝、ND形状/类型检查及旧路线回归。主机ND/HIAI使用测试替身，不证明外置权重加载或模型计算正确。

## 外置模型加载描述符诊断入口

只读API检查在 `20261028T075743Z-31932-5ac257/api-inventory.xml` 中与已有描述符驱动合跑27项主机测试，全部通过。覆盖按已有构建记录定位SDK与库、声明/符号提取、身份变化拒绝、工具缺失报告，以及重复检查保留原设备证据和模型文件；未在本地执行所选DDK的符号工具，不能据测试断言接口存在。

2026-10-28，固定目录迁移后新增 `probe_draft_load.py`，从当前成功external编译记录核验小主OMC，使用与完整draft相同的运行库选择规则构建并部署当前runner。仅上传主OMC（上限64MiB）、runner及运行库，不读取或上传权重sidecar，也不执行AIR/OMG、模型Load或Process。

原生入口为 `--probe-model-description`，在两份新 `AiModelDescription` 上分别执行路径→buffer与buffer→路径，记录每步setter状态和getter读出的路径、buffer存在/匹配状态及长度。用户现已回传DDK `100.520.020.010`、运行库 `800.631.120.010`、PID63057的设备观察：四次setter返回值均为0；`SetModelBuffer` 后路径为空、buffer保留且长度2,905,990；`SetModelPath` 后路径保留、buffer为空且长度0。两个setter互相清空另一字段，交换调用顺序无法同时保留两项。该次诊断没有修改正常加载代码；随后新增V2路线见上节。

这份用户粘贴报告绑定主OMC SHA256 `cc9a70f56639bf96b74b773f31cf5213c8fba65c63f46d3d8d6abd574bed9132`、大小2,905,990字节，回收 `probe.json` SHA256为 `3b79e966c46f9d934d0845fe5df28d7d3eba080d8bab1bbf94c1ce343fda1ebf`，远端退出0，传输错误为空。本地未重新获取原始文件独立复核；证据范围是用户回传的所选设备诊断。没有调用Load/Process，不能把 `INSPECTED` 当作外置权重加载、模型执行或数值正确。

该观察确认现有“路径→buffer”加载代码最终失去路径，但不能单独证明NPU所收权重16字节的全部原因。历史提交 `bf545c2` 在tiny target／draft bring-up中记录过只设置路径后HCL `CreateLocalBuffer` 因 `data != nullptr && size > 0` 失败；该记录不覆盖当前原版40层OMC。该轮决定不盲目换setter顺序仍保留；后续显式FILE_ONLY完整实验见[新候选](#resident-v1-file-target-candidate)，并非自动fallback。732的Load失败与6599的embedded序列化失败仍未解决。

结果集中在 `full-draft/load-diagnostic`，设备诊断复用固定工作目录锁，重试只替换该入口已识别的旧诊断目录，不清理模型阶段。主机端验证回收结果的模型身份、两种调用顺序和未执行Load/Process声明，完整观察才标为 `INSPECTED`；原生非零退出或传输/身份错误不得升级为成功。随后 `probe_draft_load.py --inspect-api` 从已有 `native/command.json` 收集所选头文件声明和动态符号，写入 `api-report.json`、`api-report.txt`；不调用HDC、不读取模型或权重、不重跑或重置设备诊断。本轮本地缺少 `cpro-blue-pending` 配置，实际文件恢复/外置权重加载路线尚未验证。

本轮 `20261028T073909Z-31045-e9d76e` 的 `probe-driver-final.xml` 记录19项Python测试通过，`native-description-ctest.xml` 记录7项原生CTest通过。覆盖不读取/上传sidecar、身份和传输失败、诊断目录覆盖，以及模拟setter相互清除字段、非零返回码和无Load/Process调用；原有runner回归也通过。首次驱动测试的两项失败来自模拟返回缺少新增的 `evidence_kind` 字段，修正夹具后通过，未放松真实结果校验。

## 固定 draft 工作目录与内嵌模型保存失败

2026-10-28，最新用户回传 `full-draft-6599025a0b74` 的embedded重试：日志到达 `Compile end`，随后 `SaveToBuffer/CreateCompiledModelBuffer` 中 `LargeMemoryOps::Copy` 的 `memcpy_s` 失败，经 `SetModelHeader/CopyDataToBuffer` 返回 `Save compiled model to file failed`。没有生成可用于本次执行的成功编译结果，也未调用设备Process。当前只确认大缓冲区序列化失败，具体大小上限、参数或内存原因未证实；此前sidecar总量小于UINT32并不能证明编译器能生成这个大小的内嵌OMC。没有自动切换external或重新运行失败编译。

用户要求停止积累随机批次目录。入口默认改为 `$AI_RUN_DIR/full-draft`：已有合法任务自动原位恢复；成功组件保留字节并复核，失效或未完成阶段连同旧board输出清理重建，最新报告/失败日志替换旧版。AIR契约变化清理AIR/compilation/bundle，编译选项变化只清理compilation/bundle。工作目录锁拒绝并发修改，迁移日志支持中断恢复；显式全新 `--output` 仍保留旧版独立目录引用复用方式。

本次首选使用 `--resume "$AI_RUN_DIR/full-draft-732b26335347" --prepare-only`，将d7 prepared/fixture与732 v4 AIR、成功external compilation、bundle迁入固定目录，不调用OMG或设备。从732继承external模式，以保留当前已有的成功编译结果；这不证明external Load故障已修复。6599仍是最新embedded失败记录；从它迁移也有效，但只有prepared/fixture/AIR，不能一并保留732的成功编译，故不是首选。迁移采用同一文件系统内目录重命名，无大权重副本；保留组件清单原字节及哈希，仅改顶层组件目录。原prepared/export根路径作为历史来源保留，实际payload使用组件内相对路径。旧批次根目录不自动整目录删除；迁入完成且当前组件不再引用旧目录后，用户可清理旧批次。下文“保留旧目录/新目录恢复”是当时流程的历史记录，不是当前永久保留要求。

固定目录只改变本地工作目录和重试管理，不改变模型图、参考、权重、量化、HIAI加载调用或设备结论；本轮本地未运行所选DDK或真实设备。后续按[独立draft指南](../reference/draft-standalone.md)使用固定目录，待保存模型故障定位后再执行需要的编译步骤。

本轮 `20261028T070549Z-29814-248e1c/fixed-combined.xml` 记录227项主机回归通过，用时16.18秒，覆盖CLI、工作目录迁移/锁/中断恢复、已有AIR编译与产物恢复、bundle及只读检查。首次 `fixed-first.xml` 的65项通过、13项失败来自模拟夹具缺少 `artifact_kind`，修正后通过上述最终组合测试；原失败记录保留。这些测试未执行所选DDK或真实设备，不是6599序列化故障或732加载故障已修复的证据。

## 完整 draft v4 越过 CPUCL 初始化后的权重内存失败

2026-10-28，用户回传 `full-draft-732b26335347`：v4 AIR编译及打包成功，PID46426日志显示CPUCL初始化成功，已越过前两版的 `v501/BroadcastTo` 障碍。随后NPU `SubGraph_1` 的 `InitRuntimeParams` 失败，报告 `weightsMemBaseSize=16`、`modelWeightMemorySize=1151486976` 字节。失败仍在Load阶段，没有Process或数值比较；掩码替代的设备数值和完整模型正确性均未验证。

同一日志中 `compiled_model_utils.cpp operator()(524)` 对15个子图均打印 `weight hash:0x1dabbe2881ec5472`，其权重尺寸不同。这个运行时字段的含义尚未确认，不能当作文件SHA256、文件内容相同或错配根因的证明；该批次实际sidecar文件哈希清单尚未回传。16字节同样是运行时内存大小记录，不能直接解释为上传文件被截断。

用户随后回传编译目录的文件大小：`SubGraph_1.weight` 为1,151,486,976字节，与模型要求一致；`model.omc` 为2,905,990字节。其余14份权重的大小也已记录到工作流清单。该观察排除所列编译端 `SubGraph_1.weight` 被截断为16字节，但不证明其内容、设备副本或运行时选中的文件正确；没有收到这些原文件的独立SHA256复核。

本轮 `20261028T063649Z-28459-d40778` 进行了源码只读审查：`existing_air._generated_inventory` 按完整相对路径枚举并记录大小/SHA256；`draft_board` 保留目录与文件名，按路径字典选取OMC，不依赖子图编号排序；`retained_air` 恢复同一路径后重新核验；上传前检查本地SHA256，并在运行前执行远端 `sha256sum -c`。这未发现脚本层的索引或basename错配，但既不是远端文件复核，也不证明HIAI实际打开了已上传的sidecar路径。

该轮runner先 `SetModelPath` 再 `SetModelBuffer`；较旧的[Huawei署名公开源码镜像](https://github.com/946166920/hiai_ddk/blob/42142161fb414a8d53973b0d5ccbd0de7872d5d9/src/framework/model_manager/compatible/AiModelDescription.cpp#L113-L131)提供清空路径线索。所选DDK的相同行为现已由上节用户回传getter观察确认，反向setter也清空buffer；原先头文件声明本身不提供这项证据。路径丢失仍不能单独解释16字节来源。此前成功tiny未启用外置权重，不验证当前加载契约；实际外置文件选择和文件恢复接口仍待核对。

用户随后要求尝试将权重与模型一起编入OMC。新增显式 `--weight-storage embedded` 将 `save_weights_as_external_data` 改为false，从732复用v4 AIR及d7 prepared/fixture，只重建compilation/bundle；新任务默认仍为external，未指定模式的续跑继承原模式。原732外置产物和失败证据保留，模型图、权重内容、参考、量化和输出ABI不变，HIAI加载调用也不变。这是装载方式兼容候选，不是已确认根因或已验证设备修复。

所列15份外置权重合计2,849,468,580字节，小于runner UINT32长度上限4,294,967,295字节；该合计不是新OMC长度的保证。打包前与原生加载按实际OMC文件大小检查；当时尚待验证的内嵌保存步骤，随后在6599批次失败，见上节；Load和数值仍未验证。当前先迁移固定目录，命令见[独立draft指南](../reference/draft-standalone.md)。

本轮 `20261028T063649Z-28459-d40778/embedded-regression.xml` 记录203项主机回归通过，用时8.68秒。覆盖模式选择与续跑继承、只重建编译/bundle、模拟OMG收到false参数及相应元数据，以及无sidecar的单OMC bundle校验和上传计划；并回归既有AIR编译、产物恢复和只读检查。OMG与板端路径使用模拟环境，未在本地执行所选DDK或真实设备，不能作为内嵌OMC实际编译、Load或数值通过证据。

## 完整 draft 的 INT32 形状参数重试仍失败

2026-10-28，用户回传 `full-draft-b2ecbb36f2cd`：v3 AIR已完成编译和打包，板端PID29314仍在CPUCL的 `v501/BroadcastTo` 初始化处报告 `InitInput1Desc` 输出维16对应输入维0，随后 `SubGraph_0` Prepare与HIAI Load失败。该节点的源/导出描述仍为 `BOOL [1,1,1,80] → [1,1,16,80]`。这说明只修改形状控制常量宽度不足以解决当前故障；没有证明运行时出现0的解码原因，也没有执行Process或数值比较。下节INT32版本的主机通过记录保持原验证范围，不能作为设备修复证据。

下一候选仅替换动态BOOL掩码的单例维复制路径：Cast成INT32的0/1，GatherV2通过全零索引复制query/head单例维，再Cast回BOOL。范围为完整六层的6处query-row扩展及6处Select补出的head扩展，共12处。保留掩码值与布尔语义、模型权重、PyTorch参考、浮点计算及输出ABI。没有选择直接BOOL ConcatD，因为公开ConcatD类型契约不包含BOOL；既有tiny的KV Gather仅提供精确复制思路，不能替代本次BOOL路径的设备验证。

当时新契约为 `cpro-fileconstant-relative-int-fc-conv2d-mask-gather-v4`，从 `b2ecbb36f2cd` 复用d7 prepared/fixture，重建AIR及依赖产物；后续732批次的编译与CPUCL初始化结果见上节。检查用户指定4B参考分支时，integrated与incremental源码仍使用mask expand，没有找到Cpro CPUCL BOOL BroadcastTo的现成替代；不能从其310P运行记录推定本次兼容性。

主机回归在 `20261028T060146Z-27345-da9c4b/mask-combined.xml` 记录279项通过、4项因未配置真实GE头文件/运行库跳过。独立recipe解释器覆盖完整六层12处mask复制、tiny、两条导出路径、短/部分/完整前缀和含NaN/Inf/符号零的Select逐位比较；阶段恢复测试确认v2/v3旧产物只复用prepared/fixture，原目录不变。失败日志摘要为空时还会打印筛选状态、PID及大小限制原因，捕获成功不再与摘要生成混淆。上述结果不包含所选DDK编译或板端执行。

## 完整 draft 的 BroadcastTo 加载失败与 tiny 对照

2026-10-28，用户补充 `full-draft-d50bcea5ed15` 的PID13294 HiLog：`broad_cast_to_op_execute.cpp InitInput1Desc(167)` 报输出维16对应输入维0，随后 `cpu_graph_executor` 明确记录 `Op[name:v501,type:BroadcastTo] Init error`，并导致 `SubGraph_0` Prepare及模型Load失败。当前完整六层S64图的 `v501` 是最后一层full-attention掩码复制，输入 `BOOL [1,1,1,80]`，输出 `[1,1,16,80]`；源码和导出描述中的尺寸均为正数。模型没有进入Process，数值比较仍为NOT_EVALUATED。证据来自用户粘贴，未独立回收并核验该批次远端产物。

对照历史：`5db0ef1` 只把有界静态Const广播提前展开，动态掩码广播保留；`78d94ef` 把KV头复制的五维广播改为四维GatherV2，保持顺序及FP16原始位模式。两项均仍存在。`0d108c1` 记录tiny板端执行成功，建议源码为 `c1dd0aa`，但未提供实际提交与产物哈希；当时对应动态掩码为 `[1,1,1,12] → [1,1,4,12]`。当前 `ge_draft.py` 与该成功记录时的版本一致，没有发现遗失修复，也没有找到相同InitInput1Desc错误的历史证据。

当时的兼容候选仅把BroadcastTo形状控制常量从INT64改为经范围检查的INT32，同时覆盖图中显式广播和Select生成的同形输入广播，沿用已有Reshape控制常量约定。布尔数据和目标尺寸保持不变；权重、浮点计算、PyTorch参考及模型ABI不变。该版本导出契约为 `cpro-fileconstant-relative-int-fc-conv2d-broadcast-i32-v3`。后续上述b2批次确认编译通过但设备Load仍失败；运行时对形状常量的解码问题没有得到证明。

固定版本的[官方TorchAir BroadcastTo定义](https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/python/torchair/_ge_concrete_graph/auto_generated_ge_raw_ops.py)允许BOOL数据，形状输入允许INT32或INT64。因此旧INT64符合该公开schema；此次修改是面向当前X90路径的兼容候选，不能推广成所有GE运行时都拒绝INT64形状。

主机回归在 `20261028T033225Z-24433-7bb3f5` 完成：去重后235项通过；4项依赖真实GE头文件/运行库的测试因未配置资源跳过。完整S64/block16图的12处广播均核验INT32描述与4字节维度数据，动态mask的Select保留NaN及符号零位模式，旧v2板端失败任务恢复时会重建AIR及依赖产物。独立对照旧/新发射记录，tiny的528条和完整图1346条记录除了广播shape控制张量外一致。证据为该run内 `broadcast-cli.xml`、`broadcast-regression.xml`、`broadcast-i32-final.xml`；没有执行实际OMG或设备验证。

## 完整 draft 的 HIAI 加载失败与错误回传

2026-10-28，用户回传 `full-draft-d50bcea5ed15` 的完整编译/打包成功状态，随后补充原生记录：`LOAD_OR_SETUP`，`HIAI Load failed: 1`，`process_called=false`，运行时800.631.120.010、固件OpenHarmony-6.0.0.43。这明确发生于Load，未进入推理和数值比较；返回码本身不能证明外置权重路径、库兼容性或某个图节点有问题。已请求只读设备HiLog，未据此改变图或模型加载方式。

随后HiLog节选显示PID13294已解析模型子图元数据，CPUCL初始化后 `SubGraph_0` 的GraphOp Prepare失败，逐级传回Load。需要该PID的CPUCL失败上下文；较早的可选runtime库查找失败不能单独当作根因，子图权重元数据也不等于所有外置权重已正确加载。

修复了两级错误回传：旧board入口因非零返回码或缺输出bin，先抛“execution/transport incomplete”，原生reason尚未复制；CLI又未显示comparison_error。现在原生身份、runtime/firmware和哈希验证后，保留failure_stage/reason，运行失败不索取成功输出bin，transport_error、retrieval_errors和comparison_error分别保留。身份/传输不完整不能提升为设备执行成功；数值不匹配与运行失败仍分开。

原生报告增加实际process_id，失败后读取一次 `hilog -x`，每次HDC调用最多10秒，保留并校验快照、stderr、退出码。只按已验证的PID筛选；日志轮转、PID复用和共享进程记录仍限制归因。日志采集失败不会覆盖原生原因，不会清日志、修改设置或自动重跑模型。终端打印有界运行日志及候选系统日志，完整证据留在board目录。

主机验证105项Python用例及1个原生测试组通过，包含真实子进程PID匹配、模拟HDC的Load/Process失败、完整成功、输出缺失/篡改、身份或退出码冲突，以及日志采集独立失败。证据见 `20261028T033225Z-24433-7bb3f5/draft-load-reporting-evidence.json`。这些验证只覆盖诊断链；实际模型Load故障尚未修复，设备状态仍失败。

## 完整 draft 编译成功与嵌套 OMC 恢复

2026-10-28，用户回传 `full-draft-a091f16d13d8`：OMG日志明确记录保存 `model/model.omc` 以及 `model/SubGraph_0..14.weight`，随后 `OMG generate offline model success`。旧脚本因只检查 `generated/model.omc` 而误报失败。此前完整fc的K限制已越过，下面“卷积表达”小节的编译未验证状态是当时记录；当前编译成功仍限于用户日志，尚未本地复核远端产物，也未执行设备比较。

修复支持两个明确OMC布局，保留外置数据的相对目录并拒绝空、链接和歧义主模型。续跑对旧版明确的output-check错误执行恢复：核验日志哈希及成功标记、原导出与全部编译输入、编译参数/DDK、生成文件清单及TBE审计，在新目录独立快照；旧失败报告和文件不改。旧日志未记录成功退出码，报告明确标为从原后处理阶段和成功日志推断，不伪称本次运行了编译器。

主机回归覆盖两种布局、嵌套外置权重上传、原生runner从无关cwd读取嵌套路径、篡改/缺失/额外文件拒绝、错误阶段/非零退出拒绝、旧AIR失效处理、恢复后直接进入打包上板以及无GE/DDK的compile-only恢复。原生HIAI替身只验证路径处理，不模拟专有运行时外置权重解释。实际加载、数值和执行位置仍需本次上板确认。

本轮 `20261028T031352Z-23260-7e0c53` 共134项Python主机回归和1个原生测试组通过，证据汇总为 `retained-draft-output-evidence.json`。本地主机没有可用的所选DDK/设备环境，未重新编译或上板。

4B参考改用用户指定 `feature/gdr-chunk-verify` 的 `9018673`。该分支文档记录六层FP16 draft 20.22ms/次，源码含合并投影、GQA重排与增量context KV；此前其他分支的计时记录缺失判断不适用于它。未把310P计时变成本项目Cpro结论，未将规划中的融合head/W8 kernel写成已有实现。

## 完整 draft K 限制与完整点积卷积表达

2026-10-28，用户回传 `full-draft-23abc47e25e7`，实际命令已含 origin-format 与 FP16 参数，仍然在同一 `__ge/v20/flat_output` MatMul 兼容检查处失败。上一轮恢复参数的修改消除了配置差异，但没有解决本次失败。dtype 3/12 的单个 store 拒绝没有在用户摘要中绑定到该 MatMul，不能据此下类型错误结论。

随后用户通过grep提供精确节点上下文：L13661 `matmul_nd_check_support.cc CheckBatchMatMulCubeDim` 明确拒绝 K>8000，L13662是batchmatmul检查包装错误，L13663对应 `__ge/v20/flat_output`。完整图仅该投影K=16384超过限制；其余投影K为2048/4096/6144。该结论只适用于已观察的编译器实现，不等于芯片所有MatMul均有相同限制。

修复候选只将完整draft.fc表达为单个Conv2D：X[m,c,h,w]=X原[m,16c+4h+w]，W[n,c,h,w]=W原[n,16c+4h+w]，c=0..1023、h/w=0..3。无padding/bias、stride1、groups1，输出是原始16384项完整点积，只在结果保留一次FP16边界。没有使用多个FP16 MatMul再求和；后者会引入额外舍入。tiny与通用默认lowering不变，原PyTorch参考保留。NCHW视图用显式描述符和向后兼容recipe v2；ND reshape压缩不得跨越格式变化。

主机检查覆盖完整K/N的CPU卷积与独立矩阵乘、可精确表示的dyadic输入逐位对照、随机值与抽样FP64点积、完整六层GE recipe及全部剩余MatMul的K、旧AIR失效后权重/reference复用。测试阈值仅用于这些host算例，不是新增完整模型验收阈值。实际GE序列化、OMG对Conv2D的支持及板端误差尚未验证，不能宣称已跑通或提速。

保留可选 `--inspect-only --resume`：核对manifest、recipe、编译输入记录，提取命名节点及相邻K限制；用所选DDK库的静态schema尝试解码有界IR。不会运行库、OMG或模型，也不打开外置权重文件。schema/文件缺失或过大均明确报告。日志摘要现在优先显示命名节点和矩阵尺寸限制，避免dtype候选拒绝淹没真正原因。tiny内嵌Const与完整FileConstant的差异仍存在，但当前证据无需把它当成本次拒绝的解释。

本轮 `20261028T024744Z-21237-d382b5` 去重458项主机用例通过，4项需要真实GE的用例跳过。初次整图测试有3项夹具误用recipe返回键，修正后通过；共享recipe升级还触发stateful reader的严格版本防护，已修复并验证Q35GF001原有字节布局不变。保留初次失败和替代通过记录。4B框架仅作只读参考，借鉴增量features/KV及常驻执行；未移植其权重或把310P约20ms报告当作Cpro实测。

## 完整 draft 对照成功 tiny 的编译参数

2026-10-28，用户回传 `full-draft-5133743f7c9c` 已复用 prepared/reference，重新导出的 AIR 越过 FileConstant 读取，随后在 `__ge/v20/flat_output type MatMul don't support` 失败。该节点是 `draft.fc` 的 FP16 特征投影：完整 `[64,16384] × [2048,16384]^T -> [64,2048]`；tiny 对应 `[8,32] × [16,32]^T -> [8,16]`。导出器及转置属性相同。

历史提交 `51e07a7` 和 `0d108c1` 保存 tiny 编译及执行成功的用户报告；对应源码显式传入 `--use_origin_format=true --weight_data_type=FP16`。新入口复用了默认不覆盖原量化 AIR 精度/格式的通用编译函数，因此遗漏这两个参数。本次只为 FP16 draft 恢复它们，记录实际 requested_options 和 dtype override；原 target AIR 默认仍不传。续跑校验编译参数，变化时仅重新编译及打包，保留正确版本 AIR 和之前的权重/reference。

日志摘要优先显示明确的 unsupported-node、infer-shape、format 失败，原始日志与尾部保留。主机测试覆盖两项参数实际传递、默认不覆盖、参数校验、恢复后不重导 AIR，以及被普通 E 级跟踪日志遮蔽的 MatMul 报错。该改动修复了已确认的编译配置差异；完整尺寸能否编译尚未验证，未据此修改 MatMul kernel 或宣称设备成功。经验限定为本项目 FP16 draft 入口回归，不推广为所有 AIR 都应使用 FP16 参数。

## 完整 draft 的 FileConstant 与续跑

2026-10-28，用户回传 `full-draft-d7b4db958850`：71个权重准备、完整六层 PyTorch 参考和 GE AIR 导出完成，OMG 读取首个 `draft.fc.weight` 时报告 `dtype` 未识别，并将绝对权重路径拼到编译工作目录下。此处失败发生在 AIR 解析阶段，未运行设备；用户报告未由本地原始产物重新核验。

新 draft 导出契约将路径写为 AIR 父目录下的 `weights/wNNNNN.bin`，并从该父目录执行序列化。显式 CPro 策略将 FileConstant 的 dtype 写成整数 GE ID；通用 GE 默认仍使用 Type 属性。主机 recipe 写入器、Python/C++ 读取器同时核对 dtype 与输出描述，区别 GE ID 与 recipe 内部编码。整数策略是针对当前私有解析器的兼容候选，公开 TorchAir 使用 Type 属性；不能由主机测试推出真实 OMG 已接受。

续跑校验旧 manifest 后，因导出契约不同仅使 AIR、compilation、bundle 失效；新任务复用 prepared 和 reference，旧目录不改动。测试覆盖版本失效、恢复中断 receipts、校验损坏拒绝、整数元数据及相对路径定位。真实 OMG 和板端比较仍需从上述批次续跑。已将本次经验限于此模型的导出契约与回归检查，未提升为通用 DDK 规则；完整模型精度与加速结论不变。

## target 按原 Cpro forward 对齐

2026-10-28，继续按用户提供的原模型代码修改 PyTorch/GE target。修正重建路径提前降为 FP16 的 embedding/残差、GDN 门计算、gated RMS 与 SiLU 乘积、RoPE 表及最终 norm/head 输入。基础算子权重仍保留已有 FP16 缓存；算子输入精度按明确边界处理，不把结构对齐当作原 AIR 位级一致。

新增原风格 PyTorch wrapper：token IDs 或外部 embedding 二选一，scale 只乘一次，`output_pos` 在 LM head 前选择行，返回完整词表 FP32 logits；所有输入 token 仍更新完整状态和 features。拒绝原导出占位值中越界的 `output_pos=seq_len`、不匹配的有效长度及不连续位置。保留功能式状态，不原地覆盖调用者缓存。GE 同步支持指定行和追加 logits；DynamicQuant 的 FP16/FP32 输入必须来自实际绑定，转换节点和输出位置写入报告。

验证使用显式 CPU 量化参考和合成 GE schema，包含独立 FP32 公式、超过 FP16 精确整数范围的 RoPE 位置、选中行与完整 logits 对照、prefill/decode 状态接续、完整40层64/1构图及 typed recipe。原始源码继续留在私有区。真正的厂商量化/GDN ABI、原 AIR RefData/NZ 兼容性、整模设备精度及性能仍未验证。进行 PyTorch/GE 数值比较时还必须匹配 DynamicQuant 和 GDN 端口精度；当前 PyTorch GDN 门固定转 FP16，GE 按声明绑定转换。

本轮去重327项主机用例及27个子测试通过，证据为 `20261028T010212Z-16908-f051e3/source-target-alignment-evidence.json`。其中 GE 大套件曾有一个 RoPE 测试夹具缺少 Graph 必需输入的错误；修复夹具后同一用例在独立25项量化回归中通过，原失败及替代通过记录均保留。未重复运行已通过的大套件，也没有将它计作设备验证。

## 原 AIR 同级外置权重与目录前缀

2026-10-28，用户提供 `full-target-compile-e019e4a7ad53` 完整日志并确认权重位于 `export.air` 同级目录。OMG 已开始解析 AIR，首个致命错误是 `arg3_1` 的 `ConvertFileConstant` 无法读取 `./glm_export_air_cpro_40block_55_quant/model_model_layers_0_input_layernorm_weight`。此前仅选中 AIR 的 6,361,049 字节；无扩展名权重被筛掉，且原导出目录前缀未保留。此批次失败不是此前的输出参数校验，也不能据启动时的库警告判定算子不支持。

收集器增加非隐藏、无扩展名普通文件，沿用有限深度/数量和数据目录白名单，跳过符号链接。整模入口默认 `export-directory` 布局：AIR 与 cwd 均在 `generated`，同级权重快照在 `generated/<原导出目录名>/`，从 cwd 或 AIR 父目录解析该相对路径均指向快照。通用接口默认保留 `air-directory` 布局；不重写 AIR、不扫描工程父目录、不替换成 YAML 中的原始检查点文件。输入快照从生成物清单中排除，输出仍为 `--output=model`。实际整模编译/设备验证仍待远端执行。

73项主机回归通过，覆盖上述精确路径的模拟 FileConstant 文件读取、两种布局、无扩展名权重、源文件不变、输入/产物清单隔离、输出名冲突、原89输入双档契约及接口检查。证据为 `20261028T004816Z-16432-ffea8e/air-sidecar-tests.xml`；OMG为模拟，不能视作真实整模编译通过。

## 整模失败摘要直接显示

2026-10-28，最初用户只回传 `full-target-compile-e019e4a7ad53` 的 CalledProcessError，尚不足以判断原因；随后完整日志将失败定位到上述外置权重读取。

修复 CLI 只打印完整异常命令、没有显示已有日志摘要的问题。现在终端打印失败阶段、退出码或超时、结果路径及有界日志摘要；完整异常和命令保留在文件中。帮助阶段未进入编译时显示 help.log；不会为显示诊断重跑工具。18项主机回归通过，包含长命令保留/终端截短、早期错误不被长尾淹没、帮助失败和超时。证据为 `20261028T004418Z-16266-569091/compiler-console-tests.xml`，实际OMG未在本地重跑。

## 原量化源码适配与 OMG 输出参数修复

用户回传 `full-target-compile-771648b65da5` 的完整日志后，确认失败发生于 `CheckOutputPathValid` 的 `--output` 参数校验，尚未进入图编译。没有证据把这次错误归因于缺失权重或算子。入口已改用隔离工作目录内的 `--output=model`，输入快照和外置输出保持在同一目录；主机覆盖长路径、空格路径和源文件名冲突。实际 OMG 重试待用户环境执行。

原六份文件保存于当前 run 的私有参考区，AST 检查通过，未执行、未纳入 Git。2026-10-28，用户另行授权发布本轮代码和文档更新，六份原件继续保留在仓库外。新增[原量化模型接入](../reference/original-quantization.md)：按 YAML 原路径读取 HF 基础参数、原 W4/scale、独立 INT8 embedding 及 FP32 scale；PyTorch 与 GE 都接入量化权重。I4ZN 与原件纯函数在四组跨 tile 的普通/专家输入上逐字节相同；没有猜测专家 QPARAM 编码。

当前 run `20261027T130835Z-9607-68b43f` 保留 230 项不同主机用例的通过记录：YAML/加载器/失败边界、独立 CPU 量化 Linear/专家参考、小模型状态接续、完整40层64/1构图、typed recipe、完整端口表及准备权重→GE源码集成。汇总为 `local-adaptation-evidence.json`，其中记录各批次 JUnit 与去重数量。GE 测试使用明确的合成 schema，QPARAM 测试使用哨兵值；它们都不是实际厂商转换、AIR/OMC 或设备通过证据。真实量化绑定、QPARAM、整模执行/精度和性能仍待验证。

## 原始完整AIR及双档编译入口

2026-10-27，用户提供原始 `export.air` 路径及OMG转换命令。新增 `compile_full_target.py`，冻结89个输入（其中80个RefData）和5个动态轴的64/1档，保留外置权重及分区参数。纠正原命令中 `d11ynamic_dims` 的拼写；所选OMG帮助未展示内部选项时仍交由实际编译器判断，参数不丢弃。不强制FP16或探针布局，不自动改变原AIR的Conv/RMS节点。

原文件使用独立reflink/副本暂存；复制前显示总字节，reflink失败后检查实际复制可用空间。复用原TBE诊断，编译日志、失败图及生成文件均保留。成功后可提取原始图描述；解析只记录真实可见节点和原始dtype值，不猜测输入顺序或HIAI ABI。只有 `generated/` 下的文件进入当前接口检查暂存；其它编译工作文件仍保留，外部依赖完整性尚待实际图/加载验证。无需重新跑平台或GDN矩阵。

本轮133项不同主机用例通过：114项综合回归、17项OMC检查，以及后续新增的2项复制空间回归；加入最后两项后AIR编译测试文件23项全部复验通过。检查包含完整动态形状逐项一致、失败日志、审计拒绝、外置产物、源码不变、可选图检查不覆盖编译成功，以及备用流式量化/续跑和留存GDN分析。证据目录为 `20261027T123103Z-6824-18a3c3`。OMG/HDC为模拟，**没有本地执行用户整模AIR或板端推理**；实际整模编译、状态持续执行、GDN精度与性能均待验证。

备用量化准备覆盖完整target的693个权重张量；逻辑缓存约18.1GB，保留原始HF norm权重，校验已完成张量后可续跑。它不等于I4ZN/QPARAM，也不等于已接通自建GE的W4A8。当前优先现成AIR，不要求用户重新量化。

## GDN诊断第二轮HDC通信失败与续跑

2026-10-27，用户回传 `gdn-diagnostic-c0cbac5ac9cf`：`0x00` 首轮两例执行完成，进入 `0xA5` 轮后在 `upload` 阶段失败。嵌套错误是 `HDC verify-checksums missing unique remote exit marker`，HDC stdout为 `[Fail]Failed to communicate with daemon`。该轮尚未进入GDN执行；这次失败是传输问题，不能判为算子Process失败，也未提供新的完整三轮数值结论。原始远端文件仍未本地取得。

修正诊断总报告隐藏具体异常、失败时不展示已完成轮次的问题；细分runner复用和库选择阶段。新增 `--resume`，在新目录重新校验原结果、每个回收文件及runner构建/上传哈希，保留同一设备、runtime、firmware、模型及输入，复用首轮runner，只跑缺少轮次，原文件不改。只对fused探针的只读checksum查询启用精确daemon通信故障的有限重试（最多3次，每次≤30秒，间隔1秒）；不重传、不重跑Process、不重启daemon，真实checksum失败不重试。完整模型上传默认超时和单次行为不变。

本轮147项主机测试通过：17项GDN诊断/续跑及130项板端编排与HDC传输回归。覆盖首轮保留而只执行余下四次Process、已完成批次再次核验不执行、输出/runner篡改拒绝、真实构建记录自动库发现的连续复用、daemon短暂故障恢复/持续失败、真实checksum失败不重试及完整模型默认超时不变。工作流JSON、4段当前文档Bash语法和CLI帮助通过。工作区证据为 `20261027T115942Z-26038-e68837`；主机使用模拟HDC/runner，没有验证真实连接已恢复，也没有新增GDN精度通过结论。

## 环境恢复后九例探针最新回传

2026-10-27，用户回传新批次 `fused-all-f6b81ead7ae8`：恢复后的环境已重新尝试全部九例，GDN、CacheUpdate、Attention各prefill/decode共六例完成AIR导出、OMC编译、加载和执行。以下来自用户粘贴报告，原始产物未在本地主机重新获取、校验或复跑。

| 用例 | AIR导出 | OMC编译 | 加载 / 执行 | 数值 |
|---|---|---|---|---|
| GDN prefill/decode | PASSED | PASSED | PASSED / PASSED | DIAGNOSTIC_COMPLETE；最新回传核心输出均全零、relL2=1 |
| CacheUpdate prefill/decode | PASSED | PASSED | PASSED / PASSED | SHAPE_ONLY_UNVERIFIED_CONTRACT；未验证精度 |
| AdnFusedInferAttention prefill/decode | PASSED | PASSED | PASSED / PASSED | SHAPE_ONLY_UNVERIFIED_CONTRACT；未验证精度 |
| ConvStateUpdate prefill/decode | PASSED | FAILED | NOT_RUN / NOT_RUN | NOT_RUN |
| AdnRmsNorm | PASSED | FAILED | NOT_RUN / NOT_RUN | NOT_RUN |

Attention已完成两种调用，先前的端口校验与导出器const接口错误未再阻止该批次。Conv仍在rank-4的y2（`[1,8192,1,S+3]`）维推后遇到格式配置失败；RMS仍找不到 `AdnRmsNorm` 维推函数。历史五类调用未全部通过，不能据此认定整个DDK没有相关kernel；随后用户选择取消Conv/RMS自定义依赖，原失败保留，不改为通过。

用户随后粘贴同批次 `board-1/comparison/report.txt`，GDN数值如下；原始远端文件未取得本地核验：

| 用例 / 输出 | 非零元素数 | relative L2 | 最大绝对误差 |
|---|---:|---:|---:|
| prefill core_attn | 0 | 1 | 0.019012451 |
| prefill last_recurrent_state | 33172 | 0.9603274111031198 | 0.08654785 |
| decode core_attn | 0 | 1 | 0.005325317 |
| decode last_recurrent_state | 23424 | 0.9745207122180418 | 0.040130615 |

`DIAGNOSTIC_COMPLETE` 只表示诊断已生成。两例核心输出全零、末态大偏差确认精度问题仍在；Cache/Attention仅做形状检查，不能归为accuracy通过。实际NPU执行位置、完整模型与连续状态精度也未验证。

下一步入口为 `run_gdn_diagnostics.py --probe-run ...`：复用原两例OMC和输入，runner只构建一次，分别以 `0x00/0xA5/0x5A` 预填输出后各执行一次，共六次Process。将记录输入缓冲区回读SHA256、输出对象/容量/填充值残留、三轮输出差异，并复制所选DDK的GDN cpp/py/h与tiling证据。不重新编译模型、不执行平台诊断、不恢复Conv/RMS候选。该诊断用于检验输入交接、输出写回和重复执行行为；尚无真实设备回传，不代表问题已定位或解决。

主机独立参考核验已将当前探针输入与固定环境的Transformers recurrent/chunk实现对照，FP16结果的relative L2均小于 `1.6e-5`。prefill/decode参考核心输出全部非零，分别有24431/4022项处于FP16正常数范围；最终输出舍入下溢解释不了板端全零。旧初态在128×128两轴严格对称，尚不能检验初态转置；如缓冲诊断排除交接问题，后续需要非对称初态读入用例。核验及复现脚本保存在本次工作区运行记录 `20261027T113904Z-21762-792bc2/gdn-reference-audit/`，没有执行DDK或设备。

GDN诊断及原编排共63项主机测试通过（初轮62项，增加有效前缀检查后最终诊断13项复验）；C++编译和5组CTest通过。模拟故障覆盖输出未写/部分写/写零、输出对象或缓冲被替换、后续输入或输出填充覆盖先前输入、输入回读/回收文件损坏、FP32末态解析及padding不误判。所有输入在拷贝和输出填充完成后、Process前统一回读；对象替换或输入不符时保留诊断并停止。三轮复用同一runner、OMC和输入，原文件不变。源码与编译证据有文件数/字节限制，仅复制可读文件，不保证找到所有tiling实现。上述主机测试及内存观察不证明设备缓冲一致性、kernel实际执行或数值正确。

本次补齐 `--fresh` 中Conv/RMS失败后的只读检查，并新增 `run_fused_probes.py --inspect-only --probe-run ...`：支持探针目录及all-ops批次，打印关键日志与安装注册文本摘要，保留完整复制证据；已成功重试的历史失败不再触发自动检查。29项板端编排主机测试通过，覆盖禁止编译/TBE/runner/HDC、原产物与DDK不变、损坏产物拒绝、模式冲突及fresh/重试回归。CLI帮助、29个文档Bash代码块及diff检查通过。本地验证使用模拟编译/设备夹具，未在用户DDK或设备执行，也未修复Conv/RMS本身。

## 当前模型算子选择与保留的Conv/RMS失败

用户随后回传 `fused-inspect-bc256ae0b0f4`：检查扫描176个文件、`limited=False`，未找到Conv/RMS注册文本。该结论只覆盖扫描范围，不证明整个DDK缺少实现；原始证据未在本地重新获取或校验。Conv的rank-4 y2后续格式配置失败及AdnRmsNorm维推缺失仍是当前已观察到的失败点。

用户明确决定取消Conv状态更新和RMS的自定义算子依赖，采用本仓库已有基础算子路径；保留GDN、CacheUpdate、Attention三类六例。原工程的自定义Conv用于Cpro，AdnRmsNorm用于ND，Cpro RMS调用内置 `npu_rms_norm`，这两种原工程路径不能混为同一自定义依赖。

`run_fused_probes.py --model-ops --probe-run ...` 复用保留批次中的六例OMC，只检查并上板这六例；`--all-ops` 仍可作为历史九例的可选诊断，原Conv/RMS失败不变。生产GE使用 `--fused-ops gdn`，沿用基础算子的Conv历史拼接/截取和RMS；当前融合接口仅支持 `gdn,conv_state,rms`，六例探针通过不表示CacheUpdate/Attention已接入生产GE图。

本次选择改变依赖范围，不构成基础Conv/RMS在完整模型目标执行或数值精度上的通过证据。GDN历史数值偏差、所选六例精度、NPU执行位置、完整模型状态连续性和性能仍待验证；不继续新增Conv/RMS融合候选。

本轮200项主机测试通过：35项板端编排、43项探针与筛选、122项GE目标构图及导出回归。新增覆盖旧Conv/RMS即使已编译也不上传、不执行、不参与模型模式比较；fresh模型模式只编译六例；旧清单和脚本保持不变。抽取后的板端脚本在全部512个用例子集上与旧生成器逐字节一致。34个文档Bash代码块、工作流JSON、CLI帮助及diff检查通过。GE/OMG/HIAI使用主机替身；没有新增真实DDK编译或上板证据。现有六例OMC未变，用户随后回传的 `board-1` 数值见上表；这些主机测试不覆盖新的GDN缓冲区诊断。

## Docker环境恢复与删除产物后的重建

2026-10-27，用户说明Docker退出，且运行目录除权重外已清空。此前粘贴的编译/上板结果继续保留为历史证据，旧路径不再作为可复用产物。新增 `source targets/cpro/scripts/env.sh`，私有配置存于仓库旁并在检查成功后恢复变量；`run_fused_probes.py --all-ops --fresh` 从选定DDK源码绑定开始重建平台诊断和九例探针。

按用户随后要求，配置仅保存固定环境，不再收录历史产物路径或运行时身份覆盖值；这些变量由调用命令提供，source保持调用者值不变。主机环境恢复10项测试通过，覆盖配置首次创建且不覆盖、不收录运行变量、重复source路径稳定、带空格/特殊字符的路径、保持调用者运行变量（即使旧配置含同名项）、配置错误时不改变调用者环境，以及显式HIAI自动查找对主机库、非OHOS路径和歧义候选的排除。板端编排主机回归22项通过，新增覆盖删除旧路径后的初始化、九例仅编译一次、初始化失败保留阶段记录、新批次重试自动继承诊断、原生平台更新成功时直接编译。GE/OMG/设备使用替身；本次没有真实目标执行。Bash语法、CLI帮助及diff检查通过。

进一步按用户要求复用已激活的vendor/conda环境：默认配置缩减为七项并附中文解释，DDK根目录取 `DDK_PATH`、Python取当前conda、HDC从PATH查找，已有vendor模块及库路径保留。环境回归更新为11项通过，新增覆盖无重复配置时的派生值，以及切换DDK后不沿用旧的派生 `CPRO_DDK_ROOT`；显式配置仍优先。未实际运行vendor环境脚本或目标工具。

随后将当前指南的执行命令统一为 `python3`，项目入口移除 `MODEL_PYTHON` 配置依赖并保持调用者PATH不变。更新后的11项环境测试通过，覆盖未设置解释器变量、旧配置含失效Python路径、首次生成配置及重复source；6份当前指南的41个Bash代码块和 `env.sh` 语法检查通过。仅验证主机配置与命令语法，未重新编译或上板。

用户随后回传HIAI自动选择为0，候选位于 `tools_sysdbg`、`ddk/ai_ddk_lib`、`ddk/hiai_pipe_lib` 的 `lib` / `lib64`。旧逻辑要求路径包含 `ohos`，因此在读取实际ELF前就排除了这些文件。已取消该目录名条件，按ELF架构、已知GLIBC依赖和真实路径去重筛选；多候选继续要求明确选择，并逐项说明原因。16项主机环境测试通过，包含无ohos目录名、ARM/x86排除、真实readelf读取GLIBC依赖、符号链接去重和多候选拒绝。尚未取得用户库的ELF/链接结果，也未恢复此前成功上板使用的确切HIAI库路径。

用户再回贴旧回答，补齐配置：`ddk/ai_ddk_lib/include`，以及 `ddk/ai_ddk_lib/lib64` 下的 `libhiai.so:libhiai_ir.so:libhiai_ir_build.so`。据此优先识别完整三库布局，避免将三个目录里的单个 `libhiai.so` 继续当作未知选择；文档给出独立于 `REPO_DIR` 的私有配置恢复命令。20项主机环境测试通过，新增完整三库恢复、排除其他同架构副本、缺库/错误架构拒绝及显式配置优先。该记录来自用户粘贴的旧配置，不据此宣称三库的OHOS ABI或设备兼容性已验证。

首次生成私有配置时现自动写入HIAI头文件与三库两项，保留DDK变量引用供加载时展开；生成前已有显式HIAI路径则原样保存。仍只需手填原来的七项，已有配置不覆盖。22项环境测试通过，覆盖模板展开、切换到含空格的新DDK路径、未加载DDK时引用配置内根目录，以及后续不再手动exportHIAI路径。

## Attention导出器C++接口修正

用户回传 `fused-all-9759eb4f9b0b`：Attention两例在构建导出器时失败，尚未进入GE序列化或OMG。CANN 9.1的 `GNode::GetInputIndexByName` 为非const方法，新增helper误传 `const GNode&`。现改为可变引用；主机测试的同名方法也按用户日志中的真实声明改为非const，先复现旧代码编译失败，再验证修正。此前测试夹具错误地将该方法声明为const，未能拦截这次回归。

该批次复用的GDN/Cache再次完成加载/执行，数值验收仍未通过或未定义。Conv/RMS未重新编译；只读扫描未找到匹配的注册文本，这不证明整个DDK缺少实现。原始产物未在本地重新核验。

## 五类算子早期批量回传

2026-10-27，用户回传 `fused-all-97b7612ef879`，来源为粘贴总表，原始产物未在本地重新核验：

| 用例 | AIR导出 | OMC编译 | 加载 / 执行 | 数值 |
|---|---|---|---|---|
| GDN prefill/decode | PASSED | PASSED（复用） | PASSED / PASSED | DIAGNOSTIC_COMPLETE；此前明显偏差尚未解决 |
| CacheUpdate prefill/decode | PASSED | PASSED | PASSED / PASSED | DIAGNOSTIC_COMPLETE；无golden，不能判精度通过 |
| ConvStateUpdate prefill/decode | PASSED | FAILED | NOT_RUN / NOT_RUN | NOT_RUN |
| AdnRmsNorm | PASSED | FAILED | NOT_RUN / NOT_RUN | NOT_RUN |
| AdnFusedInferAttention prefill/decode | FAILED | NOT_RUN | NOT_RUN / NOT_RUN | NOT_RUN |

全部执行位置仍为UNVERIFIED。随后完整日志确认：Attention两例报 `GE recipe failed: input port order mismatch`；导出器将8个连接端口当成完整18个注册端口校验，需保留可选端口间隔，修正后重试真实GE/OMG和上板。conv宽度4已通过初始维推，y2分别为 `[1,8192,1,11]` / `[1,8192,1,4]`，后续报 `Not supported OP`（`generated_lib`）及 `Get format configs failed`；这是本次编译路径的格式配置失败，不能直接断言整个DDK没有kernel。RMS仍报 `AdnRmsNorm` 维推函数缺失。conv/RMS未观察到TBE兼容适配调用（0次），不应归因于此前GDN的平台更新错误。

本次已修正Attention导出器按注册槽位校验描述符和连边。主机C++回归运行实际生成的校验函数：旧逻辑复现端口顺序失败，新逻辑接受可选槽位空缺，并拒绝错序、错误描述符/来源及额外连接；recipe解析测试本身不覆盖这一步。真实GE导出、OMG编译与上板待上述重试。

## 自定义算子调用覆盖与GDN板端回传

2026-10-27，用户回传 `fused-board-ec7f1df7f0b1`：GDN prefill/decode均加载、执行成功，但数值明显不符。prefill核心输出relL2约0.9992、末态约0.8856；decode核心输出全零（参考非零）、末态relL2约0.9696。来源为用户粘贴报告，原始结果未在本地重新核验；该记录只证明报告中的调用完成，不能作为精度或NPU执行位置通过。

按用户要求先补齐全部五类自定义算子的调用覆盖，暂不深入优化GDN数值。新增CacheUpdate与AdnFusedInferAttention各prefill/decode，共9例；保留旧三类bindings和默认五例兼容。conv宽度4是显式调用候选，双输出不使用猜测的golden；RMS缺维推注册保持独立失败，不自动替换算子。

`run_fused_probes.py --all-ops` 复用原编译产物，缺项另建批次并继续执行编译成功项，最后汇总全部9例。两批清单与ABI哈希独立保留，同次runner构建可复用。新增Cache/Attention仅为独立调用探针，生产图接入、实际数值和完整模型仍待验证。

主机验证：算子构图/绑定/候选相关39项通过，HDC及批量编排11项通过，native CMake构建与CTest 5/5通过；真实CLI生成全部9例源码成功，Attention recipe另经主机C++解析通过。GE/OMG和设备执行使用替身或未调用，这些结果不替代黄区编译与板端回传。

## 融合GDN最新真实编译回传

2026-10-27，用户回传DDK `100.520.020.010`、目标 `kirinx90` 的两例真实编译结果：

| 用例 | 构图 / AIR导出 / OMC编译 | TBE适配 | 加载 / 执行 / 数值 | 执行位置 |
|---|---|---|---|---|
| `gdn_prefill` | PASSED / PASSED / PASSED | APPLIED 1次，拒绝0次 | NOT_RUN / NOT_RUN / NOT_RUN | UNVERIFIED |
| `gdn_decode` | PASSED / PASSED / PASSED | APPLIED 1次，拒绝0次 | NOT_RUN / NOT_RUN / NOT_RUN | UNVERIFIED |

来源为用户粘贴报告，本次未取得原始产物重新核对SHA256。该结果不覆盖完整模型、设备执行或数值验收；conv/RMS仍保留此前未通过的结论。下一步复用这两例编译产物，执行[当前更新首节的上板命令](../current-status.md#用户现在执行什么)，不重复编译或扩展探针。

上板前发现runner仍硬编码FP32末态，已改为接受绑定中显式声明的FP16/FP32，保留FP16初态→FP32末态兼容；更新测试替身按对应字节数写出，避免FP16缓冲区越界。主机CMake构建及CTest 5/5通过，探针/回收比较23项通过，覆盖prefill和decode的FP16末态及三种状态dtype组合。无需重编OMC，只重建runner；未修改数值阈值，真实设备结果仍为NOT_RUN。

新增单命令入口 `run_fused_probes.py`，复用已编译OMC，自动构建runner、发现并上传私有库、查询运行版本/固件、执行、回收与比较。9项主机协议测试通过，覆盖零参数入口、自动库部署、失败回收、HDC假成功、传输损坏、路径越界及身份错配；构建和设备执行使用替身，主机Shell、SHA256与比较逻辑实际执行。板端验证仍待用户运行，数值验收阈值保持未定义。

## 融合探针编译排查历史

2026-10-27，用户回传DDK `100.520.020.010` 的5例结果：构图与AIR导出通过，OMC编译均失败，未上板。conv维推明确要求 `x2=[1,8192,4]`，与文档绑定的3不符；RMS当前注册表找不到 `AdnRmsNorm` 维推函数；GDN进入 `chunk_gated_delta_rule.py` 编译入口后子进程退出。GDN编译日志的末态是FP16，导出绑定为FP32，尚不能断定变化原因。`Ascend310B1` 内部日志及缺库提示也不能单独证明目标选错或崩溃根因。

本轮修复编译工作目录被删除的问题，新增只读 `inspect-fused-probes`，读取现有日志与匹配的安装接口文本；7项主机测试通过。没有修改文档绑定、生产缓存语义或共享DDK，也没有重新执行真实编译。下一步命令见[当前更新](../current-status.md#用户现在执行什么)。

随后用户回传只读检查结果：两例GDN都在TBE `platform_info.py` 的 `te_update_version()` 抛出 `RuntimeError`，说明本次直接失败点是编译子进程的平台设置；尚不能判定参数、包版本或配置文件中的哪一项不匹配。路径位于Conda的 `site-packages` 不足以证明环境混用，`Ascend310B1` 与 `kirinx90` 的日志也不足以证明需要切换目标。安装目录确有GDN的Python编译入口及C++源码；仅凭通用 `DTYPE_MAP` 不能确认状态dtype支持范围。

针对首次检查遗漏调用帧的问题，检查器补充完整traceback、保存的OMG环境记录、报错指向的TBE平台函数源码、选定DDK配置目录及GDN状态读写片段。8项主机测试通过，覆盖调用帧保留、仅读源码、环境字段过滤及状态读写片段提取；独立代码审查通过。该次诊断修正只读源文件、不执行TBE，当时尚未验证编译修复。

第二次检查回传进一步确认：安装版GDN的 `isGlobal`、`lsGlobal` 均为 `GlobalTensor<half>`，分别以 `reinterpret_cast<__gm__ half *>` 绑定 `initial_state` 和 `last_recurrent_state`。这为该源码的FP16状态ABI提供依据，与文档的FP32末态不同；不推广到其它DDK。新增 `bindings-from-document --gdn-state-source`，校验两处直接half指针后生成FP16末态绑定，并用组合来源文件绑定原文档和安装源码SHA256；默认文档绑定不变。

TBE源码显示：包装函数调用 `cce.te_update_version("kirinx90", "AiCore", "", "false")` 后丢弃非success返回值，抛出通用异常。新增 `check-tbe-platform` 在独立Python进程中读取该底层返回值，使用与OMG相同的环境构造，记录调用前后平台信息、模块哈希及加载库。仅导入初始化，不能宣称复现OMG工作进程；无论成功失败，都不能单独判断目标支持或编译通过。该次工具改动时尚未取得黄区执行结果。

本轮40项主机定向测试通过；命令行测试另验证源码覆盖、两例GDN的FP16输出描述和参考字节数，以及模拟底层返回成功/失败、字符串对象的退出状态。使用模拟TBE验证子进程流程，未调用真实DDK或设备。

用户随后回传：安装源码绑定生成成功，GDN C++ SHA256为 `01634d23da746856c1119a9743d7f6db5f66aef9560d745423556a9c3162125a`。轻量平台调用返回字符串 `error`；调用前后SOC为 `Ascend310`、core为unknown，未复现原OMG工作进程的 `Ascend310B1` 初始化状态。进程同时加载本DDK TBE目录的 `libtvm.so` 和OMG目录的 `libplatform.so`。这些是定位线索，尚不能证明该DDK不支持目标或两库不兼容。

随后增加当前DDK内的库搜索顺序与已安装初始化接口对照。最多四个独立子进程，均使用 `kirinx90`；不更改正式OMG环境，不自动采用任何对照结果。各进程的平台初始化、实际加载库和原始返回值分别记录；报告收集完成不等于编译环境通过。

对照工具17项主机测试通过，覆盖子进程隔离、同目标初始化、库顺序、缺失/不匹配接口、初始化失败和全部失败时的状态。命令行与文档Shell语法验证通过；使用模拟TBE。

用户回传的四组结果均失败：两种库顺序下，仅导入时SOC为 `Ascend310`，调用 `te_set_version("kirinx90")` 后为 `Ascend310B1`，随后底层更新再次收到 `kirinx90` 均返回 `error`。这说明单改库顺序不足以解决该次失败；初始化与更新对平台名的处理差异是下一项假设。`Ascend310B1` 是该DDK的实测返回值，不能据此把OMG目标改为该名字；用户分析文档的 `Ascend310P3` 仍只描述原工程平台。

新增 `check-tbe-platform --diagnose`，把包来源（当前环境/当前DDK的TE wheel副本）、库顺序（OMG/TBE优先）、更新参数（请求名/初始化后读取的SOC名/空字符串）组成最多12组、4路并行。每组都先初始化 `kirinx90`，独立进程、独立输出和超时；包副本仅解压到运行目录，不安装或改写DDK。记录实际导入位置、映射库、更新前后SOC与原始返回值；成功仅代表该次API调用成功，需后续真实OMG验证。此轮不修改生产转换配置或算子计算。

批量诊断工具24项主机测试通过，覆盖并发上限、别名与空参数、崩溃后继续、wheel来源隔离及不产生错误候选。真实CLI用模拟DDK完成12组及逐例进度检查；命令互斥和文档Shell语法通过，独立代码审查通过。

随后用户回传12组真实API结果：传 `kirinx90` 的4组均返回 `error`，传初始化后查询到的 `Ascend310B1` 或空字符串的8组均返回 `success`，SOC均保持。当前环境与DDK wheel所列7个模块哈希一致；改变OMG/TBE库顺序没有改变结果。这把本次平台API失败定位到名称处理差异，不支持继续把已列模块版本或库顺序当作主要原因，也不证明OMG完整编译已修复。

新增显式 `probe-fused-ops --tbe-diagnosis`：核对该诊断与选定DDK的来源，在运行目录复制TE wheel，只适配 `platform_info.te_update_version` 的底层平台名称参数。只有请求为 `kirinx90` 且当前SOC与已验证的初始化结果一致时才转换，其余参数保持；记录真实工作进程调用。OMG平台参数仍为 `kirinx90`，DDK及已安装Python包保持不变。`--cases gdn_prefill,gdn_decode` 只重试两张真实尺寸GDN图，conv/RMS另行处理。

兼容适配的28项环境测试与40项探针/绑定测试通过；最终AST守卫的4项定向复测通过。实际CLI以模拟GE/OMG工具联通两例GDN，独立Python工作进程均加载运行目录副本并留下 `APPLIED` 记录；参数仍为 `--platform=kirinx90`。这验证主机接线与审计，不是实际DDK编译通过。文档命令及独立审查通过；后续真实OMG编译回传见本页首节。

## 完整结构与独立融合调用

2026-10-27，已在主机实际生成完整40层 target、6层 DFlash：PyTorch 输出全尺寸 meta 模型及可执行的 decoder 层级 FX；GE 输出完整节点、连边、C++、recipe 和缓存 ABI，分别生成了非融合、融合 prefill 与融合 decode 结构。均未加载真实权重，也未生成真实 AIR。

新增独立 GE 探针只覆盖 `ChunkGatedDeltaRule`、`LinearAttentionConvstateUpdate`、`AdnRmsNorm`，共5例。每张图只有一个自定义节点，使用完整模型的算子尺寸；不依赖模型权重。

主机联调已贯通接口绑定、生成、打包、批量脚本、native runner、回收及比较，5例均通过接口校验；GE/OMG 和 HIAI 使用测试替身。CTest 5/5、真实 DDK 头文件编译通过。GDN 检查有效前缀及 FP32 末态，conv 的 `y1` 顺序未定义，只记录形状与非有限值；数值误差仅作诊断，没有设验收阈值。

此后的真实GE导出和目标编译回传见本页首节；融合算子板端调用尚未执行。命令见[融合探针](../guides/ge-guide.md#融合算子调用探针)，主机证据见 [specs/verification.json](../../specs/verification.json) 的 `latest_full_graph_and_fused_probe`。

## 状态执行与融合接口接入

2026-10-27，本轮新增了可选 PyTorch 融合适配、GE 状态图独立编译及输入打包、正式 runner 的 `--stateful-step` 模式。

| 范围 | 主机验证结果 |
|---|---|
| PyTorch 融合选择、模型及缓存事务 | 108 项和27项 subtests 通过；其中一项测试夹具修正后单独复测 |
| GE 编译与来源、ABI 校验 | 133项回归及4项补充检查通过 |
| 状态输入打包与连续调用校验 | 35项通过 |
| C++ runner | CTest 5/5通过 |
| Python → native prefill → 状态回传 → native decode | 通过；past_length 为 `0→3→4`，conv、FP32 recurrent、K/V 四组状态字节与 SHA256 精确接续 |

跨模块验证只替换外部 GE/OMG 运行产物和 HIAI 实现为明确标记的测试替身，保留真实的图构造、产物校验、打包和 native 执行流程。它证明主机接口与状态传递接通，**不证明真实融合 kernel、OMC 编译或板端精度通过**。常驻状态管理及 native DFlash 调度仍未接通。当前环境也未取得可用的目标资源清单，真实设备联调待完成。

命令见 [GE 指南](../guides/ge-guide.md#状态图编译和输入打包)和[板端单步执行](../guides/board-guide.md#6-新状态图的单步执行)，证据定位见 [specs/verification.json](../../specs/verification.json) 的 `latest_integration`。

## 本次代码整理的验证

2026-10-27，模型解释器下的定向回归：

| 范围 | 结果 |
|---|---|
| PyTorch 新模型、算子接口、缓存及验证事务 | 89 tests + 27 subtests 通过 |
| GE 融合绑定、状态图、生成器与原导出回归 | 154 tests 通过 |
| HIAI/HDC 生产路径与依赖回归 | 150 tests 通过 |
| C++ 主机 runner 构建与 CTest | 构建成功，5/5 通过 |

剩余1843项 Python 测试收集通过，无导入错误；88个本地文档链接、29段Shell和3段Python示例通过静态检查。PyTorch/GE tiny参考均为PASS，文档接口生成与非融合状态图源码生成也已实际通过。

移动代码的定义与入口保留情况另做 AST 对照；整理验证没有执行真实 GE 库、OMG、融合 kernel 或板端程序。新代码的 host/reference/fake 测试覆盖接口和行为，不证明黄区 kernel 已安装或新 ABI 已在真实设备接通。

需修改模型或接口时，可在[环境配置](../guides/environment-setup.md)后执行对应回归；仅使用项目不必重跑：

```bash
export TEST_RUN="$(mktemp -d "$AI_RUN_DIR/tests-XXXXXXXX")"
"$MODEL_PYTHON" -m pytest "$REPO_DIR/tests/test_fused_model.py" \
  "$REPO_DIR/tests/test_fused_session.py" "$REPO_DIR/tests/test_ge_fused_target.py" \
  -p no:cacheprovider --basetemp "$TEST_RUN/tmp" --junitxml "$TEST_RUN/results.xml"
```

## 用户回传的设备结果

旧 tiny 双图已回传加载与生成成功。普通/DFlash 在其测试中一致，但尚未与主机参考完全对齐。41个旧 GDN repair 用例全部编译/加载/运行，数值32通过、9失败；这些是用户粘贴报告，未在本次获取原始设备产物重新校验。

已收敛的结论：

- 所测 rank-3 padding/window 及 beta/decay 布局修复后，对照恢复；weighted value 和 scan 对照通过。
- 小幅度 Q/K 的 MatMul 归约丢值，原生 Sum 对照保留16/16个非零结果；正确 sum 输入下的加 epsilon/Rsqrt 对照通过。
- 原生归一化首层最终输出通过，max_abs=`0.000793457`、relative_L2=`0.005915196552830379`；分阶段 `normalized_key` 仍失败，max_abs=`0.00390625`。
- 部分 MatMul/BMM 边界用例仍丢小值。最终 BMM 对照虽然过原阈值，仍只有92/128个非零结果，relative_L2约3.81%；不能只看 PASSED，也不能据此把所有算子概括为同一种下溢机制。

这些结果结束了相应探针的反复定位，未改变原有数值阈值。它们不覆盖最终融合 GDN、完整模型、缓存连续生成或 DFlash 加速，执行位置也未独立确认。

## 后续验收

围绕实际采用的融合路径，集中验证算子接口、同权重目标输出、prefill/decode 状态连续性、部分接受回滚，以及普通/DFlash token 与停止原因一致性。最终性能比较必须匹配权重、量化、缓存策略、提示、生成长度和测量条件。完整模型质量与至少2倍加速目标均未验收。

## 历史归档

本次删去的探针源码、用例、旧文档和逐次验证 JSON，已与整理前的未提交更改一起归档到仓库外。归档有逐文件 SHA256 清单，包含原 HEAD `4a043641fb5872b5af24c39b86b4b86e70b77014`；已提交版本也可从 Git 历史找回。

归档文件为 `before-reorganization/source.tar.gz`，SHA256：

```text
4269cccd3b3a57f04a15e8b520fb81f248a77a441593fa8a3b2d4ba0651071c3
```

本地运行记录和归档定位见 [specs/verification.json](../../specs/verification.json)。源码仓库只保留当前结论与必要生产回归。


## 2026-10-30：复用现有 OMC 的短提示词执行主线

用户要求优先在当天接通真实完整 DFlash，当前里程碑允许精度误差／乱码。执行入口统一为 `run_serial_dflash_smoke.py`：新8-token提示词由同一原40层decode1从80个FP16零态逐token输入，随后生成8个token；原完整六层S64 draft以完整前缀右padding输入，每次15候选，host先验证前2。target-next anchor、首错纠正、全接受bonus及每个最终token的state advance沿用已有调度器。没有新AIR/OMG/OMC，没有裁剪context或重置RoPE。实验控制语义仍未验证。

真实执行验收需要至少两次完整draft调用、16次实际target decode1调用及所有输入／输出日志哈希。协议调度器prefill汇总为1次观察，不能误写成1次native调用；新backend保存全部8次prompt调用的组合receipt。只比较已考虑候选的匹配率，不把丢弃的未验证候选计作接受。EOS仅在固定预算执行实验中关闭。接受率0也允许完成执行里程碑；数值精度、普通基线等价、最终语言质量和性能仍保持未通过。

本轮主机验证覆盖真实89/82 extent、80state显式槽位、原INT8/FP32行查表、current metadata身份、native receipt与buffer回读、短prefix右padding、runner复用、不复制模型和失败停止。HDC/实际模型输出由明确的host doubles模拟；主机通过不是实机PASS。设备执行仍由用户在原DDK／手机环境运行。

row63独立prefill已由用户回传PID28669成功：82输出齐全，feature SHA `be3c0f2f2e76e2cb4d89ec484b0a31dfde8ed65a89ca5a2ef82ec7905370cbba` 与旧row64一致；logits SHA `c1701fb48439921032d481139c4a2284b9f402cad4e49e6f57236ca37e67bdfa`，argmax6。此单变量观察不作精度修复结论，也不成为短提示词新请求的历史状态。
