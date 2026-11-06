# Target精度定位交接（2026-11-06）

当前任务是定位原40层量化Target的精度。用户要求优先抽取实际使用的自定义算子，组成小模型，使用与整模相同的shape验证；后续实验在新窗口继续。约60ms的Draft保留为性能基线，离线权重转置和其它Draft性能实验暂停。本次完成源码核对和交接，没有新增模型编译、设备执行或精度通过结论。

## 1. 新窗口从哪里继续

本地**确实保留两份完整建模文件**，不只是文件名或哈希记录。2026-11-06重新读取原6份源码／配置，现有审查入口返回`SOURCE_AUDIT_MATCHED`，未导入模型、未读权重。

当前机器的工作区根为`/home/cann/workspaces/ai-model-workspace`，以下路径相对此根：

| 资源 | 位置 |
|---|---|
| 当前开发checkout | `.work/qwen3.5-35B-moe/20261029T021635Z-30632-21a345/checkout` |
| 用户原始私有源码根 | `.work/qwen3.5-35B-moe/20261027T130835Z-9607-68b43f/private-reference/qwen35_moe` |
| NZ建模 | 上述源码根的`models/modeling_qwen3_5_moe_hiai_nz.py` |
| CPro建模 | 上述源码根的`models/modeling_qwen3_5_moe_hiai_cpro.py` |
| loader／wrapper／converter | 同根`utils.py`、`models/export_model_wrapper_qwen3_5_moe.py`、`export_glm_model_qwen3_5_moe.py` |
| 本次6文件审查结果 | `.work/qwen3.5-35B-moe/20261106T074803Z-29115-c19d48/out/target-source-audit.json` |
| 声明的模型Python | `shared/envs/qwen3.5-py310/python` |

两份文件身份：

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| `modeling_qwen3_5_moe_hiai_nz.py` | 45638 | `f7ea28355ffa2fa2d92b2e1d171041ce72366c9437ecd68554797e961b0d6739` |
| `modeling_qwen3_5_moe_hiai_cpro.py` | 45698 | `e994b0e194efee0ade63478647f0b11acd8a8bc40ba9b188a001a12f7d20651f` |

原源码继续作为私有资源留在Git之外，不复制到公开仓库。`models/qwen3.5-35B-moe`是薄工作区，不能把其中旧源码当作当前checkout；`.work`内同名pytest夹具也不是模型源。当前分支`operator/draft-top1`，本次文档修改前HEAD为`7015530`且工作区干净；该提交已包含Draft归档实现。本文提交后的版本以`git log -1`为准。

用户报告NZ版本在310P短序列能正确吐字，这是重要的对照来源；本次没有复现，也尚未把该310P运行的checkpoint、源码版本和输入快照与上述文件绑定。新窗口先核对对应版本，再采集同输入算子输出。不要把“短序列文本正确”扩大为全部长度／全部中间张量精度通过。

可直接开始的本地只读命令（本机路径，换机器时重设）：

```bash
workspace_root=/home/cann/workspaces/ai-model-workspace
repo_dir="$workspace_root/.work/qwen3.5-35B-moe/20261029T021635Z-30632-21a345/checkout"
source_root="$workspace_root/.work/qwen3.5-35B-moe/20261027T130835Z-9607-68b43f/private-reference/qwen35_moe"
cd "$workspace_root"
kit/bin/ws context qwen3.5-35B-moe --task model --target cpro
cd "$repo_dir"
git status --short
"$workspace_root/shared/envs/qwen3.5-py310/python" -B \
  targets/cpro/scripts/append_original_ge_features.py \
  --inspect-source --source-root "$source_root"
```

随后读本页、两份建模文件及其loader／converter，生成**一份统一算子用例矩阵**；再实现小模型导出与比较入口。新的生成文件先按工作区规则创建run。本地尚未实现本轮完整真实shape精度套件，不存在一条已验证的“运行所有Target算子”命令。

## 2. 当前整模与平台边界

Target是原量化工程导出的完整40层模型：每4层中3层GatedDeltaNet、1层full attention，总计30＋10层；hidden=2048，词表248320，256个专家、每token选8个，专家／共享专家中间维512。Full attention为Q16／KV2、head_dim256；GDN为Q/K16、V32、head_dim128、卷积宽4。配置来源是`specs/target-config.json`和实际原工程配置，部署KV容量2048。

现有主线采用分别导出的Prefill64、Decode1 AIR／OMC、原量化外置权重和HIAI V2。原接口89输入、81输出，追加8层真实feature后82输出；80路状态是30个conv state、30个recurrent state、20个K/V cache。具体编译后的顺序／物理布局必须由各自ABI绑定，不能仅凭大小或输出编号推断归属。

原Target、Draft及常驻文字循环已有用户回传的执行证据；**Target普通生成精度仍未闭环，端到端数值／接受率／配对加速均未验收**。仓库里的functional PyTorch／GE Target适配器是另一条实现，其状态ABI与原89输入接口不同；不能无说明地拿它代替本次CPro原模型。

目标为CPro X90／`kirinx90`，最近实测DDK为`100.520.020.010`。310P用于参考，不将其kernel、算子注册或物理格式直接当作CPro支持证据。TE日志中的`Ascend310B1`也不能用于改掉OMG的`--platform=kirinx90`。复用已有`common/tbe-diagnosis`做需要的进程内适配；不修改共享DDK。

用户负责服务器／设备执行，本地不连接服务器或探测板卡。此机器`ws env ... --target cpro`目前因缺少`cpro-blue-pending` profile manifest失败；不等于服务器DDK不可用。本次只读源码审查已用声明的模型Python完成。实际测试需在用户现有导出／DDK环境执行，先记录软件与算子版本。

## 3. 两份源码已经确认的差异

以下是本次直接读取文件后的静态差异，行号对应上面的固定SHA；它们是实验入口，尚不是已定位根因。

| 位置 | NZ（310P参考路径） | CPro导出路径 | 首要核对 |
|---|---|---|---|
| 普通RMSNorm，NZ:157–159／CPro:116–118 | 输入和gamma显式FP32，结果转回输入dtype | 直接传hidden及`weight+1` | 实际入参dtype、累加及输出舍入；含H2048和Q/K的D256 |
| 卷积状态，NZ:648–654／CPro:644–652 | `causal_conv1d_update`或本文件fallback | `LinearAttentionConvstateUpdate`＋depthwise Conv1d＋SiLU＋copy_ | 控制数组、旧state顺序、两个输出和状态回写；NZ需记录实际选中的分支 |
| GDN，NZ:670–683／CPro:667–680 | `initial_state.to(float32)` | 直接传recurrent_state | 当前导出state为FP16；g/beta端口与内部Cast也需逐一绑定 |
| Q/K重复 | `repeat_interleave` | reshape＋concat的`repeat_qk` | 32个head的排列逐字节核对 |
| KV更新，NZ:332–348／CPro:283–296 | 整数整除／取余 | FP32乘逆数、减法后转INT32 | block边界、block table、offset和实际CacheUpdate端口 |
| MoE路由，NZ:976–982／CPro:975–983 | FP32 softmax＋topk | `npu_moe_gating_top_k_softmax` | 专家集合、并列规则、归一化与输出dtype |
| W4A8专家GMM，NZ:793–815／CPro:790–815 | 两次GMM显式传bias；中间`npu_swiglu` | 不传bias；中间拆分＋SiLU×up | bias实际是否全零；不能仅凭注册参数存在判断影响；激活切分顺序和精度 |

另有必须保留的source问题：原loader对norm参数加1，普通NZ/CPro forward又加1；gated norm则直接用已加载gamma。只在两段加载／执行语句均实际生效时才得到相应的双加／折叠结果。应读取310P成功运行的**已加载gamma**和导出常量确认，不能在建立对照之前顺手“修正”。旧[算子分析](reference/qwen35-moe-operators-and-quantization-analysis.md)包含不同变体及勘误，当前真实调用优先于旧表。

## 4. 统一小模型覆盖表

“自定义算子”包括用户converter显式生成的算子和模型调用的厂商融合／量化算子；不是只测仓库自己写的StableArgMax。以下是起始清单，最终以**实际AIR可达节点与源码调用**合并、去重，按不同shape、dtype、layout、属性、权重处理及状态模式分组。同shape不同层的真实数据仍要覆盖，不只取首层。

设`T=64/1`、batch=1。表内逻辑形状来自源码／配置，物理shape、端口dtype和输出extent必须再核对实际导出与编译ABI。

| 优先级／算子族 | 整模尺寸与关键合同 | 比较内容 |
|---|---|---|
| P0 `LinearAttentionConvstateUpdate`，30层 | x `[1,T,8192]`，旧state `[1,8192,4]`，INT32控制 `[128]`；source meta输出state `[1,8192,4]`、padded `[1,8192,T+3]` | 两个输出、未覆盖区、copy_；再与Conv1d `[8192,1,4]`、groups8192和SiLU组成最小串联图 |
| P0 `ChunkGatedDeltaRule`，30层 | Q/K重复后及V `[1,T,32,128]`，g/beta `[1,T,32]`，state `[1,32,128,128]`；chunk64/1、INT16 effective_length、内部QK L2开启 | core_attn及最终state；零／非零state、FP32参考和原FP16回写边界；先查下节旧输出填充异常 |
| P0 `npu_rms_norm`与gated norm | 主norm `[1,T,2048]`，Q/K norm `[1,T,16/2,256]`；gated按实际reshape通常为`[T×32,128]` | 原gamma、eps=1e-6、入参／输出dtype；gated还比较FP32 SiLU乘gate与FP16回写 |
| P0 `DynamicQuant`＋`npu_quant_matmul` | dense投影实际`M,K,N`；W4 group256、原INT32载体、scale与逐token scale | 量化整数、scale、MatMul输出分别比；输入为真实FP16/FP32边界，不重新量化权重 |
| P0 `MoeGatingTopKSoftmax` | router logits `[T,256]`，k=8，保留随后可选归一化 | 原FP32 softmax＋topk对照，专家ID／概率；近并列单独记录，不能仅比较最终hidden |
| P1 `CacheUpdate`，20路 | cache `[64,32,32,16]`，updates由`[1,T,2,256]`展平为`[T,32,16]`，block_size32 | 整个cache精确写入／未写区保持；31→32、63→64及多次写入，沿真实调用的拆分规则 |
| P1 `AdnFusedInferAttention`，10层 | Q物理 `[1,256,T,16]`，K/V为上述分页cache；mask `[1,1,T,2048]`，Q16/KV2、D256、scale=1/16、BNSD、inner_precise2 | 有效query输出、mask与位置、有效KV长度；独立已填cache测试后再接CacheUpdate |
| P1 `MoeInitRoutingV2` | `[T,2048]`＋`[T,8]`专家ID，expert范围256，quant_mode=-1，tokens_per_expert转INT64 | expanded rows、row map、256组计数及索引合法性；语义顺序允许差异时先还原再比 |
| P1 `npu_grouped_matmul`，每层两次 | 第一段K2048→N1024，第二段K512→N2048，256专家；展开行通常8T（512／8），实际分组counts为准 | 原INT32 W4及INT64 qparam scale、group_list_type1／split_item3／group_type0；空专家、偏斜路由、非整齐M；保留完整专家维度 |
| P1 `MoeFinalizeRouting` | expanded `[8T,2048]`＋row map＋`[T,8]`权重，drop_pad_mode2 | 单独加权还原及Routing→GMM→Finalize串联；最后接shared expert和gate |

Dense量化矩阵至少覆盖：GDN qkv `2048→8192`、z `2048→4096`、a/b `2048→32`、out `4096→2048`；full attention q+gate `2048→8192`、k/v `2048→512`、o `4096→2048`；router `2048→256`；shared gate/up `2048→512`、down `512→2048`、gate `2048→1`；词表head `2048→248320`（实际选择输出行后M可能为1）。这些是逻辑投影，是否走QLinear、实际M和特殊padding以loader及AIR为准，不能把所有层都强行替换为同一量化调用。

Attention converter的特殊映射必须保留：源Python在export时用`pse_shift=allQLen`传参，但converter把它写到GE **`all_seq_lengths_q`**，GE `pse_shift`为空；actual q/kv lengths转INT64。不要根据Python参数名字接错GE端口。原源码声明kv长度为容量2048，实际可见范围还依赖mask／控制，不能擅自改成测试时的有效长度。

`AdnRmsNorm`虽有注册converter，当前两份建模的活路径实际调用`npu_rms_norm`；先看AIR再决定是否纳入。`StableArgMax`是Draft优化，不是本轮Target算子起点。CPro量化MoE中的SiLU已展开，NZ的`npu_swiglu`用于对照；不要把未激活分支统计成实际CPro算子。

## 5. 现有探针结果与必须避免的误判

历史结果记录在`specs/model-workflow.json → paths.ge_stateful_target.fused_operator_probes`，不是本轮真实shape验证的通过凭据：

- 2026-10-27 GDN Prefill／Decode曾完成编译、加载、Process，但core输出全零且参考非零。后续同OMC／同输入的`0x00/0xA5/0x5A`输出填充诊断中，输入buffer回读均匹配，**有效输出区域依赖填充模式**。零填充下core相对L2约0.003919／0.056429，state约0.146377／0.155230；其它填充误差骤增。根因仍未定位，可能涉及输出覆盖、累加前置条件或可见性；不能写成已证实某个kernel错误，也不能用清零来掩盖。
- 当时CacheUpdate／Attention比较标注`SHAPE_ONLY_UNVERIFIED_CONTRACT`，不能算精度通过。Convstate／AdnRmsNorm有历史编译失败，也不能推出当前CPro活路径或全型号不支持。
- 上述设备结果来自用户回传摘要，本地没有重新取得原始张量校验。新套件需要检查旧输入／算子／ABI与当前模型是否一致，不直接复制旧fixture或盲目重跑旧批次。
- Draft实验已证明小模型正确不保证整模正确；增加观察输出、改dtype、拆图都可能改变编译分区与数值路径。独立算子通过后仍需串联和整模复核。

## 6. 新窗口的验证方法与交付顺序

1. **冻结调用合同。** 对齐310P成功版本、当前CPro源码、loader、导出converter与实际AIR；列全可达算子及不同签名。记录逻辑／物理shape、dtype、layout、属性、权重与scale处理、输入／输出顺序、状态别名和控制数组含义。先解决未知合同，不用全零占位或仅shape相同的参考冒充golden。
2. **采集真实输入并建立双参考。** 优先从310P成功的短序列Prefill和连续Decode采集每个算子的已加载权重、输入和状态；用同一组逻辑输入跑CPro最小图，同时使用独立CPU／框架数学参考。必要的跨布局转换逐元素可逆验证。参考必须匹配原量化／gate／norm公式；NZ与CPro有意不同的表达单独对照，不能直接要求两套整图逐字节相同。
3. **一次设计完整矩阵。** 覆盖实际T64/T1及真实层数据、有效长度和尾部、零／非零历史状态、连续状态反馈、KV块边界、MoE空专家／偏斜计数、量化极值和近并列路由。加入可识别索引、单位基等布局诊断数据；常量／全零／退化shape不能是唯一数据。小模型保留整模尺寸和原外置权重格式，不以缩维结果替代真实shape；按组串行加载控制内存和磁盘。
4. **先定数值门槛再执行。** 沿用已有算子明确门槛；没有的在测试配置中先写明。浮点报告finite、max_abs、逐元素超限数、相对L2和首差位置，不以单个L2代替验收；同时报告对NZ参考和数学参考的误差。搬运、布局转换、索引、未更新state区域要求精确；路由并列与整数重量化的规则需明确。跨硬件浮点输出不默认逐字节相同，也不因失败放宽阈值。
5. **执行与准确性分开判定。** 保留compile/load/process/retrieve/numerical状态。首批包含GDN输出覆盖诊断：有效区域必须按已确认算子合同写完；若需要预初始化，查清API约定并复现实模初始化。不同输出填充值不能改变本应覆盖的结果。固定输入重复，另做反馈调用；`exitcode=0`或重复稳定都不是精度通过。
6. **按依赖闭环。** 单算子→Convstate/Conv/GDN链、Cache/Attention链、MoE链→一个真实decoder block→40层Target的普通Prefill64→Decode1。最后才恢复Draft接受率及端到端性能。若全部算子通过，继续查边界Cast、量化scale装载、norm gamma、mask／位置、state归属与回写，不把“自定义算子导致错误”当作预设结论。

新套件使用一个配置和固定执行槽位，例如`$AI_RUN_DIR/target/tests/operator-accuracy/`，区分编译、运行、回收、比较；这只是建议位置，入口尚待实现。失败先回收同一请求，确认退出后复用目录，不反复生成`v1/v2/...`。单算子临时产物不复制全套Target；现有Prefill／Decode公共模型不动。结束的测试按用户清理偏好删除，代码／权重在各自正式目录，简短结论保留在文档。

可复用的代码入口：

| 文件／目录 | 用途与边界 |
|---|---|
| `model/qwen35_moe_dflash/target/original_ge_target.py` | 6文件身份及`required_operators()`调用合同；builtin实际GE类型仍需导出确认 |
| `targets/cpro/scripts/append_original_ge_features.py --inspect-source` | 当前可执行的只读源码核对 |
| `model/qwen35_moe_dflash/target/original_air_inspect.py`、`original_target_contract.py` | 原AIR与接口检查；schema必须匹配 |
| `reference/fused_ops.py` | 已有数学参考，使用前对照本轮实参／边界，不作为自动可信的旧golden |
| `targets/cpro/tools/tensor_probe_main.cpp`及`build_hiai.py` | 现有小模型runner基础；当前声明dtype不覆盖本轮全部INT8／INT16量化接口，须扩展并验证，不能直接保证可用 |
| `targets/cpro/tools/probe_draft_vector.py` | 可借鉴编译／回收／比较分层；其Draft固定合同不能直接套到Target |
| `targets/cpro/scripts/compile_full_target.py` | 原Target整模编译／恢复；首轮不必重编整模 |

## 7. Draft存档与清理交接

保留packed基线1791018585字节，SHA256：
`bc693db0bc3833d532f6b2158b3f1f19939311beb6941ff43c8c05b9aa4631dd`。

38次诊断调用加权准备＋同步API为**60.4067ms**；离线W.T候选73.6574ms，增加13.2507ms，已暂停。该计时含CPU pack／SetInput／ReuseInput，不含文件I/O、KV快照、commit或加载，不是纯NPU或端到端速度。两模型原CPU精度均FAILED；488个对照区域只有isolated第0例hidden不同，其余487个相等。单独NZ导入实验也未建立可用合同，不再继续。

归档／清理代码已在`7015530`提交，16项主机检查通过，真实服务器迁移回执尚未收到。新窗口先检查`draft/archive.json`及实际目录；**不能说已经迁移或删除，也不能假定旧tests还在**。未完成时使用[当前状态](current-status.md)中已有归档后清理命令，无需重建模型。

```text
draft/source/prepared/   原始准备权重
draft/air/               当前AIR及其权重引用
draft/omc/               当前OMC及必要ABI／manifest
draft/archive.json      紧凑基线摘要，放在tests之外
draft/tests/            用户要求全部清空，不留源码、权重、摘要或测试结果
```

当前packed Draft是17输入／14输出、C16／cache192／K15、host FP16 embedding＋CPU feature pack＋NPU StableArgMax；与旧token-ID常驻接口不同，归档不等于生产接口切换。结构及逐项优化收益已经写在[模型结构](model-architecture.md)、[优化策略](optimization-strategy.md)、[结果](results.md)，无需新窗口重做Draft实验。

## 8. 本轮完成范围与经验记录

本轮完成：私有6文件实际哈希复核、NZ/CPro关键调用静态对照、历史Target／Draft证据分级、下一轮真实shape算子覆盖计划和文档导航。6份源文件身份、18个本地文档链接、3份状态JSON检查通过，记录在同一run的`out/handoff-checks.json`。只验证本地源码及文档；没有读取学习权重payload、编译模型、运行310P或X90。当前失败仍是模型精度未闭环，新的Target精度矩阵尚未执行。

本轮捕获的norm dtype、GDN state dtype、MoE bias／路由／激活差异是模型专属观察，保存在本页；旧GDN输出填充依赖仍为待定位现象。没有新的通用skill修改，也没有候选被提升为精度通过或生产版本。后续沿用`deliver-model-end-to-end`和`develop-model-on-x90`；真正进入某个算子的实现修改时再按`develop-custom-operators`界定任务。

可直接粘贴到新窗口：

> 请先读`docs/target-accuracy-handoff.md`。暂停Draft及离线权重性能工作，保留约60ms基线。使用已保留的NZ/CPro原始源码，先核对310P短序列成功版本，再逐项提取CPro实际自定义／融合／量化算子，按整模Prefill64／Decode1真实shape、原权重和状态生成统一精度测试矩阵。重点检查GDN输出覆盖及初始state精度、Convstate、RMSNorm、MoE路由／GMM以及CacheUpdate／Attention。先提交可复用的小模型导出／比较流程和用户执行命令，不连接服务器／设备，不改数学公式或放宽门槛，不重启Draft优化。
