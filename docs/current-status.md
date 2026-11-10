# 当前阶段：Target 数值异常清单与剩余编译项

更新：2026-11-10。累计 **146/158 项编译成功，剩余 ConvState 12 项**。新增 Norm／Conv 26 项中，用户已回传 **22 项数值摘要（Norm 18、Conv1d 4）**，均为 `METRICS_RECORDED`、dtype 保持、重复稳定。结合原 120 项，目前 **142 项有数值回传：2 PASSED、2 FAILED、138 METRICS_RECORDED**；另 4 个已编译项待回传。旧 `environment-recheck-001/omc/compile.json` 保留原记录。

本次 Norm relative-L2 为 **0.039795%～0.189404%**，Conv1d 为 **0.021476%～0.027210%**。按用户安排，先降低这些小误差边界的排查优先级，集中解决剩余 ConvState 编译失败；保持原始数值状态，不新增验收门槛。详细范围、4 个待回传项及[原始 CSV](../specs/target-norm-conv-metrics-20261110.csv)见[Target OMC 与精度状态表](target-accuracy-status.md)。此前主机／runner 的 16 MiB 限制已修正为 32 MiB，本次收到结果的 22 项无需重跑。

| 当前分组 | 数量 | 结论 |
|---|---:|---|
| 输出明显异常 | 51 | 量化投影 38、RoPE 4、Prefill GDN 6、Prefill FusedAttention 1、Decode Cache K/V 2 |
| 数值零误差 | 36 | repeat Q/K 12、writeback 6、residual 16、Prefill Cache K/V 2；只有 Cache 两项原状态为 PASSED |
| 误差较小、仍待浮点验收 | 55 | 原 33 项，加本次普通 Norm 18、Conv1d 4 |
| 编译成功、数值待回传 | 4 | L0 Prefill input Norm、L3 Prefill Q Norm、L0 Prefill／Decode Conv1d |
| 编译失败、尚未数值验证 | 12 | ConvState，L0–L2，Prefill／Decode，两个输出 |

51 项是输出异常排查清单，不是已确认有 bug 的 51 个内核。小图使用各自的 310P 输入，不能把独立结果串成上游误差传播；重复稳定也不等于精度通过。数据来自用户粘贴的比较摘要，原始 native 输出与完整 report.json 未在本地主机重新核验。

用户确认 DDK／OMG 编译在 **35 服务器**，310P golden 仍在此前的**权重服务器**通过已跑通的四层推理入口采集；35 不是 Python 环境名，147 仅为此前参考命令来源。CPro dump 尚无样例。[离线批量比较入口](target-dump-compare.md)已完成主机验证，可从 capture 生成待绑定清单；真实文件格式和节点映射仍需首批 dump 元数据。本轮没有启动整图执行。

此前 Norm 的 Add 常量计算／tiling 和 Conv 的输入 rank 阻塞已通过。现已收到 [ConvState 失败小图与成功整图对照](target-compile-triage.md#成功整图提供的对照)：两者都有候选 store 拒绝／强制 NPU 分区，但成功 T64 整图随后调用专用 graph builder、保存 `executor_linear_attention_convstate_update` 并完成 OMC 保存；小图在 `generated_lib` builder／format config 处退出。实现存在，具体构建路径分歧原因仍未证实。下一步用 L0 Prefill／Decode 两张既有 AIR，仅省略小图显式添加的 `use_origin_format=true` 做[参数对照](target-compile-triage.md#下一步两个小图的格式选项对照)，尚未执行，不重编已成功的 146 项。[Norm／Conv 的执行与比较命令](target-norm-conv-validation.md)保留，4 个待回传项优先复用已有结果。本地主机只更新证据和命令；四层整图 dump、设备数值及连续状态反馈仍未验证。旧环境 GDN SIGSEGV 保留为历史记录，不代表本轮编译状态。

## 2026-11-09 独立诊断记录

以下保留当时的设备证据和复现命令；其中“下一步”和单算子优先级为历史安排，当前进度以上方异常与待编译清单为准。

更新：2026-11-09。按用户提供的patch使用master源码，先检查原40层Target实际使用的算子，覆盖Prefill64／Decode1。当前优先Linear Attention与Self Attention中的非矩阵乘算子；量化MatMul与MoE数值排查后移。暂停离线权重转置及其它Draft性能实验，保留最新packed基线。

Linear Attention按Convstate更新、卷积／SiLU、GDN、门控RMSNorm推进；Self Attention按Q/K RMSNorm、RoPE、CacheUpdate、融合Attention推进。先测状态与布局，再串联；GDN与CacheUpdate已有独立诊断入口，其余尚未数值验证。

## 本轮设备结果与下一步

用户回传`attention-validation-tbe/result.json`：四张单算子图均编译成功，TBE worker审计均为`APPLIED`。GDN与CacheUpdate各四例均完成设备执行和回收，native、archive及各case退出码均0；本地主机未连接服务器或重验原始产物。

GDN每例十次输出逐字节稳定且全部有限。以下为对独立参考的相对L2；尚无校准过的误差门槛，维持`METRICS_RECORDED`。

| GDN用例 | core_attn | last_recurrent_state |
|---|---:|---:|
| T64零初态 | 0.36166% | 0.06612% |
| T64非零初态 | 0.61187% | 0.08209% |
| T1零初态 | 0.05123% | 0.05066% |
| T1非零初态 | 0.05012% | 0.05029% |

这仍是六输入、FP16诊断，未验证master七输入接口、部分有效长度或连续状态回灌。T64 OMC SHA256为`c74fde9447192d08c9c72d0220c975f9280e322bfe64ff814497093b984c5fa4`，T1为`b96815646df5937aeaee3d6cb7cd4ec09130dfacb2b9b2e2215c9ecfb19b69d8`，来源为用户回传字段。

CacheUpdate四例均有限且十次重复稳定，但整缓存比较失败。用户回传离线逐位分析已确认：连续块T64和两个T1的写入区完全正确、未写区全为正零；置换块T64仅首个token写对，另63×512=32256个预期写入元素错误，并在未写区域出现32256个非零元素。

主机根据冻结夹具，独立模拟下列行为：初始输出全零；仅取`targetBlock[0]`、`offsetInBlock[0]`作为起点；第t个token写入`p=32*targetBlock[0]+offsetInBlock[0]+t`，即`out[p//32,:,p%32,:]=updates[t]`。四组生成结果的SHA256与用户回传的完整2MiB快照SHA256全部一致，不再仅是RMSE相同。它证明这四组观察与该行为模型一致，不能仅据此断言源码读取了哪些元素、是否另行修改输入，或该算子承诺支持任意分页写入。

| CacheUpdate用例 | 请求物理块 | 匹配的连续起点模型物理块 | 写入区错误数 | 未写区错误数 |
|---|---|---|---:|---:|
| T64连续块 | 0、1、2 | 0、1、2 | 0 | 1015808 |
| T64置换块 | 11、28、45 | 11、12、13 | 32256 | 1015808 |
| T1块尾 | 28 | 28 | 0 | 1048064 |
| T1块首 | 45 | 45 | 0 | 1048064 |

因此不能只处理未写缓存：即使保留旧数据，跨非连续物理块的T64仍需解决地址语义。当前探针的逐token索引、完整返回cache合同是待核对假设；测试失败不直接等于DDK内核违约。主模型使用RefData，诊断使用普通Data和独立输出缓冲区，这一差异仍需实测核对。GDN暂保留当前数值证据，不重复同组调用。

下一步只读取此前已扫描到的DDK实现，核对`Init`的输入／输出地址绑定、索引加载以及`Process`的跨块逻辑：

```bash
(
set -eu
CACHE_SRC="${CPRO_DDK_ROOT:?}/tools/tools_ascendc/custom_op/ascend310b/src/cache_update.cpp"
sha256sum "$CACHE_SRC"
sed -n '1,160p' "$CACHE_SRC"
)
```

请回传该输出。已有清单记录SHA256为`4cb690d632b569978547d19a50aec71e1b2f4d18f4ee9b5259d8eef84c532f12`；若已变更，需单独保留版本差异。源码仍是候选实现证据，不能替代编译绑定。根据结果选择按原位／起点约定修正诊断调用，或对实际合同内的实现问题作隔离修复；不修改共享DDK、不放宽整缓存门槛，也不把单token循环或整缓存复制直接放进生产模型。

离线分析入口`validate_target_attention.py --output <原验证目录> --analyze`已增加三个候选的逐位对照：逐token索引且输出从零开始、首索引连续写且输出从零开始、首索引连续写且保留原输入。它保留原来的数值失败结论；此次不用再次运行，四组快照hash匹配已在主机复现。11项主机检查通过，包括原9项，以及回传hash的独立标量地址复现和越界候选拒绝；使用合成主机数据，不是新设备结果。

## 直接执行Target单算子验证

此前编译失败已定位并修复：GDN和CacheUpdate的T64／T1均在TE worker的`_prerun → _update_options → te_update_version`退出。新单算子入口漏传了已有成功流程的`--tbe-diagnosis`；现已在两组探针、直接编译CLI及自动验证入口补齐传递。沿用`common/tbe-diagnosis`的已完成诊断，严格核验同DDK的包、库和平台初始化证据，仅在本次编译目录生成TBE副本并保留真实worker审计。OMG目标仍为`kirinx90`，不改算子输入／属性、不改共享DDK或已安装Python。

TBE接线修复先通过5项主机检查，随后用户完成真实DDK编译和设备验证，结果见上节。

不用重建已成功构建的native运行器，也不用重跑平台诊断。旧失败结果保留在`attention-validation`，修复后的验证使用`attention-validation-tbe`。本轮已完成，优先使用上方离线分析命令；下方保留完整执行命令供新批次使用，已有目录不会覆盖。`APPLIED`只证明worker使用了兼容处理，不等于数值通过。

使用已有X90运行环境及`runtime/runtime.json`。下面的入口连续完成夹具和独立参考生成、单算子编译、编译IO绑定、板端执行、结果回收和数值比较；不需要先回传编译摘要。默认依次执行GDN和CacheUpdate，不重编整模。

```bash
(
set -eu
cd "${REPO_DIR:?}"
git pull --ff-only origin main
: "${AI_RUN_DIR:?}" "${CANN_ROOT:?}" "${CPRO_DDK_ROOT:?}" "${SN:?}" "${BIP:?}"
python3 targets/cpro/tools/validate_target_attention.py \
  --config "$AI_RUN_DIR/runtime/runtime.json" \
  --tbe-diagnosis "$AI_RUN_DIR/common/tbe-diagnosis" \
  --output "$AI_RUN_DIR/target/tests/operator-accuracy/attention-validation-tbe"
)
```

运行器复用此次已成功构建的`runtime/native`。使用模型Python环境；GE头文件和`libgraph.so`来自`CANN_ROOT`，OMG来自`CPRO_DDK_ROOT`，DDK身份和运行时版本从已有runtime配置读取。`SN`、`BIP`、`HDC`沿用已工作的X90连接设置。运行配置只提供环境、固件身份和公共运行器，不执行其中的整模。

仅复用尚未编译或已经成功编译的探针时，在验证命令后加`--reuse-gdn "$AI_RUN_DIR/target/tests/operator-accuracy/gdn" --reuse-cache "$AI_RUN_DIR/target/tests/operator-accuracy/cache"`；已有失败编译不在原目录重试。只执行一组用`--families gdn`或`--families cache`。这些选项无需重新清点算子。一次编译部分失败时继续运行该组中已编译的用例，并明确报告覆盖不完整。

每例预热1次、记录10次完整输出。下发前按编译图的命名Data和GraphDef.input顺序重排输入，输出按编译NetOutput绑定参考；相同shape的Q/K/V不靠shape猜槽位。现有传输流程校验OMC、运行器和输入SHA256，回收时再次校验保留夹具。报告包含编译GraphOp归属；仅有HIAI执行不能宣称所有计算均在NPU上。

- GDN：T64／T1，零／非零非对称初态，共4例。遵循已报告DDK的六输入、双FP16输出接口；独立FP64递推参考，逐次报告core输出和末态的max-abs、RMSE、relative-L2、cosine及字节稳定性。尚未校准误差阈值，有限且稳定的输出记为`METRICS_RECORDED`；非有限值或重复不稳定明确失败。master七输入接口、部分有效长度和连续状态回灌尚未覆盖。
- CacheUpdate：T64跨31/32及63/64边界，连续／置换block表；T1分别写块尾和下一块首，共4例。参考按`cache[block,:,offset,:]=updates[token]`独立重建。整个FP16 cache逐位比较，分别报告写入区、未写区错误数及坐标。通过只证明这些用例的返回cache；alias/copy-back和移动位置的连续反馈仍需后续测试。

入口默认等待每组最多30分钟。命令中断或返回`PENDING`时，继续回收原请求：

```bash
python3 targets/cpro/tools/validate_target_attention.py \
  --output "$AI_RUN_DIR/target/tests/operator-accuracy/attention-validation-tbe" --resume
```

恢复不会重复下发已有请求。模型编译/执行失败保留日志和阶段；未确认旧任务结束时不会启动下一组。修复后使用新的验证输出目录。需要检查未清零输出时，另选输出目录并加`--output-initialization untouched`；这不等于已经完成00/A5/5A填充对照。

回传`attention-validation-tbe/result.json`即可，其中包含两组最终数值结果、逐次误差及详细回收路径。`VALIDATION_COMPLETE`表示已完成这组诊断流程；GDN的误差仍须判断，`all_operators_passed`保持false。13项主机检查通过，覆盖夹具、编译IO重排mock、回收包、误差检测、恢复流程和CLI运行器预检；本轮真实设备结果见上节，主机回归不替代该证据。

## 当前覆盖和结论

源码检查覆盖12类显式调用：Convstate、GDN、CacheUpdate、Attention、RMSNorm、DynamicQuant、QuantMatMul、MoE TopK、MoE InitRouting、GroupedMatMul、MoE FinalizeRouting，以及权重准备阶段的TransQuantParam。整理出68组带用途标签的逻辑shape；这不是68组已编译或已通过的用例，也不等于覆盖全部导出后算子。真实AIR还包含30个Conv2D和40个ReduceSumCpro，现单独记录为导出后发现的类别。注册的AdnRmsNorm在当前CPro主路径没有调用，实际Norm使用npu_rms_norm。

接口检查工具的28项主机检查通过；该清单工具本身不编译或运行设备。用户回传已确认两份真实AIR成功解析；源AIR描述仍不能替代编译后接口与设备数值。

此前清单报告中的`PARTIAL`仅由两个预设候选目录缺失触发（`ddk/ascendcpp`、`tools/platform/kirinx90`）；两份AIR均为`GRAPH_OBSERVED`，扫描没有读取错误。这不表示DDK损坏或算子精度失败。

此前的`hiai.proto.AttrDef.list_list_float` wire-type错误已通过优先使用离线TorchAir `ge.proto` schema修复；两份真实AIR均以`BUNDLED_TORCHAIR_GE`成功解码。报告保留实际schema及失败原因。

目前优先保留以下问题线索：

- 两份AIR各有30个GDN。展示的代表节点都包含7个输入，`effective_length`为INT16[1]，`g`和末态描述为FP32，初态FP16；Prefill／Decode的chunk_size分别64／1。回传DDK注册和入口均为六输入、双FP16输出。差异已从源码线索推进到源AIR证据，但仍需核对编译处理，不能直接删掉有效长度或宣称内核数值故障。
- 两份AIR均确认351个`QuantBatchMatmulV4444`、80个`GroupedMatmulV555`、40个`MoeInitRoutingV3`与40个`MoeFinalizeRoutingV2`。这些已不是仅DDK目录中的候选名；具体编译实现仍未绑定。
- 用户已确认INT8／INT32是INT4数据的承载容器，不能由容器dtype或`QBMInt8Compute`类名判断成W8。scale同样是编码参数，用户描述为类似FP32、尾数10位；INT32／INT64标签不代表整数数值，不能用数值`.float()`转换当作解码。后续若回到矩阵乘排查，应核对打包／解包、逻辑K/N、布局与分组scale，具体位分布与舍入规则仍以实现为准。
- 历史GDN结果受输出缓冲区00／A5／5A填充值影响，零填充下state相对L2仍约0.146／0.155。来自旧版本用户回传，根因未定位，不能视为当前master内核已确认失败或已修复。
- CPro量化MoE在`pertoken_scale.dim() > 1`时执行`x_quant = x.view(...)`，覆盖动态量化结果；两份AIR展示的GMM代表节点都直接连接DynamicQuant整数输出，未支持该分支在这些节点触发。全部节点及真实输入仍未验证，此项暂后移，保留master代码。
- Convstate控制数组和输出rank、GDN末态dtype需与当前DDK核对。CacheUpdate已有本轮整缓存数值失败证据；Attention仍仅有旧shape检查，精度未通过。
- master的`quant_bits`默认8，使用原W4数据导出时需显式指定4。旧AIR不能作为新源码已经重新导出的证据。

## 已有清单的复查（可跳过）

已有本轮`inspection/inventory.json`时直接复用它，不必重新提供源码目录或重扫DDK。以下命令只重读其中已绑定的AIR，追加注意力输入生产者和输出消费者信息；保留Cast／Bitcast，不把空shape当标量，不由载体dtype推断计算位宽。

```bash
(
set -eu
cd "${REPO_DIR:?}"
git pull --ff-only origin main
INSPECTION_DIR="${AI_RUN_DIR:?}/target/tests/operator-accuracy/inspection"
python3 targets/cpro/tools/inspect_target_operators.py \
  --from-inventory "$INSPECTION_DIR/inventory.json" \
  --output "$INSPECTION_DIR"
)
```

这次回传`inspection/review.json`即可。重点是注意力算子的声明差异、输入连接与状态输出的直接消费者；完整节点仍在inventory中。源AIR未推导出的shape保持未知；源AIR描述和注册表不同，只记为待编译核对，不记为设备ABI失败。复用模式保留上次DDK声明快照；DDK或AIR已更换时再用下方完整命令重新清点。

在已有DDK服务器环境运行下面的接口检查。`REPO_DIR`是包含新入口的本工程目录，`TARGET_SOURCE_ROOT`是master原工程的`qwen3_5_moe`目录，`AI_RUN_DIR`是现有运行根，`CPRO_DDK_ROOT`是本次选定的X90 DDK。Python需为本工程声明的模型Python。命令不读取权重payload，不编译、不启动设备。

```bash
(
set -eu
cd "${REPO_DIR:?}"
git pull --ff-only origin main
: "${AI_RUN_DIR:?}" "${CPRO_DDK_ROOT:?}" "${TARGET_SOURCE_ROOT:?}"
python3 targets/cpro/tools/inspect_target_operators.py \
  --source-root "$TARGET_SOURCE_ROOT" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --prefill-air "$AI_RUN_DIR/prefill/air" \
  --decode-air "$AI_RUN_DIR/decode/air" \
  --output "$AI_RUN_DIR/target/tests/operator-accuracy/inspection"
)
```

上面使用现有运行根的`prefill/air`、`decode/air`目录。目录中只有一个`.air`时直接读取；有多个时会列出候选，此时将参数改为明确的文件路径。也可直接传入位于其它位置的AIR文件。默认优先使用本工程离线TorchAir schema，成功时显示`schema_source=BUNDLED_TORCHAIR_GE`；也可用`--torchair-pb2 <实际ge_ir_by_protoc_*_pb2.py>`显式指定。暂时只有旧AIR时可以用于接口观察，报告始终标记源码版本绑定未验证，不要求先重编整模。

结果为`inspection/inventory.json`与`inspection/summary.json`，请回传后者。新版summary包含完整GDN端口和dtype、注册签名、源码Meta、自动输入名差异，以及GDN／Cache／量化MatMul的内核入口；省去重复用例列表。AIR摘要保留每个实际GE类型的首个可达节点及输入／输出描述，全部节点和不同shape／dtype仍保存在inventory中。全部算子的完整JSON注册记录保存在inventory的`ddk.registrations`中，dtype／format的逗号分隔组合保持原顺序。

`INVENTORIED`仅表示所请求的清点完成；`PARTIAL`／`SOURCE_REVIEW_REQUIRED`会列出缺失目录、解析错误或未知调用。目录缺失单独标为`CANDIDATE_ROOTS_ABSENT`；声明差异单独记录在`contract_review`，不作为已验证的设备ABI故障。工具不会导入DDK Python、自动修改输入端口或重选算子实现。所有数值状态仍为`NOT_RUN`。重复执行覆盖同一位置的清单和摘要。

## 尚未完成时的Draft归档

保留 `offline-weights/full/baseline/model.omc`，1791018585字节，SHA256 `bc693db0bc3833d532f6b2158b3f1f19939311beb6941ff43c8c05b9aa4631dd`。最新38次调用加权准备＋API为60.4067ms；nd_n候选73.6574ms，不采用。两模型原数值门槛仍FAILED，488个对照区域中只有isolated第0例hidden不同；其余487个区域相等。详细限制见[结果](results.md)与[优化策略](optimization-strategy.md)。服务器归档回执尚未收到；已经完成时无需为本轮算子检查重复执行以下操作。

在已有产物的DDK服务器执行。该命令不导出、不编译、不启动设备：先从现有编译文件导入一次ABI绑定、保存紧凑对照摘要，将AIR同盘移动到`draft/air`，将自包含OMC及必要元数据发布到`draft/omc`，确认能从公共目录重新绑定后，再清空 **整个 `draft/tests` 的全部内容**。

```bash
(
set -eu
cd "${REPO_DIR:?}"
git pull --ff-only origin main
: "${AI_RUN_DIR:?}" "${CPRO_DDK_ROOT:?}"
full="$AI_RUN_DIR/draft/tests/offline-weights/full"
python3 targets/cpro/scripts/archive_incremental_draft.py \
  --root "$AI_RUN_DIR" \
  --compiled "$full/baseline" --air-export "$full/export" \
  --source "$AI_RUN_DIR/draft/source/prepared" \
  --comparison "$full/nd_n-result" --ddk-root "$CPRO_DDK_ROOT" --apply
python3 targets/cpro/tools/clean_draft_probes.py \
  --root "$AI_RUN_DIR" --source "$AI_RUN_DIR/draft/source/prepared" \
  --all-tests --apply
df -h "$AI_RUN_DIR"
)
```

归档应返回`ARCHIVED`或`ALREADY_ARCHIVED`，清理返回`CLEANED`或`ALREADY_CLEAN`。归档失败时`set -e`会阻止清理。已完成后重复执行不需要原tests输入；中途移动失败按同一命令恢复，不另建版本目录。服务器物理迁移与释放空间以该命令回执为准，本地没有访问服务器或设备。

最终保留：

```text
draft/source/prepared/  原始准备权重
draft/air/              当前export.air及引用权重
draft/omc/              当前model.omc、pytorch-abi.json、compiler.air、manifest.json
draft/archive.json      基线身份、已有数值状态及本轮对照摘要
draft/tests/            空目录
```

权重、仓库代码与摘要都不留在tests里。旧探针、nd_n候选、编译副本、输入／golden、快照和历史结果全部随tests删除；不把它们整体搬进archive。若旧runtime配置仍引用被替换的公共模型，工具仅保留该真实依赖于`.omc-previous`／`.air-previous`并调整原配置引用。当前packed模型的host embedding接口尚未接入旧token-ID常驻循环，不把归档当作运行接口切换。

## 后续工作

正常Draft生成入口已改为公共`draft/air`、`draft/omc`，不再把可复用模型发布在tests中；从零配方见[完整流程](run-end-to-end.md)。本次直接保留已有模型，不重建。

Target按单算子→Conv/GDN、Cache/Attention、MoE串联→真实decoder block→40层Prefill64／Decode1推进；通过后再恢复DFlash的批量verify与端到端接受率工作。每一步分别报告编译、执行和数值状态。
