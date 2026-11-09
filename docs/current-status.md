# 当前阶段：检查Target自定义、融合与量化算子

更新：2026-11-09。按用户提供的patch使用master源码，先检查原40层Target实际使用的算子，覆盖Prefill64／Decode1。当前优先Linear Attention与Self Attention中的非矩阵乘算子；量化MatMul与MoE数值排查后移。暂停离线权重转置及其它Draft性能实验，保留最新packed基线。

Linear Attention按Convstate更新、卷积／SiLU、GDN、门控RMSNorm推进；Self Attention按Q/K RMSNorm、RoPE、CacheUpdate、融合Attention推进。先测状态与布局，再串联；GDN与CacheUpdate已有独立诊断入口，其余尚未数值验证。

## 当前覆盖和结论

源码检查覆盖12类显式调用：Convstate、GDN、CacheUpdate、Attention、RMSNorm、DynamicQuant、QuantMatMul、MoE TopK、MoE InitRouting、GroupedMatMul、MoE FinalizeRouting，以及权重准备阶段的TransQuantParam。整理出68组带用途标签的逻辑shape；这不是68组已编译或已通过的用例，也不等于覆盖全部导出后算子。真实AIR还包含30个Conv2D和40个ReduceSumCpro，现单独记录为导出后发现的类别。注册的AdnRmsNorm在当前CPro主路径没有调用，实际Norm使用npu_rms_norm。

接口检查工具的28项主机检查通过，尚未在本轮编译或运行设备。用户回传已确认两份真实AIR成功解析；源AIR描述仍不能替代编译后接口与设备数值。

当前回传的`PARTIAL`仅由两个预设候选目录缺失触发（`ddk/ascendcpp`、`tools/platform/kirinx90`）；两份AIR均为`GRAPH_OBSERVED`，扫描没有读取错误。这不表示DDK损坏或算子精度失败。

此前的`hiai.proto.AttrDef.list_list_float` wire-type错误已通过优先使用离线TorchAir `ge.proto` schema修复；两份真实AIR均以`BUNDLED_TORCHAIR_GE`成功解码。报告保留实际schema及失败原因。

目前优先保留以下问题线索：

- 两份AIR各有30个GDN。展示的代表节点都包含7个输入，`effective_length`为INT16[1]，`g`和末态描述为FP32，初态FP16；Prefill／Decode的chunk_size分别64／1。回传DDK注册和入口均为六输入、双FP16输出。差异已从源码线索推进到源AIR证据，但仍需核对编译处理，不能直接删掉有效长度或宣称内核数值故障。
- 两份AIR均确认351个`QuantBatchMatmulV4444`、80个`GroupedMatmulV555`、40个`MoeInitRoutingV3`与40个`MoeFinalizeRoutingV2`。这些已不是仅DDK目录中的候选名；具体编译实现仍未绑定。
- 用户已确认INT8／INT32是INT4数据的承载容器，不能由容器dtype或`QBMInt8Compute`类名判断成W8。scale同样是编码参数，用户描述为类似FP32、尾数10位；INT32／INT64标签不代表整数数值，不能用数值`.float()`转换当作解码。后续若回到矩阵乘排查，应核对打包／解包、逻辑K/N、布局与分组scale，具体位分布与舍入规则仍以实现为准。
- 历史GDN结果受输出缓冲区00／A5／5A填充值影响，零填充下state相对L2仍约0.146／0.155。来自旧版本用户回传，根因未定位，不能视为当前master内核已确认失败或已修复。
- CPro量化MoE在`pertoken_scale.dim() > 1`时执行`x_quant = x.view(...)`，覆盖动态量化结果；两份AIR展示的GMM代表节点都直接连接DynamicQuant整数输出，未支持该分支在这些节点触发。全部节点及真实输入仍未验证，此项暂后移，保留master代码。
- Convstate控制数组和输出rank、GDN末态dtype需与当前DDK核对。旧CacheUpdate／Attention只有shape检查，精度未通过。
- master的`quant_bits`默认8，使用原W4数据导出时需显式指定4。旧AIR不能作为新源码已经重新导出的证据。

## 用户现在执行什么

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

## GDN单算子对照

`probe_target_gdn.py`已准备两张独立图（T64／T1），每张图各有零态与非零非对称态夹具。图严格使用回传DDK的六输入、双FP16输出声明，32个head、128维状态；这项诊断不修改master，也不代表已解决master七输入接口差异。独立NumPy FP64递推参考与现有Torch FP32参考已做四组交叉检查，公共输出均转为FP16，最大相对L2差异约`1.12e-5`；该结果只验证主机参考，不是设备误差门槛。

在已激活的本工程模型Python环境中执行。`CANN_ROOT`提供GE导出所需的头文件和`lib64/libgraph.so`；`CPRO_DDK_ROOT`和`CPRO_DDK_ID`使用已选定的X90环境。准备只依赖NumPy；编译使用现有GE导出和OMG入口，不重编整模。

```bash
(
set -eu
cd "${REPO_DIR:?}"
: "${AI_RUN_DIR:?}" "${CANN_ROOT:?}" "${CPRO_DDK_ROOT:?}" "${CPRO_DDK_ID:?}"
GDN_PROBE_DIR="$AI_RUN_DIR/target/tests/operator-accuracy/gdn"
python3 targets/cpro/tools/probe_target_gdn.py \
  --output "$GDN_PROBE_DIR" --plan-only
python3 targets/cpro/tools/probe_target_gdn.py \
  --output "$GDN_PROBE_DIR" --compile
python3 targets/cpro/tools/probe_target_gdn.py \
  --output "$GDN_PROBE_DIR" --runner-commands
)
```

准备成功返回`PREPARED`；两图均编译成功返回`COMPILED`，编译摘要保存在`gdn/compile.json`。先回传该摘要及新版AIR的`inspection/summary.json`。编译成功后仍需核对实际放置和IO，再执行板端对照；本工具不连接服务器或设备。已准备完成时续跑从`--compile`开始，不重复生成夹具。该固定诊断目录不覆盖已有编译记录；失败日志先用于定位，修复后确认原进程退出，再清理该诊断目录并原位重跑。

`--runner-commands`输出现有`qwen35_tensor_probe`的四条调用参数模板；部署时保持模型和夹具的相对目录，填入实际运行器、部署根、运行时版本与新输出根。每例预热1次、记录10次，两项输出均完整回收。`zero`与`untouched`需分别使用输出目录；它们不等同于已覆盖00／A5／5A填充实验。

回收目录应包含四个case子目录，每个有`result.json`及`0.bin`至`9.bin`，再比较：

```bash
python3 targets/cpro/tools/probe_target_gdn.py \
  --output "$AI_RUN_DIR/target/tests/operator-accuracy/gdn" \
  --compare "${GDN_RETRIEVED_ROOT:?}" \
  --report "$AI_RUN_DIR/target/tests/operator-accuracy/gdn/comparison.json"
```

比较给出每次调用的注意力输出／末态误差和字节一致性，当前为`METRICS_ONLY_NO_CALIBRATED_THRESHOLD`，不会自动宣布精度通过。每次使用原始初态，尚不覆盖状态回灌或部分有效长度；设备回执也不能独自证明部署模型、输入与本地文件一致。这些限制保留到后续真实AIR和板端结果核对。

## CacheUpdate单算子对照

物理cache为`[block, heads*D/16, token_offset, 16] = [64,32,32,16]`，更新张量为`[T,32,16]`；一次写入是`cache[block, :, offset, :] = updates[token]`。两个相邻维度都是32，单看shape无法发现轴互换，因此夹具对各轴做非对称编码。

准备T64／T1两图、四例：T64从位置31开始，覆盖31／32、63／64边界，分别使用连续与非连续物理block；Decode1分别写位置63和64。独立物理参考已与逻辑unpack→Torch append→repack交叉检查，8项主机检查和4项编译入口mock检查通过；真实DDK编译与设备执行仍未运行。

沿用上节环境，执行：

```bash
(
set -eu
cd "${REPO_DIR:?}"
: "${AI_RUN_DIR:?}" "${CANN_ROOT:?}" "${CPRO_DDK_ROOT:?}" "${CPRO_DDK_ID:?}"
CACHE_PROBE_DIR="$AI_RUN_DIR/target/tests/operator-accuracy/cache"
python3 targets/cpro/tools/probe_target_cache.py --output "$CACHE_PROBE_DIR" --plan-only
python3 targets/cpro/tools/probe_target_cache.py --output "$CACHE_PROBE_DIR" --compile
python3 targets/cpro/tools/probe_target_cache.py --output "$CACHE_PROBE_DIR" --runner-commands
)
```

先回传`cache/compile.json`。准备过的目录从`--compile`续跑；失败后的固定目录复用规则与GDN相同。运行器每例预热1次、保留10次完整cache快照。回收四个case目录中的`result.json`、`0.bin`至`9.bin`后执行：

```bash
python3 targets/cpro/tools/probe_target_cache.py \
  --output "$AI_RUN_DIR/target/tests/operator-accuracy/cache" \
  --compare "${CACHE_RETRIEVED_ROOT:?}" \
  --report "$AI_RUN_DIR/target/tests/operator-accuracy/cache/comparison.json"
```

门槛为整个cache的FP16比特完全相同，分别报告写入区和未写区的错误数、坐标及十次稳定性。该测试检查返回tensor，未独立证明输入／输出alias或master中的copy-back；重复相同写入是幂等操作，也不证明移动位置的多轮状态反馈。即使`ALL_BITS_MATCH`，部署身份尚未核对时也不记为完整算子或整模通过。

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
