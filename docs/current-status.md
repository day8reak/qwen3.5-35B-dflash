# 当前阶段：检查Target自定义、融合与量化算子

更新：2026-11-09。按用户提供的patch使用master源码，先检查原40层Target实际使用的算子，覆盖Prefill64／Decode1。暂停离线权重转置及其它Draft性能实验，保留最新packed基线。

## 当前覆盖和结论

源码检查覆盖12类调用：Convstate、GDN、CacheUpdate、Attention、RMSNorm、DynamicQuant、QuantMatMul、MoE TopK、MoE InitRouting、GroupedMatMul、MoE FinalizeRouting，以及权重准备阶段的TransQuantParam。整理出68组带用途标签的逻辑shape；这不是68组已编译或已通过的用例。注册的AdnRmsNorm在当前CPro主路径没有调用，实际Norm使用npu_rms_norm。

接口检查工具的20项主机检查通过，尚未编译或运行设备。需要继续核对实际AIR和所选DDK的端口、dtype、layout，再生成同输入的独立数学参考与板端对照。

服务器首次回传的`PARTIAL`由两个预设候选目录缺失触发（`ddk/ascendcpp`、`tools/platform/kirinx90`）；扫描未达到上限且没有读取错误。这不表示DDK损坏或算子精度失败。`airs={}`表示本次没有提供AIR，尚未观察整模实际使用的GE类型。

后续AIR回传的`hiai.proto.AttrDef.list_list_float` wire-type错误暴露了检查器的schema选择问题：旧入口默认使用DDK的HIAI schema。现已优先复用本工程AIR工具所用的离线TorchAir `ge.proto` schema（`specs/torchair-ge-ir.pb`），严格解码失败时才尝试所选DDK的HIAI schema；显式指定`--torchair-pb2`时只使用该指定。报告保留实际schema及失败原因，不将解析错误归因于模型损坏。

目前优先保留以下问题线索：

- GDN的源码converter包含7个输入；回传DDK的两个JSON注册表、Python入口和C++入口均只有6个，缺少`effective_length`。源码生成FP32的`g`，Meta将末态声明为FP32；两个DDK注册表将`g`、`initial_state`和两个输出均声明为FP16。仍需查看实际AIR的Cast及编译选择，不能直接删掉有效长度或断言这些声明差异已经造成数值故障。
- DDK文本中出现`QuantBatchMatmulV4444`、`GroupedMatmulV555`、`MoeInitRoutingV3`和`MoeFinalizeRoutingV2`。新版工具按实际名称记录这些候选；未证明原Python调用最终选择了哪个变体。`ascend310b`目录名和头文件中的310P注释也不能替代X90编译选择证据。
- 回传的V4444 C++入口实例化`QBMInt8Compute<SCALE_FP32>`，没有显示A8W4分支。目录中存在A8W4头文件不能证明该入口使用了它；需对照实际量化MatMul节点、权重载体和编译实现，保持原W4配置。
- 历史GDN结果受输出缓冲区00／A5／5A填充值影响，零填充下state相对L2仍约0.146／0.155。来自旧版本用户回传，根因未定位，不能视为当前master内核已确认失败或已修复。
- CPro量化MoE在`pertoken_scale.dim() > 1`时执行`x_quant = x.view(...)`，覆盖动态量化结果。需采集实际scale维度和GMM输入dtype，确认该分支是否触发；当前保留master代码。
- Convstate控制数组和输出rank、GDN末态dtype需与当前DDK核对。旧CacheUpdate／Attention只有shape检查，精度未通过。
- master的`quant_bits`默认8，使用原W4数据导出时需显式指定4。旧AIR不能作为新源码已经重新导出的证据。

## 用户现在执行什么

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
