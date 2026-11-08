# 当前阶段：检查Target自定义、融合与量化算子

更新：2026-11-06。按用户提供的patch使用master源码，先检查原40层Target实际使用的算子，覆盖Prefill64／Decode1。暂停离线权重转置及其它Draft性能实验，保留最新packed基线。

## 当前覆盖和结论

源码检查覆盖12类调用：Convstate、GDN、CacheUpdate、Attention、RMSNorm、DynamicQuant、QuantMatMul、MoE TopK、MoE InitRouting、GroupedMatMul、MoE FinalizeRouting，以及权重准备阶段的TransQuantParam。整理出68组带用途标签的逻辑shape；这不是68组已编译或已通过的用例。注册的AdnRmsNorm在当前CPro主路径没有调用，实际Norm使用npu_rms_norm。

本轮完成源码清单与8项主机检查，尚未导出小模型、编译或运行设备。需要继续核对实际AIR和所选DDK的端口、dtype、layout，再生成同输入的独立数学参考与板端对照。

目前优先保留以下问题线索：

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
: "${AI_RUN_DIR:?}" "${CPRO_DDK_ROOT:?}" "${TARGET_SOURCE_ROOT:?}"
python3 targets/cpro/tools/inspect_target_operators.py \
  --source-root "$TARGET_SOURCE_ROOT" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$AI_RUN_DIR/target/tests/operator-accuracy/inspection"
)
```

有现成AIR时，在同一命令中增加`--prefill-air <已有Prefill64 AIR>`和`--decode-air <已有Decode1 AIR>`；默认从所选DDK读取静态protobuf schema，也可用`--torchair-pb2 <实际ge_ir_by_protoc_*_pb2.py>`显式选择TorchAir schema。暂时只有旧AIR时可以用于接口观察，报告始终标记源码版本绑定未验证，不要求先重编整模。

结果为`inspection/inventory.json`。`INVENTORIED`仅表示所请求的清点完成；`PARTIAL`／`SOURCE_REVIEW_REQUIRED`会列出缺失目录、解析错误或未知调用。所有数值状态仍为`NOT_RUN`，收到接口证据后再推进小模型编译和设备比较。重复执行覆盖同一个清单文件。

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
