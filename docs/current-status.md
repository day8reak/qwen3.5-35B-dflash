# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A：原版 target prefill64、独立 decode1、真实 feature → 完整六层 draft 单步均已执行。v6 已完成文件复核：18 份元数据、100 份小型 payload、1 份主 OMC 通过重验；原 AIR 的 80 个状态归属均唯一。连续状态、精度和完整 DFlash 循环仍待验证。

当前有两项可独立执行，先在 DDK 主机拉取更新：

```bash
git -C "$REPO_DIR" pull --ff-only
```

`target-decode1-row-request-v3` 已准备成功。之后的递归 chmod 改变了已记录的 `extract_rows.py` 身份；原清单未记录权限，不能靠猜权限恢复。新参数仅允许完整源码 SHA、路径、inode、大小和 mtime 一致时，对 ctime 变化进行显式重验；默认检查仍严格。现有请求、脚本、响应和旧清单均保留，不需重新提取两行。

使用原请求和已取回的响应，在新目录重试导入：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/transfer_original_decode_rows.py" import \
  --prepared "$AI_RUN_DIR/target-decode1-row-request-v3" \
  --response "$AI_RUN_DIR/target-decode1-row-response-v3/response.json" \
  --revalidate-extractor-metadata \
  --output "$AI_RUN_DIR/target-decode1-original-rows-v4"
```

应返回 `ORIGINAL_DECODE_TRANSFERRED_ROWS_BOUND_NEXT_TOKEN_PREPARED`、`selected_source_token_id: 269`、`selected_next_token_id: 0`。`copied_extractor_revalidation_count` 记录本次脚本复核。只生成原始 INT8 行、独立 FP32 scale 和 INT32 token ID，共 2,056 字节。这不是精度通过；原 256 个 ID 继续作为 context 保留。后续若需调整传输权限，请修改单独的传输副本，保留准备目录原状。

状态映射方面，v6 的阻塞来自源图 NetOutput 的 `copy_…` 边界身份与内部 producer 的 `copy…` 身份被错误合并。编译图实际保留了前者。新版本分别保留两种身份，严格按完整边界 FX 标识唯一匹配，不修改名称或按输出序号分配状态。

编译描述符没有显式 name；运行时输出名将按 [GE 的输出标签实现](https://github.com/Ascend/graphengine/blob/d2ab1c2819c90e61d98b5f88699bcd6b6de4423c/src/ge/graph/load/new_model_manager/davinci_model.cc#L1803)核验。这是固定版本的参考算法，所选 HIAI 内部实现尚未检查；只有全部 82 个实际标签及类型／形状与同一模型的元数据一致时才启用。`src_name/src_index` 在这里仅解释 I/O 标签，计算连接仍由 `OpDef.input` 确定。

在新目录检查状态归属：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/inspect_target_decode_state_lineage.py" \
  --job "$AI_RUN_DIR/target-pytorch-decode1-compile" \
  --runtime-probe "$AI_RUN_DIR/target-decode1-runtime-io-v2" \
  --output-inspection "$AI_RUN_DIR/target-decode1-output-inspection" \
  --source-torchair-pb2 "$AI_RUN_DIR/target-decode1-row-response/ge_ir_by_protoc_3_19_pb2.py" \
  --output "$AI_RUN_DIR/target-decode1-state-lineage-v7"
```

完整绑定应为 `ALL80_SAME_MODEL_STATE_BINDINGS_OBSERVED`、`counts.bound_state_files: 80`；缺少证据仍报告具体 gap。新报告保留身份来源和实际标签核验结果。这只建立同一 decode 模型内的结构映射，不执行模型，不改旧回执，也不证明状态连续性。

这两项通过后，下一步是携带 80 份独立状态的连续 decode，再接逐 token verify／DFlash 循环。控制值语义、精度与 draft 长度 64 的上下文容量仍需对齐。

B：55 档特征 AIR 保留备份，本轮无待执行命令。
