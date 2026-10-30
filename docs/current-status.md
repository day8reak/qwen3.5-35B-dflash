# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A：原版 target prefill64、独立 decode1、真实 feature → 完整六层 draft 单步均已执行。decode 的 82 个输出文件已绑定；精度和连续状态尚未验证。

原权重机器精确定位到 query ID **269**，整行 INT8 和独立 FP32 scale 都匹配，累计读取 995,392 字节。原 256 个 ID 文件仅保留为 context，不替换、不截取。下一行来自已执行 decode logits 的 argmax 0，这仍不是精度通过。

最新 OMC 检查确认 SHA256、inode、大小、mtime 均不变，仅 ctime 变化。新检查仅对此情况重哈希小主 OMC，并保留当前身份与复核记录；不修改编译报告和旧执行回执，不重编译。图连接检查也已修正：采用实际 `OpDef.input`，保留空可选槽位、排除控制边，将 `src_name/src_index` 单独作为元数据记录。

DDK 主机先拉取更新，再用新目录检查状态归属：

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/inspect_target_decode_state_lineage.py" \
  --job "$AI_RUN_DIR/target-pytorch-decode1-compile" \
  --runtime-probe "$AI_RUN_DIR/target-decode1-runtime-io-v2" \
  --output-inspection "$AI_RUN_DIR/target-decode1-output-inspection" \
  --source-torchair-pb2 "$AI_RUN_DIR/target-decode1-row-response/ge_ir_by_protoc_3_19_pb2.py" \
  --output "$AI_RUN_DIR/target-decode1-state-lineage-v4"
```

完整绑定应为 `ALL80_SAME_MODEL_STATE_BINDINGS_OBSERVED`、`counts.bound_state_files: 80`；缺证据会输出具体 gap。这只建立同一 decode 模型内的结构映射，尚未执行状态反馈。

原始行提取可独立推进。在 DDK 主机准备新请求：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/transfer_original_decode_rows.py" prepare \
  --inspection "$AI_RUN_DIR/target-decode1-output-inspection" \
  --config-yaml /home/w00949577/qwen3.5-35B-moe/config/qwen3_5_moe.yaml \
  --query-token-id 269 \
  --output "$AI_RUN_DIR/target-decode1-row-request-v2"
```

应返回 `ORIGINAL_DECODE_ROW_TRANSFER_PREPARED`，`schema_version: 2`、`query_token_id: 269`、`effective_candidate_count: 1`。若失败，保留新目录并贴回错误，不修改旧请求。

把 DDK 主机的这两个新文件：

```text
/home/w00949577/qwen3.5-35B-dflash-run/target-decode1-row-request-v2/request.json
/home/w00949577/qwen3.5-35B-dflash-run/target-decode1-row-request-v2/extract_rows.py
```

复制到原权重机器的这两个位置：

```text
/home/w00949577/qwen35-decode1-row-request-v2/request.json
/home/w00949577/qwen35-decode1-row-request-v2/extract_rows.py
```

在原权重机器执行，仅读原表的 269 和 0 两行，共 4,104 字节：

```bash
cd /home/w00949577/qwen35-decode1-row-request-v2
python3 -I -S extract_rows.py extract \
  --request request.json \
  --weight-table /home/l00886453/models/Qwen3.5-35B-A3B_quant/qwen3_5_moe_embdedding_weight.bin \
  --scale-table /home/l00886453/models/Qwen3.5-35B-A3B_quant/qwen3_5_moe_embdedding_scale.bin \
  --output ./target-decode1-row-response-v2
```

应返回 `ORIGINAL_DECODE_ROWS_EXTRACTED`、`matching_token_ids: [269]`。不再使用旧的 NO_MATCH 响应，也不把定位用的 `diagnostic.json` 当作提取响应。

把原权重机器的：

```text
/home/w00949577/qwen35-decode1-row-request-v2/target-decode1-row-response-v2/response.json
```

复制回 DDK 主机的：

```text
/home/w00949577/qwen3.5-35B-dflash-run/target-decode1-row-response-v2/response.json
```

在 DDK 主机导入：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/transfer_original_decode_rows.py" import \
  --prepared "$AI_RUN_DIR/target-decode1-row-request-v2" \
  --response "$AI_RUN_DIR/target-decode1-row-response-v2/response.json" \
  --output "$AI_RUN_DIR/target-decode1-original-rows-v2"
```

应返回 `ORIGINAL_DECODE_TRANSFERRED_ROWS_BOUND_NEXT_TOKEN_PREPARED`、`selected_source_token_id: 269`、`selected_next_token_id: 0`。只生成原始 INT8 行、独立 FP32 scale 和 INT32 token ID，共 2,056 字节。

状态归属与原始行通过后，下一步是携带 80 份独立状态的连续 decode，再接逐 token verify／DFlash 循环。控制值语义、精度与 draft 长度 64 的上下文容量仍需对齐。

B：55 档特征 AIR 保留备份，本轮无待执行命令。
