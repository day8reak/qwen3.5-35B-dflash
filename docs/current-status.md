# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A：prefill64、独立 decode1、真实 feature → 完整六层 draft 单步均已执行；decode 的 82 个文件已绑定，logits、feature、80 个状态均无 NaN/Inf。feature 达到 ±65504、logits argmax 为 0，精度与连续状态仍待验证。

主机检查已修复两项假设：编译报告允许 32 MiB；源 AIR 使用导出回执绑定的 `ge.proto`，编译 OMC 使用 DDK `hiai.proto`，两侧 enum 分别解析。当前 DDK 主机缺两张原始 embedding 表和导出 schema 的 16,543 字节 pb2 文件。下面只传小请求、所需原表行和该 pb2 文件，复用全部已编译模型与已取回输出。

先在 DDK 主机准备请求。预期 `ORIGINAL_DECODE_ROW_TRANSFER_PREPARED`。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/transfer_original_decode_rows.py" prepare \
  --inspection "$AI_RUN_DIR/target-decode1-output-inspection" \
  --config-yaml /home/w00949577/qwen3.5-35B-moe/config/qwen3_5_moe.yaml \
  --output "$AI_RUN_DIR/target-decode1-row-request"
```

把新目录中的 `request.json` 和 `extract_rows.py` 两个文件复制到原权重机器的同一目录。在那个目录执行以下命令；脚本只需标准库 Python，不需要模型环境或仓库。预期 `ORIGINAL_DECODE_ROWS_EXTRACTED`。pb2 复制可以独立进行，用于状态检查。

```bash
python3 -I -S extract_rows.py extract \
  --request request.json \
  --weight-table /home/l00886453/models/Qwen3.5-35B-A3B_quant/qwen3_5_moe_embdedding_weight.bin \
  --scale-table /home/l00886453/models/Qwen3.5-35B-A3B_quant/qwen3_5_moe_embdedding_scale.bin \
  --output ./target-decode1-row-response
cp -n /root/anaconda3/envs/qwen3_5_cpro/lib/python3.11/site-packages/torch_npu/dynamo/torchair/_ge_concrete_graph/ge_ir_by_protoc_3_19_pb2.py \
  ./target-decode1-row-response/ge_ir_by_protoc_3_19_pb2.py
```

请求中的 256 个源 ID 仅作为候选：同时精确匹配实际执行的 INT8 行与独立 FP32 scale，只有唯一 token ID 匹配才提取下一 token 行。重复位置全部保留；无匹配或多个不同 ID 匹配会明确报告。最多读取约 2.1 MB 原表，不复制整表。

把生成的 `target-decode1-row-response` 目录带回 DDK 主机，放在 `$AI_RUN_DIR/` 下，应包含 `response.json` 和 pb2 文件。以下两项可以并行运行。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/inspect_target_decode_state_lineage.py" \
  --job "$AI_RUN_DIR/target-pytorch-decode1-compile" \
  --runtime-probe "$AI_RUN_DIR/target-decode1-runtime-io-v2" \
  --output-inspection "$AI_RUN_DIR/target-decode1-output-inspection" \
  --source-torchair-pb2 "$AI_RUN_DIR/target-decode1-row-response/ge_ir_by_protoc_3_19_pb2.py" \
  --output "$AI_RUN_DIR/target-decode1-state-lineage-v3"
```

完整状态归属为 `ALL80_SAME_MODEL_STATE_BINDINGS_OBSERVED`、`counts.bound_state_files: 80`；缺字段会返回 `DECODE_STATE_LINEAGE_OBSERVED_WITH_GAPS`。pb2 必须与导出回执匹配：文件 SHA `76efc978…`、descriptor SHA `a7cd334b…`，只做 AST 读取，不执行 pb2。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/transfer_original_decode_rows.py" import \
  --prepared "$AI_RUN_DIR/target-decode1-row-request" \
  --response "$AI_RUN_DIR/target-decode1-row-response/response.json" \
  --output "$AI_RUN_DIR/target-decode1-original-rows"
```

预期 `ORIGINAL_DECODE_TRANSFERRED_ROWS_BOUND_NEXT_TOKEN_PREPARED`，生成原始 INT8 embedding、独立 FP32 scale 和 ID，共 2056 字节；导入会重新核对实际 decode、请求、脚本和响应。它证明所选原表行的绑定，尚不证明状态历史、控制语义或精度。

状态归属与原表行通过后，执行 80 状态分别回填的连续 decode，再接已有逐 token verify／DFlash 循环。现有 draft 长度 64 已被当前前缀占满，还需明确较短真实 prompt 或更大档位。

B：55 档特征 AIR 保留备份，本轮无待执行命令。
