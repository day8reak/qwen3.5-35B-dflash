# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A：prefill64、独立 decode1、真实 feature → 完整六层 draft 单步均已执行；decode 的 82 个文件已绑定且均无 NaN/Inf。精度和连续状态仍待验证。

原权重机器的提取返回 `ORIGINAL_DECODE_ROWS_NO_MATCH`：256 个 ID 去重后的 7 个候选中，INT8 行和 FP32 scale 均不对应实际 query。这只说明候选对应关系未建立，不能据此判定表错误。现在复用原请求做精确定位，不重复准备请求或导入旧的 NO_MATCH 响应。

先在 DDK 主机拉取更新，并复制独立定位脚本：

```bash
git -C "$REPO_DIR" pull --ff-only
cp -n "$REPO_DIR/model/qwen35_moe_dflash/original_decode_row_transfer.py" \
  "$AI_RUN_DIR/target-decode1-locate.py"
```

把 DDK 主机这个文件：

```text
/home/w00949577/qwen3.5-35B-dflash-run/target-decode1-locate.py
```

复制到原权重机器的这个位置：

```text
/home/w00949577/qwen35-decode1-row-request/target-decode1-locate.py
```

在原权重机器执行，继续使用已经存在的 `request.json`：

```bash
cd /home/w00949577/qwen35-decode1-row-request
python3 -I -S target-decode1-locate.py locate \
  --request request.json \
  --weight-table /home/l00886453/models/Qwen3.5-35B-A3B_quant/qwen3_5_moe_embdedding_weight.bin \
  --scale-table /home/l00886453/models/Qwen3.5-35B-A3B_quant/qwen3_5_moe_embdedding_scale.bin \
  --output ./target-decode1-row-location
```

定位会完整读取约 1 MB 的 scale 表，对所有 scale 精确匹配项比较 INT8 字节前缀，再核对剩余字节。用户态请求的数据总量上限为 2,103,300 字节；预算不足时不截取部分候选来宣称唯一性。结果保存在 `/home/w00949577/qwen35-decode1-row-request/target-decode1-row-location/diagnostic.json`。

- `ORIGINAL_DECODE_ROW_LOCATION_UNIQUE`：原表中找到唯一精确行，`matching_pair_token_ids` 给出实际 query 的候选 ID。
- `ORIGINAL_DECODE_ROW_LOCATION_AMBIGUOUS`：多个 ID 的整行与 scale 都相同，需要原生成流程进一步确认 ID。
- `ORIGINAL_DECODE_ROW_LOCATION_NO_MATCH`：完整检查后，实际行与 scale 的组合不在所声明表中；应回查原输入生成流程。
- `ORIGINAL_DECODE_ROW_LOCATION_BUDGET_CAP`：未完成所有整行比较，保留候选数量及预算信息。

这一步只输出定位报告，不生成下一 token 文件。唯一行定位通过后，再准备带有明确来源的下一 token 输入。

pb2 复制和状态归属检查可并行推进，不依赖这次行匹配。在原权重机器执行：

```bash
cp -n /root/anaconda3/envs/qwen3_5_cpro/lib/python3.11/site-packages/torch_npu/dynamo/torchair/_ge_concrete_graph/ge_ir_by_protoc_3_19_pb2.py \
  /home/w00949577/qwen35-decode1-row-request/target-decode1-row-response/ge_ir_by_protoc_3_19_pb2.py
```

把这个 pb2 文件带回 DDK 主机的 `/home/w00949577/qwen3.5-35B-dflash-run/target-decode1-row-response/`，然后执行尚未运行的状态检查：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/inspect_target_decode_state_lineage.py" \
  --job "$AI_RUN_DIR/target-pytorch-decode1-compile" \
  --runtime-probe "$AI_RUN_DIR/target-decode1-runtime-io-v2" \
  --output-inspection "$AI_RUN_DIR/target-decode1-output-inspection" \
  --source-torchair-pb2 "$AI_RUN_DIR/target-decode1-row-response/ge_ir_by_protoc_3_19_pb2.py" \
  --output "$AI_RUN_DIR/target-decode1-state-lineage-v3"
```

预期完整绑定为 `ALL80_SAME_MODEL_STATE_BINDINGS_OBSERVED`、`counts.bound_state_files: 80`，缺证据则报告具体原因。若已运行过该目录，保留并贴回已有结果。

状态归属与原始行通过后，下一步是连续 decode，再接逐 token verify／DFlash 循环；现有 draft 长度 64 的上下文容量仍需对齐。

B：55 档特征 AIR 保留备份，本轮无待执行命令。
