# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A：完整 target 的 prefill64、独立 decode1，以及真实 target feature → 完整六层 draft 单步都已执行。最新 decode 退出码 `0`、PID `29654`，输出已取回、锁已释放；四项实验控制仍未验证语义，尚未证明连续状态或精度。

现在检查已保存的 decode 输出，无需重跑模型、重新编译或上传权重。核对工具日志中的全部输出槽位、文件哈希和大小，统计 logits、feature 及 80 个状态的有限值。预期 `DECODE_OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS`，实际文件数 `82`；非有限值和极端数值仍需单独分析，槽位绑定通过不代表精度通过。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/inspect_target_decode_outputs.py" \
  --execution "$AI_RUN_DIR/target-decode1-smoke-trial" \
  --output "$AI_RUN_DIR/target-decode1-output-inspection"
```

输出绑定通过后，检查同一个 decode 模型的状态归属：从 AIR 的更新算子实际端口追溯到 RefData，经编译张量标识和真实运行时名称，最后绑定到本次已取回的输出文件。完整绑定时状态为 `ALL80_SAME_MODEL_STATE_BINDINGS_OBSERVED`、`bound_state_files: 80`；缺字段时返回 `DECODE_STATE_LINEAGE_OBSERVED_WITH_GAPS` 和原因示例。该检查读取 AIR、小 OMC、schema 和输出，不读取外置权重、不调用手机。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/inspect_target_decode_state_lineage.py" \
  --job "$AI_RUN_DIR/target-pytorch-decode1-compile" \
  --runtime-probe "$AI_RUN_DIR/target-decode1-runtime-io-v2" \
  --output-inspection "$AI_RUN_DIR/target-decode1-output-inspection" \
  --output "$AI_RUN_DIR/target-decode1-state-lineage"
```

输出绑定通过后，也可并行检查原工程的 embedding 表。下面使用原工程配置路径；只比对本次输入 token 的原始 INT8 行及 FP32 scale，匹配后提取 logits argmax 对应的下一行。最多读取表中 4104 字节、生成 2056 字节，不重建或重新量化 embedding。预期 `ORIGINAL_DECODE_EMBEDDING_ROWS_BOUND_NEXT_TOKEN_PREPARED`；原表不在当前主机时会列出配置中的缺失路径，输入与原表不符时不生成下一行。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/inspect_original_decode_embeddings.py" \
  --inspection "$AI_RUN_DIR/target-decode1-output-inspection" \
  --config-yaml /home/w00949577/qwen3.5-35B-moe/config/qwen3_5_moe.yaml \
  --output "$AI_RUN_DIR/target-decode1-embedding-inspection"
```

检查结果贴回后，下一步是把 80 个状态分别回填，验证连续 decode；随后接已有逐 token verify 和 DFlash 循环。现有 draft 长度 64 已被当前 64-token 前缀占满，还需明确较短真实 prompt 或更大档位。完整流程、数值精度及加速比仍未通过。

B：55 档特征 AIR 保留备份，本轮无待执行命令。
