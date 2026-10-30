# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A：原版 40 层 target prefill64、独立 decode1、真实 feature → 完整六层 draft 单步均已执行。用户返回的 v7 已完成全部 80 个源状态归属、编译身份、运行时名称和实际文件绑定；rows-v4 已绑定原始 query token 269，并准备好下一 token 0 的原始 INT8 行及独立 FP32 scale。

当前进入同一个 decode 模型的单次状态续接。该实验从已有绝对位置 256 推进到 257：逐份使用上一次的 80 个状态输出，接入 token 0，将 mask 的索引 257（从 0 计数）从负无穷改为 0，其余 mask 字节保留。output_pos=0、allQLen=1、effective_length=1、conv_state_pos=[1,0×127] 保留已执行的试验值。控制语义与原始历史仍未验证；这一步通过后才继续接 prefill → decode 和普通生成基线。

在 DDK 主机拉取更新，然后准备新输入：

```bash
git -C "$REPO_DIR" pull --ff-only

python3 "$REPO_DIR/targets/cpro/scripts/prepare_target_decode_continuation.py" \
  --state-lineage "$AI_RUN_DIR/target-decode1-state-lineage-v7" \
  --original-rows "$AI_RUN_DIR/target-decode1-original-rows-v4" \
  --assumption single-query-absolute-position-plus-one \
  --output "$AI_RUN_DIR/target-decode1-continuation-prepared"
```

准备成功应返回 `SAME_MODEL_DECODE_CONTINUATION_PREPARED`、`independent_state_file_count: 80`、`continuation_token_id: 0`、`next_absolute_position: 257`。

准备过程重新核对已有状态归属和原始行证据，并生成 89 个独立输入文件，其中 80 个状态逐份复制，不改变旧文件的内容或链接数。准备阶段不调用设备、不读取外置权重、不编译模型。

准备通过后执行一次续跑：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_target_decode_continuation.py" \
  --prepared "$AI_RUN_DIR/target-decode1-continuation-prepared" \
  --output "$AI_RUN_DIR/target-decode1-continuation-run" \
  --apply
```

成功应返回 `DECODE_CONTINUATION_EXECUTED_UNVALIDATED`、`feedback_input_count: 80`、`bound_child_output_count: 82`。`state-chain.json` 保存父状态输出 → 实际续跑输入 → 子状态输出的逐文件字节链，以及 logits／feature 统计。输出仍按完整工具日志绑定，不按文件名推测。

执行沿用已成功的 HIAI V2 工具及全局锁。已有模型缓存逐文件核对板端 SHA，匹配后复用；缺失文件仍通过原编译清单的严格本地检查再上传。超时保留锁、PID 和日志，不自动重试。

当前进展：

- 已完成：真实 target／draft 单步执行、80 状态结构映射、原始下一 token 输入绑定。
- 正在验证：同一 decode 模型的实验性状态续接。
- 后续：prefill → decode、普通生成基线、任意候选 token 的原始 embedding 查表、draft 上下文容量、逐 token verify 和多轮接受／拒绝／提交，再做批量 verify、精度与性能验证。

B：55 档特征 AIR 保留备份，本轮无待执行命令。
