# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A：原版 target prefill64、独立 decode1、真实 feature → 完整六层 draft 单步均已执行；用户的 rows-v4 与全部 80 状态的 v7 绑定已通过。

本次续跑 `target-decode1-continuation-run` 已完成原工具调用，远端退出码为 0。HDC 随后在取回 `output_10` 时失联，日志和锁释放也未完成，因此字节链尚未验证。用户已确认连接恢复。

当前只恢复同一次执行的输出。原失败目录、输入和模型缓存均保留；恢复过程不重跑模型，也不上传输入或权重。

在 DDK 主机执行：

```bash
git -C "$REPO_DIR" pull --ff-only

python3 "$REPO_DIR/targets/cpro/scripts/recover_target_decode_continuation.py" \
  --execution "$AI_RUN_DIR/target-decode1-continuation-run" \
  --output "$AI_RUN_DIR/target-decode1-continuation-recovery"
```

成功应返回 `DECODE_CONTINUATION_OUTPUTS_RECOVERED_UNVALIDATED`、`feedback_input_count: 80`、`bound_child_output_count: 82`，以及 `lock_state: NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT`。新目录中的 `state-chain.json` 保存父输出 → 实际续跑输入 → 子输出的逐文件字节链。

恢复会核对旧执行记录中的唯一完成标记、89 个实际输入、当时完整的 82 个输出 SHA、同一设备／缓存／attempt、远端退出文件和 PID 已不存在。取回 82 个数据文件和 4 份日志并验证字节链后，才按原 token 释放仍属于本次执行的空锁。两锁均已不存在时只取回；外来锁、孤儿 cache 锁、活跃 PID、哈希变化或连接不确定时停止并保留证据。

HDC 的 `file recv` 现在使用独占临时目标，验证新普通文件后无覆盖发布。仅“进程返回 0 但没有文件”最多尝试 3 次只读取回，保留每次原始响应；这些尝试不会触发模型调用。

这次控制值仍是明确的实验：绝对位置 256 → 257，mask 新开放索引 257（从 0 计数），本地 query 控制保持已执行试验值。字节链通过也不等于控制语义、精度或完整生成通过。

后续继续接 prefill → decode、普通生成基线、任意候选 token 的原始 embedding 查表、draft 上下文容量，以及 verify 的接受／拒绝／状态提交和多轮循环，再验证精度与性能。

B：55 档特征 AIR 保留备份，本轮无待执行命令。
