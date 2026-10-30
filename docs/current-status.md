# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A，窗口 1：完整 target 首次 prefill 已执行成功；更新后只读核对已取回文件、日志中的槽位映射、feature/logits 数值统计、draft 固定容量和原始 token 文件，预期 `OUTPUT_FILES_BOUND_TO_LOGGED_SLOTS`；不重新执行模型或复制权重。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/inspect_target_prefill_outputs.py" \
  --execution "$AI_RUN_DIR/target-prefill64-smoke-v3" \
  --draft-bundle "$AI_RUN_DIR/full-draft/residual-fix/bundle" \
  --input-dir /home/w00949577/run_35B/chunksize_64_data \
  --output "$AI_RUN_DIR/target-prefill64-output-inspection"
```

A，窗口 2：收尾工作目录已删除且已停止的旧编译进程 4906、其绑定父进程和停止的子进程；仅身份完全匹配时发送 TERM 后 CONT，预期 `STOP_REQUESTED`；该状态不表示已退出或已释放空间。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/stop_deleted_target_compile.py" \
  --pid 4906 \
  --job "$AI_RUN_DIR/full-target-compile-0491abfcd79b" \
  --apply --legacy-stopped-tree \
  --output "$AI_RUN_DIR/deleted-target-compile-stop"
ps -p 4904,4906 -o pid,ppid,stat,etime,args
df -h "$AI_RUN_DIR"
```

B：55 档特征 AIR 保留备份，本轮无待执行命令。
