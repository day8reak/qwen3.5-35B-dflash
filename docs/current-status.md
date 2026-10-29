# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A：两个探针和首次 prefill 输入计划均已通过，超时后已确认 `LOCK_ABSENT`；更新单次 HDC 锁请求超时后，用新目录执行一次 prefill，预期 `EXECUTED_UNVALIDATED`（已执行并取回输出，数值与状态接续仍待验证）。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/run_target_prefill_smoke.py" \
  --job "$AI_RUN_DIR/target-pytorch-prefill64-tmpdirfix" \
  --runtime-probe "$AI_RUN_DIR/target-prefill64-runtime-io" \
  --input-dir /home/w00949577/run_35B/chunksize_64_data \
  --output "$AI_RUN_DIR/target-prefill64-smoke-v2" \
  --apply
```

A，完成或失败后：汇总当前结果、下一步与失败日志；本命令只读取小报告，不重启任务或清理锁。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/inspect_target_morning_status.py" \
  --decode-probe "$AI_RUN_DIR/target-decode1-runtime-io-v2" \
  --prefill-smoke "$AI_RUN_DIR/target-prefill64-smoke-v2"
```

B：55 档特征 AIR 保留备份，本轮无待执行命令。
