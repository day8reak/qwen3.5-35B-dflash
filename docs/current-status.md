# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A，存储检查：当前 run 约 103GB、full-draft 约 30GB，整盘仅剩 5.7GB；先查看用户目录占用、draft 子目录、活动进程及只读去重计划，预期最后输出 `PLANNED` 和 `logical_duplicate_bytes`，本步不删文件。

```bash
du -xhd1 /home/w00949577 2>/dev/null | sort -h
du -xhd2 "$AI_RUN_DIR/full-draft" | sort -h
ps -eo pid,ppid,stat,etime,args | grep -E '[r]un_target_prefill_smoke|[c]ompile_full_target|[o]mg|[m]odel_run_tool|[r]un_full_draft|[r]un_draft_'
python3 "$REPO_DIR/targets/cpro/scripts/compact_draft_storage.py" \
  --job "$AI_RUN_DIR/full-draft"
```

A，仅去重计划显示 `planned_changes > 0` 且没有 draft 编译运行时：合并相同的只读编译输入，保留文件路径、内容、成功 bundle 与全部报告；预期 `COMPACTED`，实际释放量看 `observed_free_byte_change` 和 `df`。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/compact_draft_storage.py" \
  --job "$AI_RUN_DIR/full-draft" --apply
df -h "$AI_RUN_DIR"
```

A，prefill-v3 已创建，查看现有执行结果，不重复启动模型或清理板端锁。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/inspect_target_morning_status.py" \
  --decode-probe "$AI_RUN_DIR/target-decode1-runtime-io-v2" \
  --prefill-smoke "$AI_RUN_DIR/target-prefill64-smoke-v3"
```

B：55 档特征 AIR 保留备份，本轮无待执行命令。
