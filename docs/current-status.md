# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A，窗口 2：prefill64 探针已通过且锁已释放；更新并修复 decode1 的排队响应后，用新目录重试，预期输出 `IO_METADATA_OBSERVED`、89/82 路计数、`manager_init=NOT_RUN`，同时记录 SDK 实际返回的张量名称。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/inspect_target_runtime_io.py" \
  --job "$AI_RUN_DIR/target-pytorch-decode1-compile" \
  --tool-inspection "$AI_RUN_DIR/target-board-tool-inspection-writable" \
  --lock-wait-seconds 1800 \
  --output "$AI_RUN_DIR/target-decode1-runtime-io-v2"
```

A，窗口 1：复用已通过的 prefill64 探针，核对首次 prefill 的实际输入与槽位，可与 decode1 重试并行；通过时输出 `PREFILL_SMOKE_PLANNED`，本步不上传大权重或执行模型。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_target_prefill_smoke.py" \
  --job "$AI_RUN_DIR/target-pytorch-prefill64-tmpdirfix" \
  --runtime-probe "$AI_RUN_DIR/target-prefill64-runtime-io" \
  --input-dir /home/w00949577/run_35B/chunksize_64_data \
  --output "$AI_RUN_DIR/target-prefill64-smoke-plan"
```

A，仅计划通过后：上传缺失的编译权重并执行一次 prefill；`EXECUTED_UNVALIDATED` 表示已执行并取回输出，准确性和状态接续尚待验证。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_target_prefill_smoke.py" \
  --job "$AI_RUN_DIR/target-pytorch-prefill64-tmpdirfix" \
  --runtime-probe "$AI_RUN_DIR/target-prefill64-runtime-io" \
  --input-dir /home/w00949577/run_35B/chunksize_64_data \
  --output "$AI_RUN_DIR/target-prefill64-smoke" \
  --apply
```

A，任一步完成或失败后：汇总当前结果、下一步与失败日志；本命令只读取小报告，不重启任务或清理锁。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/inspect_target_morning_status.py" \
  --decode-probe "$AI_RUN_DIR/target-decode1-runtime-io-v2"
```

B：55 档特征 AIR 保留备份，本轮无待执行命令。
