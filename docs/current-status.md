# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A：已去掉 V2 不支持的 format 覆盖参数；更新后复用已上传并通过 SHA256 核对的权重，用新目录重试一次 prefill，预期 `EXECUTED_UNVALIDATED`（执行与回传完成，数值和状态接续仍待验证）。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/run_target_prefill_smoke.py" \
  --job "$AI_RUN_DIR/target-pytorch-prefill64-tmpdirfix" \
  --runtime-probe "$AI_RUN_DIR/target-prefill64-runtime-io" \
  --input-dir /home/w00949577/run_35B/chunksize_64_data \
  --output "$AI_RUN_DIR/target-prefill64-smoke-v3" \
  --apply
```

A，完成或失败后：汇总当前结果、下一步与失败日志；本命令只读取小报告，不重启任务或清理锁。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/inspect_target_morning_status.py" \
  --decode-probe "$AI_RUN_DIR/target-decode1-runtime-io-v2" \
  --prefill-smoke "$AI_RUN_DIR/target-prefill64-smoke-v3"
```

B：55 档特征 AIR 保留备份，本轮无待执行命令。
