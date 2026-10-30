# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A，去重已完成：71 个文件合并，实际可用空间增加约 2.81GB、现余约 8.3G；只读检查旧停止进程 4906 及父进程 4904 是否持有已删除文件，本步不结束进程或删除文件。

```bash
ps -p 4904,4906 -o pid,ppid,stat,etime,args
readlink /proc/4904/cwd /proc/4906/cwd
ls -l /proc/4904/fd /proc/4906/fd 2>/dev/null | grep ' (deleted)$'
grep ' (deleted)$' /proc/4904/maps /proc/4906/maps 2>/dev/null | head -60
```

A，prefill-v3 已创建，查看现有执行结果，不重复启动模型或清理板端锁。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/inspect_target_morning_status.py" \
  --decode-probe "$AI_RUN_DIR/target-decode1-runtime-io-v2" \
  --prefill-smoke "$AI_RUN_DIR/target-prefill64-smoke-v3"
```

B：55 档特征 AIR 保留备份，本轮无待执行命令。
