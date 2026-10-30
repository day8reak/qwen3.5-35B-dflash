# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A：完整 target prefill 与真实 feature → 六层 draft 单步已运行。现在使用提供的 `decode_data` 做一次独立 decode1 试跑；输入数值和状态历史尚未验证，本次结果不代表 prefill → decode 接续或精度通过。

先核对用户文件与实际命名运行时的 89 个输入槽位、shape/dtype/字节数，预期 `DECODE_SMOKE_PLANNED`。会打印原始控制值，不改文件，不读取模型或外部权重，不调用手机。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/run_target_decode_smoke.py" \
  --job "$AI_RUN_DIR/target-pytorch-decode1-compile" \
  --runtime-probe "$AI_RUN_DIR/target-decode1-runtime-io-v2" \
  --input-dir /home/w00949577/run_35B/decode_data \
  --output "$AI_RUN_DIR/target-decode1-smoke-plan"
```

规划通过后执行一次，预期 `EXECUTED_UNVALIDATED`、`remote_exitcode: 0`，实际工具输出保存在 `board/outputs`，日志和文件哈希保留在 `board/result.json`。使用已成功的 V2 调用方式，不重新编译、不复制主机模型；decode 的独立编译权重首次上板可能需上传约 18.8 GB，之后按 SHA 复用缓存。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_target_decode_smoke.py" \
  --job "$AI_RUN_DIR/target-pytorch-decode1-compile" \
  --runtime-probe "$AI_RUN_DIR/target-decode1-runtime-io-v2" \
  --input-dir /home/w00949577/run_35B/decode_data \
  --output "$AI_RUN_DIR/target-decode1-smoke" \
  --apply
```

此后验证普通 prefill/decode 的 80 状态接续，再接入现有候选、verify、接受/拒绝和停止循环；块 verify 的各位置预测与状态提交/回退仍需整模上板验证。旧 tiny 跑通的是 prefix 重算路径。真实输入精度、完整多轮 DFlash 和加速尚未验收。

B：55 档特征 AIR 保留备份，本轮无待执行命令。
