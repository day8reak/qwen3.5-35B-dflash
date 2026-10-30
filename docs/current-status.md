# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A：decode 试跑尚未执行。提供的目录缺 `conv_state_pos.bin`，且单 token 输入中有 `output_pos=255`、`effective_length=256`、`allQLen=1224`。按已授权的不确定输入试跑，在独立目录准备四项实验控制：conv `[1,0,…]`、output_pos `0`、effective_length `1`、allQLen `1`。conv 编码及实际 decode 语义尚未验证，运行成功只算执行通过。

先准备副本，预期 `TARGET_DECODE_TRIAL_INPUTS_PREPARED`。只复制约 5.3 MB 必需输入，保留位置 `256`、mask、embedding 和状态原字节；原目录和 golden 不改。清单记录四项控制的旧值、新值及假设。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/prepare_target_decode_trial.py" \
  --input-dir /home/w00949577/run_35B/decode_data \
  --experimental-single-token-controls \
  --output "$AI_RUN_DIR/target-decode1-trial-inputs"
```

准备通过后执行一次。先汇总检查 13 个必需文件，再绑定实际运行时的 89 个输入槽位；原件、实验副本和假设在执行前后复核。预期 `EXECUTED_UNVALIDATED`、`remote_exitcode: 0`；实际输出、日志和哈希保存在本次 `board` 目录。不重新编译或复制主机模型，手机首次可能上传 decode 的独立编译权重约 18.8 GB。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_target_decode_smoke.py" \
  --job "$AI_RUN_DIR/target-pytorch-decode1-compile" \
  --runtime-probe "$AI_RUN_DIR/target-decode1-runtime-io-v2" \
  --input-dir "$AI_RUN_DIR/target-decode1-trial-inputs/inputs" \
  --trial-input-receipt "$AI_RUN_DIR/target-decode1-trial-inputs/manifest.json" \
  --output "$AI_RUN_DIR/target-decode1-smoke-trial" \
  --apply
```

本次不证明 64→1 状态接续或数值精度。完整 prefill 和真实 feature→六层 draft 单步已运行；后续验证连续状态，再接已有逐 token verify/DFlash 循环，块 verify 与常驻模型运行用于提速。

B：55 档特征 AIR 保留备份，本轮无待执行命令。
