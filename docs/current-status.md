# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A：完整 target 首次 prefill 和 82 个输出槽位核对已完成。旧编译父进程已退出，4906 为僵尸状态，无需重复停止。下一步把真实 feature 和原始 64 个 token 接入完整六层 draft；feature 的 ±65504 极值与 logits 的 token 0 尚未通过数值验证，本次保留原始字节做单步执行。

先准备小输入，预期 `REAL_FEATURE_DRAFT_STEP_PREPARED`，锚点为第 63 位原始 token `220`。复用现有 2.85 GB OMC，不编译或复制主机模型。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/run_real_feature_draft.py" \
  --inspection "$AI_RUN_DIR/target-prefill64-output-inspection" \
  --draft-bundle "$AI_RUN_DIR/full-draft/residual-fix/bundle" \
  --output "$AI_RUN_DIR/real-feature-draft-plan"
```

准备通过后执行一次，预期 `REAL_FEATURE_DRAFT_EXECUTED_UNVALIDATED`、`remote_exitcode: 0`，取回 15 个候选 token 和 hidden。首次可能上传同一份 draft OMC，后续按 SHA 复用板端缓存；两个窗口的上板操作共用锁，串行执行。通过只代表真实 feature 的 draft 单步完成，后续还需 target decode、80 状态反馈和多轮验证。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_real_feature_draft.py" \
  --prepared "$AI_RUN_DIR/real-feature-draft-plan" \
  --output "$AI_RUN_DIR/real-feature-draft-run" \
  --apply
```

B：55 档特征 AIR 保留备份，本轮无待执行命令。
