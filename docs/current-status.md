# 当前阶段测试命令

C16／cap192／K15增量Draft已接通完整流程，独立验证也已完成。hidden相对L2约1.09%～1.59%，连续反馈KV约0.44%～0.54%；有效Top1为isolated 84/86、feedback 86/86。按当前工程决定，把该版本作为阶段性性能诊断基线，继续profiling。原逐元素比较仍记录为`NUMERICAL_FAILED`，不据此宣称语言质量或最终精度验收通过。

## 1. 采集当前Draft的算子CSV

复用`draft/omc`、现有精度输入及板端缓存，只需更新Python脚本，无需重新编译AIR、OMC或native运行器。沿用已设置的`REPO_DIR`、`AI_RUN_DIR`、HDC环境变量。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --accuracy "$AI_RUN_DIR/draft/tests/accuracy-result" \
  --allow-numerical-failure \
  --output "$AI_RUN_DIR/draft/tests/profile" --apply
```

该开关只允许对已完成的数值失败样本做性能诊断，不修改比较规则，也不放行执行失败或未回收的任务。使用部分上下文proposal样本的17路输入／14路输出，参考KV输入保持原样；只执行一次厂商工具推理，不是常驻稳态基准。

```bash
python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --recover "$AI_RUN_DIR/draft/tests/profile" \
  --output "$AI_RUN_DIR/draft/tests/profile-result"
```

若返回`DRAFT_OPERATOR_PROFILE_PENDING`，重复同一回收命令。完成后查看`draft/tests/profile-result/prof_data/`中的`*_op.csv`、`*_model.csv`。CSV原样保留，不强制解析；需核对模型归属，因为厂商导出可能包含历史记录。旧S64的282.430ms投影记录不代表本次增量结果。

## 2. 可选：离线查看误差分布

直接读取已回收的张量，分别统计hidden、整块KV、已提交前缀和本次新增行的相对L2，列出Top1不同的位置，并检查feedback两次零新增行重复的输出是否逐字节一致。无需访问板端，也不阻塞上面的profiling。

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --analyze "$AI_RUN_DIR/draft/tests/accuracy-result"
```

数值`0.01`表示1%；空区域不参与统计，参考范数为零时L2显示`null`。原`result.json`保持不变。整块缓存的正确历史行可能稀释新增区误差；isolated会重新输入参考KV，其“repeat”与前一步并非相同KV输入，不用于判断板端重复稳定性。

保留`draft/tests/accuracy`与`accuracy-result`供上述两项复用。同一功能继续使用固定目录。取得新CSV后按实测热点修改；改动后重新比较相同精度样本，并执行一次完整流程接通测试。真实特征覆盖、更多样本及Target修复后的接受率用于后续验收。
