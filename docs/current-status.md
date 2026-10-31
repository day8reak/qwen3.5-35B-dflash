# 当前阶段测试命令

C16／cap192／K15增量Draft已接通完整板端循环：1次补齐、7次proposal，Draft同步API均值608.155ms。现在验证Draft独立精度，再采集同一OMC的算子CSV；Target精度与verify暂不推进。已有`draft/air`、`draft/omc`、`draft/source/prepared`和`runtime`直接复用，不重编模型。

## 1. 更新共享运行器

沿用环境页设置的`REPO_DIR`、`AI_RUN_DIR`、DDK／OHOS及HDC环境变量。新增独立精度程序，与普通推理共用HIAI执行器；普通`dump=false`循环不增加回读。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$AI_RUN_DIR/runtime/native" --replace
```

## 2. Draft独立精度

服务器用原始准备权重生成确定性非零特征及CPU PyTorch参考，随后只在板端加载Draft。两组分别使用参考KV输入、连续复用板端KV，覆盖空上下文、补齐、部分块、候选拒绝、重复调用和容量边界。参考计算重算完整前缀，不调用GE图求值器。合成特征检验计算一致性，不代表真实语言质量。

比较有效Top1完全一致、hidden及12路KV满足`abs(a-b) <= 0.01 + 0.01*abs(b)`；未写区与提交回读逐字节一致。容差在执行前确定，失败时保留首个偏差。当前诊断支持容量不超过512；物理候选档位从OMC读取，较小`proposal_count`不等于编译了较小档位。

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --config "$AI_RUN_DIR/runtime/runtime.json" \
  --compiled "$AI_RUN_DIR/draft/omc" \
  --source "$AI_RUN_DIR/draft/source/prepared" \
  --air-export "$AI_RUN_DIR/draft/air" \
  --atol 0.01 --rtol 0.01 \
  --apply --output "$AI_RUN_DIR/draft/tests/accuracy"
```

CPU参考准备可能需要数分钟；显示`DRAFT_ACCURACY_STARTED`后回收。若仍为`DRAFT_ACCURACY_PENDING`，重复同一回收命令，不重新启动。

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --recover "$AI_RUN_DIR/draft/tests/accuracy" \
  --output "$AI_RUN_DIR/draft/tests/accuracy-result"
```

通过条件是`numerical_validation=PASSED`。详细误差在`draft/tests/accuracy-result/result.json`；失败先看`first_failure`。API计时分补齐／proposal，仅作诊断，不含输出文件读写，也不作为正式性能基准。诊断会导出KV并做内容校验，空间预检不足时不会启动板端任务。模型使用上次完整循环已部署的同一缓存，不另传2.85GB模型。

## 3. 同一增量模型Profiling

独立精度通过后执行。自动选择已验证的部分上下文proposal输入，按17输入／14输出调用厂商工具；实际`--help`在本次板端检查，不依赖已删除的旧inspection目录。

```bash
python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --accuracy "$AI_RUN_DIR/draft/tests/accuracy-result" \
  --output "$AI_RUN_DIR/draft/tests/profile" --apply

python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --recover "$AI_RUN_DIR/draft/tests/profile" \
  --output "$AI_RUN_DIR/draft/tests/profile-result"
```

看到`DRAFT_PROFILE_CAPTURED_REQUIRES_ATTRIBUTION_REVIEW`后，直接查看`draft/tests/profile-result/prof_data/`中的`*_op.csv`、`*_model.csv`，不强制解析。该工具可能同时导出服务历史记录，需核对模型与调用。旧S64的282.430ms投影记录不能替代本次增量结果。

同一功能始终复用上述固定目录；完成回收后脚本可替换前一次测试。完成profiling前保留本次accuracy输入。取得新CSV后再选择热点改动，并重新跑独立精度与完整流程接通测试。
