# 当前阶段测试命令

2026-11-03，组合attention完整Draft已执行回收，精度仍为NUMERICAL_FAILED。isolated／feedback的Top1分别82/86、83/86；hidden相对L2约1.34%～2.61%，新增KV约0.44%～0.83%。两组各19例仍无整例通过；未写KV逐位保持，重复调用逐位稳定。原0.01／0.01逐元素门槛及Top1零差异门槛保持。

用户确认当前composed-attention-v1 OMC为CPUCL=3／NPUCL=2，共5个子图，与上一融合版相同。proposal同步Process均值760.223／773.596ms，属于混合输入的精度诊断调用，包含运行时内部调度、同步和搬运，未达到50ms目标。静态分区减少不证明提速，历史GE的Conv70.968ms不能代替当前模型profiling。

复用现有draft/air、draft/tests/pytorch-omc、pytorch-run和pytorch-result。当前不重新导出、编译、生成golden或重复apply完整精度验证，生产未提升。下一步分开检查热点和剩余数值误差。

## 1. 当前OMC的逐算子profiling

在DDK服务器custom_pypto环境，沿用该机REPO_DIR、AI_RUN_DIR和原HDC环境。复用本次精度夹具的case2及模型，允许数值失败状态下采集诊断；已有native无需重建。

```bash
python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --accuracy "$AI_RUN_DIR/draft/tests/pytorch-result" \
  --allow-numerical-failure \
  --output "$AI_RUN_DIR/draft/tests/pytorch-profile-run" --apply

python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --recover "$AI_RUN_DIR/draft/tests/pytorch-profile-run" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-profile-result"
```

若回收为PENDING，继续同一recover，不重复apply。工具使用厂商model_run_tool的一次新实例推理，warmup=0／times=1。CSV用于观察当前热点，不是常驻SDK十次延迟基准，不与760ms直接相减。保留原始op／model CSV、PID和单位；跨stream耗时不能求和当作整模型时间，原精度失败状态保持。

## 2. 服务器CPU复现当前源图数学

复用已回收输入／输出、原golden、同一prepared权重及源AIR，只在CPU执行空context、首次补齐、部分context、容量边界四个代表case。当前RMS在FP32中完成归一化及gamma相乘后转FP16，原参考先把归一化结果转FP16再乘gamma；两种公式仅作CPU对照，实际AIR／OMC不改。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --audit-reference "$AI_RUN_DIR/draft/tests/pytorch-result" \
  --source "$AI_RUN_DIR/draft/source/prepared" \
  --air-export "$AI_RUN_DIR/draft/air" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-host-audit.json"
```

需要已用于生成参考的CPU PyTorch，不需要TorchAir或DDK导出环境，不访问板端。核对保留结果、权重hash、源AIR及控制量，按语义名恢复物理输入顺序。完整比较详情写入固定JSON，stdout显示摘要；原golden／输入／板端结果不改写。读取全部学习权重并执行完整六层及全词表head可能需要数分钟。

- host_vs_reference：CPU源表达与原golden的差异。
- device_vs_host：相同输入下板端与CPU源表达的差异。
- reference_gamma_boundary：仅在CPU对齐RMS舍入边界后检查差异。
- retained_kv_appended_by_layer：分别统计六层K、V新增区误差。

ANALYZED只表示诊断完成，不能提升模型。CPU源表达不复现厂商RmsNorm内部实现，也不证明编译后指令精度；feedback仍以原38次验证为准。若CPU源表达已经失败，先核对源表达与参考边界；若CPU接近参考而板端差异显著，再定位当前OMC的首个偏差，不放宽原门槛。

完整Draft仍为C16／cap192／K15、17输入／14输出，组合attention使用208行逻辑KV。Conv空间排布与小行数档位只完成主机研究，未并入当前图。历史证据见[当前结果](results.md)。
