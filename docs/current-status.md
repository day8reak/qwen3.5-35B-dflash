# 当前阶段：Draft 剩余数值误差定位

2026-11-05：`untouched` 回放的首轮额外12928个零已消失，isolated／feedback首例全部14路输出逐位相同。hidden相对L2从65.11%回到1.449%，但原有七处Top1差异及hidden／KV失败仍在；整图尚未通过。该轮支持输出初始化影响首轮快照的判断，底层缓存或buffer机制仍未确定。

profiling已完成：同一OMC的model inference为66.556ms，StableArgMax为0.415ms。保留原CSV，无需重采；热点与测量范围见[结果](results.md)。

## 用户现在执行什么

DDK服务器更新仓库后，用已回收的hidden和原始FP16 LM head离线检查七处Top1差异。只需NumPy，不需要Torch／TorchAir、上板或重新编译。使用现有 `draft/source/prepared`、原精度fixture及本次 `output-init/result`，暂时保留它们。

```bash
source "$REPO_DIR/targets/cpro/scripts/env.sh"
export TOP1_DIR="$AI_RUN_DIR/draft/tests/top1-integration"

python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --audit-head "$TOP1_DIR/output-init/result" \
  --source "$AI_RUN_DIR/draft/source/prepared" \
  --output "$TOP1_DIR/output-init/head-audit.json"
```

回传末尾JSON摘要即可。脚本校验原结果与权重绑定，对每处差异分别使用reference hidden和device hidden计算全词表排名，输出前两名、分差及FP16舍入是否改变第一名；完整数值写入指定JSON，不保存全词表logits。

先看reference hidden是否复现原golden Top1，再看device hidden是否复现设备Top1。若两者均吻合，说明这组hidden差异在CPU head计算下足以重现翻转；若不吻合，再检查head计算或快照可见性。CPU累加顺序不代表NPU内部精度，本检查不改golden、容差或模型。原有CPU源码审计已通过部分用例，不重复整套审计。

普通运行仍保留原输出初始化默认值；本轮实验尚未作为生产修复推广。首次构建见[完整流程](run-end-to-end.md)。
