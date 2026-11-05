# 当前阶段：定位 RoPE 候选的新增 KV 回退

2026-11-05：RoPE 系数查表候选已完成原 isolated／feedback 各19例回放，完整精度仍未通过。hidden 最大相对L2分别从2.576%／2.612%降至1.498%／1.772%，Top1差异由7处变为6处。但 isolated 新增KV最大相对L2从0.828%升至38.215%；feedback则降至0.608%。这些是诊断指标，验收仍使用原逐元素容差和Top1／缓存保持门槛。性能优化暂缓，详细证据见[结果页](results.md)。

两种模式共用一个已加载模型和固定测试计划：isolated 每个case都加载该case的原参考KV输入，避免前一次设备误差传入；feedback 从首例初始化后复用设备KV，只提交有效新增区间。后续feedback的hidden因此不属于与isolated完全相同输入的比较。源码中新上下文KV由本次target_features和位置生成，不依赖历史KV；这次新增KV回退不能直接用缓存历史不同解释。

## DDK 服务器离线分析

保留原 `output-init/result` 和本次 `rope-fp32/result`；下面只读回收张量，不编译、不生成参考、不访问设备，也不再次执行模型。

```bash
git -C "$REPO_DIR" pull --ff-only
export TOP1_DIR="$AI_RUN_DIR/draft/tests/top1-integration"
export ROPE_DIR="$TOP1_DIR/rope-fp32"

python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --diagnose-kv "$ROPE_DIR/result" > "$ROPE_DIR/kv-diagnosis.json"

python3 - "$ROPE_DIR/kv-diagnosis.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
keys = ('status', 'error', 'numerical_validation', 'candidate_model',
        'baseline_model', 'input_verification', 'summary')
summary = {k: d[k] for k in keys if k in d}
initial = d.get('initial_call_comparison', {})
summary['initial_call_comparison'] = {
    k: initial[k] for k in ('status', 'all_kv_inputs_zero', 'outputs') if k in initial}
print(json.dumps(summary, ensure_ascii=False))
PY
```

贴最后摘要即可。工具重新核对完整原比较、原输入语义名称和字节哈希、golden、baseline绑定及容差，然后逐case／层／K或V检查新增区间，报告最差head和cache行、异常零值、与feedback及旧模型的差异。当前候选OMC的SHA也会从回收manifest输出；之前回传的 `compiler_sha256` 是图改写身份，不能当作OMC哈希。

本地10项检查通过，覆盖完整合成回放、输入槽位重排、异常定位、原文件不写入、绑定／输入／golden／容差变化拒绝及非有限值处理。它们验证离线工具，不是新的设备精度验收。待真实结果定位后再选择修复；本轮不重跑55点探针。
