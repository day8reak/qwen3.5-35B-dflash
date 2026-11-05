# 当前阶段：复算 hidden 误差来源

2026-11-05：第0层55个单输出检查点 × case 0／2／17全部完成，165组、1650次比较均重复稳定；精度仍失败。三个上下文首次观察到超差均在Q投影 `MatMul:0`，相对L2约0.145%；输入RMS通过，Q-RMS后也通过。20组算术结果／FP16 Cast配对在三个上下文的十次比较均一致。本批未出现“参考非零、输出全零”。完整Draft精度仍未通过，性能优化暂缓。详情见[结果页](results.md)。

下一步只在CPU读取现有快照：用设备input-RMS输出复算Q投影，区分可由上游误差解释的部分；对四条Sin/Cos分支分别模拟相位舍入、输出舍入和两者同时舍入。独立前缀中的RMS不等于Q前缀内部实测输入，候选匹配不能证明内部累加精度；原golden和门槛不变。

## DDK服务器执行

更新代码后运行以下离线分析。使用已完成的 `hidden-layer0/result`、原 `top1-integration/export` 及其权重，无需编译或设备调用，也不需要Torch/TorchAir。

```bash
git -C "$REPO_DIR" pull --ff-only
export TOP1_DIR="$AI_RUN_DIR/draft/tests/top1-integration"
export LAYER_DIR="$TOP1_DIR/hidden-layer0"

python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --analyze "$LAYER_DIR/result" --air-export "$TOP1_DIR/export" \
  > "$LAYER_DIR/math-audit.json"

python3 - "$LAYER_DIR/math-audit.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
if 'math_audit' not in d:
    print(json.dumps(d, ensure_ascii=False))
    raise SystemExit(1)
def compact(r):
    return {k: r[k] for k in ('passed', 'bitwise_equal', 'max_abs_error', 'relative_l2_error')} | {
        'failed_elements': r['tolerance']['failed_elements'],
        'elements': r['tolerance']['elements']}
contexts = []
for c in d['math_audit']['contexts']:
    q = c['query_projection']
    contexts.append({
        'source_case': c['source_case'],
        'query': {k: compact(v) for k, v in q.items() if isinstance(v, dict)},
        'trigonometry': [{
            'source_edge': t['source_edge'], 'phase_edge': t['phase_edge'],
            'phase_max_abs': t['phase_max_abs'],
            'phase_fp16_rounding_max_abs': t['phase_fp16_rounding_max_abs'],
            'device_vs_candidates': {k: compact(v) for k, v in t['device_vs_candidates'].items()}
        } for t in c['trigonometry']]})
print(json.dumps({
    'numerical_validation': d['numerical_validation'],
    'counts_by_context': d['summary']['counts_by_context'],
    'math_audit_status': d['math_audit']['status'], 'contexts': contexts
}, ensure_ascii=False))
PY
```

贴最后的摘要即可。完整JSON另保留每点超差元素数量、最严重坐标、参考重放一致性和编译归属。AIR、权重及保留输入按原记录校验；只在十次输出逐位稳定时使用第0次快照复算。该功能已通过本地合成检查，真实快照的CPU分析结果待回传。
