# 当前阶段：确认176行重复177行的发生位置

2026-11-05：离线核对了新旧回放的全部输入、golden及312个新增KV区间。RoPE候选OMC SHA为 `b43ae9cfcd50999277829a21adc24099170d4a7c4856b70729385b59d362a828`。最大回退位于isolated case16（start162、valid16）的第0层K；最差head行都位于cache位置176，即本次features的局部行14。该区间无零值或非有限值，两模式相差1024个元素，数量恰为8×128。多个层的V也有大幅回退，不能用仅影响K/Q的RoPE算术解释。完整精度仍FAILED，性能工作暂缓。[完整结果](results.md)。

完整行分析已回传：仅case16的cache176异常，12路K/V各8个head、1024个元素全部与feedback不同，其余15行两模式逐位一致。六层V的176行与同次177行逐位相同，未匹配到接近的保留历史行。其余15行相对L2约0.445%～0.556%，仍未通过逐元素门槛；整行重复与剩余数值误差都需处理。

源码共享行索引为 `clamp(cache_position-start, 0, 15)`，应让176取14、177取15。下一步比较对应K是否也复制了177行：若K同样逐位一致，支持已经完成RoPE的整行被复用，优先检查共享Gather索引／Gather之后的数据路径；若K不同，上游features／投影行重复等可能仍需检查。两种结果都不能单凭输出定为某个算子故障。离线工具现在同时输出K配对和保留features行对比。

## DDK 服务器执行

保留原 `output-init/result` 和本次 `rope-fp32/result`。仅重新读取这两份结果，既不重跑设备，也不重新编译或生成参考。

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
audit = d.get('row_audit', {})
summary = {k: d[k] for k in ('status', 'error', 'candidate_model') if k in d}
summary['row_audit'] = {
    k: audit[k] for k in ('changed_case_indices', 'changed_tensor_regions',
        'changed_case_positions', 'fingerprint_coverage', 'corpus') if k in audit}
summary['patterns'] = [
    {'index': p['index'], 'tensor': p['tensor'],
     'rows': [{'position': r['cache_position'],
               'heads': [h['head'] for h in r['heads']],
               'different_elements': sum(h['different_elements'] for h in r['heads'])}
              for r in p['changed_rows']],
     'unchanged_rows': p['unchanged_rows'],
     'unchanged_rows_vs_reference': p['unchanged_rows_vs_reference']}
    for p in audit.get('patterns', [])]
fingerprints = audit.get('value_fingerprints', [])
for row in fingerprints:
    for mode in row['value_modes'].values():
        mode['nearest'] = {k: v[:1] for k, v in mode['nearest'].items()}
summary['value_fingerprints'] = fingerprints
print(json.dumps(summary, ensure_ascii=False))
PY
```

本轮重点是 `value_modes.*.paired_keys` 和 `retained_feature_pair_comparisons`。完整文件保留全部差异区间和每组前三个最近V行；摘要显示每组最接近的一行。若匹配误差大或只有后续调用匹配，不把它判为旧数据来源。feature行哈希和输入KV零值检查只代表保留的文件，不能证明NPU当时读到的内容。搜索最多32个异常位置，并显式报告遗漏数量；全部差异位置仍完整列出。

此前行匹配9项本地检查通过；新增K配对5项检查通过，覆盖K/V同时复制、仅V复制、无重复的对照、搜索上限和features文件行差异。未改变模型、运行器、原golden或容差。case16是只补KV的调用，没有proposal hidden比较；下一轮isolated重载参考KV，因而hidden整体改善不代表该KV异常已解决。
