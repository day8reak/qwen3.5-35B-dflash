# 当前阶段测试命令

本轮C16／cap192／K15整模型推理518.990ms，算子合计142.440ms；入口Conv2D 70.968ms。已确认整模型仍约0.52秒，当前检查CPU/NPU分段及静态16行投影的计算量。

## 1. 查看已有OMC的分段报告

只读取编译时保存的报告，无需重编或上板。子图数量用于定位融合候选，不等于实测切换次数或耗时；逐项只列精度策略检查覆盖的节点。

```bash
python3 - <<'PY'
import json, os
from collections import Counter, defaultdict
from pathlib import Path
path = Path(os.environ['AI_RUN_DIR']) / 'draft/omc/compiled-precision.json'
report = json.loads(path.read_text())
keys = ('status', 'graphop_counts', 'cpu_graphs', 'required_nodes', 'matched_nodes')
print(json.dumps({key: report.get(key) for key in keys}, ensure_ascii=False, indent=2))
groups = defaultdict(Counter)
for node in report.get('nodes', []):
    groups[node.get('owner', 'UNRESOLVED')][node['source_type']] += 1
for owner, counts in sorted(groups.items()):
    print(owner, json.dumps(dict(counts), ensure_ascii=False))
PY
```

当前API计时仅包住同步`client.Process()`。模型CSV另列加载558.829ms、卸载56.097ms。推理与算子合计相差376.550ms，尚未归因；不能直接写成CPU耗时，也不能用算子起点与duration拼出设备执行时间线。

## 2. 可选：离线查看精度误差

当前仍采用约1%误差的阶段性工程基线；旧逐元素规则的失败记录保留，不阻塞性能诊断。该命令读取已回收张量，统计新增KV与已提交区域、Top1差异和feedback零新增行重复性。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --analyze "$AI_RUN_DIR/draft/tests/accuracy-result"
```

保留当前`draft/tests/accuracy`、`accuracy-result`及profiling CSV用于同输入比较。下一步优先定位分段开销，同时评估入口投影与有效context行数优化；更改后再跑独立精度比较和完整流程接通测试。
