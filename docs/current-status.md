# 当前阶段：Draft 首轮 hidden 零块定位

2026-11-05：首轮isolated的hidden有12931个零，feedback同例只有3个；多出的12928个零恰好覆盖全部差异，其余元素逐位相同。Top1和12路KV也逐位相同。已发现运行器分配输出后会由CPU清零，下一步验证省略这次写入是否影响首轮零块；尚未确定缓存、写回或buffer生命周期根因，原数值失败保持。

profiling已回收：同一 `fadec11fab3e7…` OMC、case 2，model inference 66.556ms，StableArgMax 0.415ms；热点是末尾MatMul、Conv和布局转换。保留原CSV，不再重跑采集。测量范围和精度见[结果](results.md)。

## 用户现在执行什么

DDK／HDC服务器更新仓库后，重编运行器并复用旧精度夹具。无需Torch/TorchAir、权重加载、AIR导出或OMC重编。保留 `top1-integration/run`、`result`、`compiled` 和原fixture；本次只新增 `output-init` 测试槽位。

```bash
source "$REPO_DIR/targets/cpro/scripts/env.sh"
export TOP1_DIR="$AI_RUN_DIR/draft/tests/top1-integration"

python3 "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$AI_RUN_DIR/runtime/native" --replace

python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --replay "$TOP1_DIR/result" --config "$AI_RUN_DIR/runtime/runtime.json" \
  --output-initialization untouched \
  --output "$TOP1_DIR/output-init/run" --apply
```

`--replay` 是一次新诊断执行，沿用原输入、golden、38次调用顺序和容差；`untouched` 仅省略调用方对输出的初始化写入，SDK分配行为不变。输入仍照原方式初始化，首轮不跳过。普通运行默认仍为 `zero`。出现启动确认后只回收同一请求：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --recover "$TOP1_DIR/output-init/run" --output "$TOP1_DIR/output-init/result"
```

若为PENDING，继续同一回收命令，不重复 `--apply`。完成后读取首例差异及零值统计：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --diagnose "$TOP1_DIR/output-init/result" > "$TOP1_DIR/output-init/diagnosis.json"
python3 - "$TOP1_DIR/output-init/diagnosis.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
print(json.dumps({'initial_call_comparison': d['initial_call_comparison'],
                  'first_cases': [r for r in d['cases'] if r['index'] == 0]}, ensure_ascii=False))
PY
```

回传回收摘要与上述诊断即可。即使零块消失，也须单独比较hidden/KV原有误差及Top1，不能把初始化实验当作整图精度通过或缓存根因证明。原模型、算子和生产配置保持；首次构建见[完整流程](run-end-to-end.md)。
