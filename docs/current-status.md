# 当前阶段测试命令

目标为C16／cap192／K15常驻同步API预热后平均≤50ms，加载与初始catchup另计。用户已接受最差hidden相对L2 2.61%、Top1 7/172差异作为性能基线；原NUMERICAL_FAILED和golden保留。

`packed-spatial-rope-table` 已由用户完成编译产物的只读检查：OMC为2808047355字节，SHA256 `a045f7629920fd962d21c8ce694b1ee694a3be61942c4cac6a556f00cc631027`。CPUCL3／NPUCL2变为CPUCL2／NPUCL1，中间CPU两个Mul和回流依赖消失，`intermediate_cpu_partitions=[]`。当前结构是前置CPU控制→一个NPU主体→末尾CPU ArgMax；仅剩logits这条NPU→CPU路径。NPU分区20输入／14输出是内部边界，不是公共输入数量。

新模型尚无设备精度或时延。上一已测4d版本仍是profile184.568ms、算子汇总78.986ms，两者105.582ms差额尚未归因；不能直接将其视为这次收益。相位表的220例主机回放及实际head_dim128检查通过，编译后的常量精度和执行结果仍需设备验证。

## 现在执行：当前OMC的完整accuracy

沿用已有REPO_DIR、AI_RUN_DIR、CPRO_DDK_ROOT和native runner。无需再次导出或编译。已完成测试槽位按类型复用，旧模型身份及结果已记录在[结果](results.md)。启动一次：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --config "$AI_RUN_DIR/runtime/runtime.json" \
  --compiled "$AI_RUN_DIR/draft/tests/pytorch-optimized-omc" \
  --source "$AI_RUN_DIR/draft/source/prepared" \
  --air-export "$AI_RUN_DIR/draft/tests/pytorch-optimized-air" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-run" --apply
```

返回DRAFT_ACCURACY_STARTED后，回收同一次执行：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --recover "$AI_RUN_DIR/draft/tests/pytorch-optimized-run" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-result"
```

若为PENDING，仅重复recover；不重启apply。完整结果重点看native_api_latency、Top1差异、hidden／新增KV的相对L2、未写KV保持以及两组重复稳定性。仍使用原0.01／0.01逐元素门槛，NUMERICAL_FAILED可能与已接受基线一致，需要看实际差异，不能仅看状态判定退化或通过。

先回传这份结果，确认数值变化及API时延后再决定是否采集新profile。accuracy里的时延来自不同上下文和proposal数的调用，不能代替固定输入、同频率3次预热＋10次测量的50ms验收。末尾CPU ArgMax仍是后续独立优化项。
