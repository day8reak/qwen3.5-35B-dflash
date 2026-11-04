# 当前阶段测试命令

目标为C16／cap192／K15常驻同步API预热后平均≤50ms，加载与初始catchup另计。用户已接受最差hidden相对L2 2.61%、Top1 7/172差异作为性能基线；原NUMERICAL_FAILED和golden保留。

`packed-spatial-rope-table` 已由用户完成编译产物的只读检查：OMC为2808047355字节，SHA256 `a045f7629920fd962d21c8ce694b1ee694a3be61942c4cac6a556f00cc631027`。CPUCL3／NPUCL2变为CPUCL2／NPUCL1，中间CPU两个Mul和回流依赖消失，`intermediate_cpu_partitions=[]`。当前结构是前置CPU控制→一个NPU主体→末尾CPU ArgMax；仅剩logits这条NPU→CPU路径。NPU分区20输入／14输出是内部边界，不是公共输入数量。

该模型随后在MODEL_LOAD失败，尚未进入推理。执行cc7399ff087d40be8e647bcdd2f46a7c已回收，锁已释放。日志候选报告单个NPU分区权重2806299136字节（2.614 GiB），随后memcpy_s失败并终止AllocateWeightMemory。公开Secure C默认单次destMax上限为0x7fffffff；大权重块触及拷贝上限是当前优先假设，实际设备库和本次拷贝参数仍未核实。见[结果](results.md)。

## 现在执行：限制分区大小的编译对照

更新代码，沿用已有REPO_DIR、AI_RUN_DIR、CPRO_DDK_ROOT、CPRO_DDK_ID。保留现有相位表AIR、原始权重和native runner，仅重编OMC。新增显式--limit-partition-size true传给所选OMG；省略时保留原来的编译选项。该候选允许多个NPU分区，重点是每个权重块受控且无中间CPU回流。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/compile_incremental_draft.py" \
  --air-export "$AI_RUN_DIR/draft/tests/pytorch-optimized-air" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --tbe-diagnosis "$AI_RUN_DIR/common/tbe-diagnosis" \
  --limit-partition-size true \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-omc" --replace
```

编译摘要新增compiler_options和partition_weights。启用true后，所有GraphOp权重大小须可读且不超过2147483647字节，否则保留编译产物、在partition-weight-check停止，不打包成可执行候选。此阈值是本轮诊断条件，不是设备分配或拷贝参数的实测结果。OMG支持情况和具体分区方式以所选工具结果为准。

编译成功后顺序执行只读检查：

```bash
python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --inspect-boundaries "$AI_RUN_DIR/draft/tests/pytorch-optimized-omc" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-boundaries.json"

python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --analyze-boundaries "$AI_RUN_DIR/draft/tests/pytorch-optimized-boundaries.json" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-dependencies.json"
```

回传编译和依赖摘要，确认每个权重块大小以及intermediate_cpu_partitions为空，再验证新OMC能否加载、数值及同步API时延。当前失败尝试已完成，不重复apply或recover；原失败日志保留在pytorch-optimized-result，编译不会重用该槽位。

上一已测4d仍是profile184.568ms、算子汇总78.986ms。相位表和新分区限制均无设备数值或性能结论，105.582ms差额未归因；原golden、误差门槛和50ms目标保留。
