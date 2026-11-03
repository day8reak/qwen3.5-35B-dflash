# 当前阶段测试命令

目标为C16／cap192／K15常驻同步API预热后平均≤50ms，加载与初始catchup另计。用户已接受最差hidden相对L2 2.61%、Top1 7/172差异作为性能基线；原NUMERICAL_FAILED和golden保留。

最新已测 `packed-spatial-4d`：accuracy API均值189.147～200.515ms，单次profile推理184.568ms、算子汇总78.986ms。105.582ms差额尚未归因。其依赖报告确认中间CPU接收两个来自NPU的浮点[1,16,1]张量及一个BOOL[1,208]向量，再向后续NPU输出两个[1,16,64]张量和两个mask，与源码RoPE位置处理及有效性扩展相符。

新候选 `packed-spatial-controls` 已由用户完成编译，policy及INT32位置／Concat改写字段确认，仍为CPUCL3／NPUCL2、17输入／14输出。compiled_abi=NOT_BOUND、compiled_precision=NOT_VALIDATED，设备未执行；尚未收到新OMC SHA。不沿用上一模型的数值、分区细节或时延。数量未变，不能仅凭此判断三条依赖是否改变。

## 现在执行：只读检查新OMC及中间回流

服务器更新代码后，沿用已有REPO_DIR、AI_RUN_DIR、CPRO_DDK_ROOT。此入口现已支持只读已完成编译的OMC，不要求先运行accuracy；仍核对所选DDK schema与模型打包记录，并记录检查前后模型SHA。无需重编、上传或上板。

```bash
python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --inspect-boundaries "$AI_RUN_DIR/draft/tests/pytorch-optimized-omc" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-boundaries.json"

python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --analyze-boundaries "$AI_RUN_DIR/draft/tests/pytorch-optimized-boundaries.json" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-dependencies.json"
```

两条顺序执行，第一条成功后再运行第二条。回传两条摘要，重点看新模型SHA、cpu_subgraphs、intermediate_cpu_partitions和npu_to_cpu。编译器可能重新编号子图，因此按端口来源、shape及Cast路径比较，不能只按SubGraph_2名字判断。

这次省略--accuracy-result：旧结果绑定上一版OMC，不能附到新模型上。输出accuracy_binding=NOT_PROVIDED、source_numerical_validation=NOT_EVALUATED是预期状态；只表示没有附带精度证据，不表示通过或失败。若显式提供accuracy结果，仍要求模型和夹具哈希匹配，不会遇到不匹配就静默降级为无绑定模式。

先确认改写是否改变中间回流，再决定保留或调整候选。设备数值验证及固定输入、同频率3次预热＋10次测量仍待执行；元数据不能给105.582ms差额分配传输或等待时间。NPU ArgMax作为单独实验，当前尚未接入。完整证据见[结果](results.md)。
