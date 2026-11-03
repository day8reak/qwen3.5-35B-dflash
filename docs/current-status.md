# 当前阶段测试命令

目标为C16／cap192／K15常驻同步API预热后平均≤50ms，加载与初始catchup另计。用户已接受最差hidden相对L2 2.61%、Top1 7/172差异作为性能基线；原NUMERICAL_FAILED和golden保留。

最新已测 `packed-spatial-4d`：accuracy API均值189.147～200.515ms，单次profile推理184.568ms、算子汇总78.986ms。105.582ms差额尚未归因。现有依赖报告确认中间CPU子图接收NPU的两个[1,16,1]浮点张量及一个BOOL[1,208]向量，再向后续NPU输出两个[1,16,64]浮点张量及两个mask。这与源码的RoPE位置处理及有效性扩展相符，未解码NPU内部任务；不能把全部差额认作这条边界的时间。

新候选 `packed-spatial-controls` 将位置的INT32 Unsqueeze放在FP32 Cast之前，并把有效性Concat改成精确的0/1 INT32，再沿用现有Gather扩展。目的是让控制计算在NPU主体之前完成，减少中途回CPU的依赖。全词表head、ArgMax、权重和17输入／14输出接口保持。主机缩小宽度的220例全部输出逐位相等；完整形状捕获、权重／接口及模拟编译流程检查通过。实际DDK编译、分区和设备收益尚待验证。

## 现在执行：导出并编译控制量候选

服务器更新代码后，沿用已有REPO_DIR、AI_RUN_DIR、CPRO_DDK_ROOT和CPRO_DDK_ID。已完成候选的测试目录按类型复用；已回收的基线数值、profile和模型SHA记录在[结果](results.md)。此轮不需要重建native runner。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/export_incremental_draft.py" \
  --prepared "$AI_RUN_DIR/draft/source/prepared" \
  --frontend pytorch --optimization packed-spatial-controls \
  --context-rows 16 --cache-capacity 192 --max-draft-tokens 15 \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-air" --replace

python3 "$REPO_DIR/targets/cpro/scripts/compile_incremental_draft.py" \
  --air-export "$AI_RUN_DIR/draft/tests/pytorch-optimized-air" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --tbe-diagnosis "$AI_RUN_DIR/common/tbe-diagnosis" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-omc" --replace
```

回传编译摘要。确认optimization.policy为packed-spatial-controls、17／14接口及graphop_counts。希望中间CPU回流消失，使NPU主体合并；数量减少本身不是成功证明。随后沿用完整accuracy执行／回收，再对同一模型做边界检查和依赖分析，确认实际NPU→CPU→NPU连接是否消除。数值仍按原门槛报告，与已接受基线对照；固定输入、同频率3次预热＋10次测量判断同步API收益。

## 后续：完整词表NPU ArgMax

[Ascend C ReduceMax文档](https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/reduction_compute/ReduceMax.html)明确列出X90支持half／float及返回最大值索引，并列取最小索引。可按UB容量分块归约248320词表，块内索引加INT32偏移，再合并最大值与索引；half索引不能直接覆盖整个词表。目标是让最终归约留在NPU，只输出15个INT32（60字节），消除当前CPU ArgMax的FP32 logits消费边界。

这是可行路线依据，尚未验证所选DDK的头文件、注册和编译支持，也没有实现或测量新ArgMax。下一项先做小型完整词表归约探针，验证NPU归属、索引／并列值及数值行为；不以TopK(k=1)默认替代ArgMax，不裁词表。当前先测控制量候选，分开判断两条边界的收益。
