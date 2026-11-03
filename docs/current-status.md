# 当前阶段测试命令

目标为 C16／cap192／K15 常驻同步 API 预热后平均 ≤50ms，计入 API 内部调度、同步和搬运，加载与初始catchup另计。用户已接受上一版最差hidden相对L2 2.61%、Top1 7/172差异作为性能基线；原NUMERICAL_FAILED和golden保留。

第一轮packed-spatial的API均值为279.693～298.002ms，单次V1厂商推理245.989ms，Conv 8.664ms。随后V2对照为258.682ms、Load507.168ms，未见收益，继续用V1。新边界报告确认仍有3个CPU／2个NPU子图：Conv前的五维Permute落CPU且接口变为FP32，六层mask仍重复展开，末尾CPU ArgMax仍接收14.9MB FP32 logits。约70ms算子汇总与模型时间的差额仍未归因，不能按边界字节数认定搬运耗时。完整记录见[结果](results.md)。

本轮候选为 `packed-spatial-4d`：Conv前改为两个四维Transpose，共享full／sliding两种mask。保持完整K点积、原权重、完整词表Top1及17／14接口。220次缩小宽度连续回放全部14个输出与上一空间版逐位一致；完整宽度投影、meta捕获、权重绑定和mock导出检查通过。尚未用所选DDK编译或上板，四维Transpose是否留在NPU仍需实测。

## 导出和编译候选

沿用现有custom_pypto环境、REPO_DIR、AI_RUN_DIR及所选DDK。下面复用已完成的优化候选AIR／OMC测试槽，替换的是候选产物；prepared权重、公共模型和原组合版数值基线保留。上一轮profile CSV和数值报告用于历史对照，不能拿旧accuracy结果去绑定新OMC。

```bash
git -C "$REPO_DIR" pull --ff-only

python3 "$REPO_DIR/targets/cpro/scripts/export_incremental_draft.py" \
  --frontend pytorch --optimization packed-spatial-4d \
  --prepared "$AI_RUN_DIR/draft/source/prepared" \
  --context-rows 16 --cache-capacity 192 --max-draft-tokens 15 \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-air" --replace

python3 "$REPO_DIR/targets/cpro/scripts/compile_incremental_draft.py" \
  --air-export "$AI_RUN_DIR/draft/tests/pytorch-optimized-air" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-omc" --replace
```

先核对编译摘要中的optimization.policy、graphop_counts和新OMC SHA。编译成功不代表放置改善或提速；随后用新模型绑定完整精度回放，再以默认V1记录同样样本的profile，读取与新accuracy结果绑定的边界报告。重点检查Conv前CPU Permute和FP32接口是否消失、mask展开是否共享，以及完整同步调用时间。完整词表NPU Top1尚未实现，本轮不能承诺达到50ms。
