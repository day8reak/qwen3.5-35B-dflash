# 当前阶段测试命令

目标为C16／cap192／K15常驻同步API预热后平均≤50ms，加载与初始catchup另计。用户已接受原组合版最差hidden相对L2 2.61%、Top1 7/172差异作为性能基线；原NUMERICAL_FAILED和golden保留。

2026-11-03，用户回传 `packed-spatial-4d` 编译成功：17输入／14输出，3个CPU／2个NPU子图；四维特征重排、共享mask、packed GQA和共享RoPE契约均存在。源图6组组合attention，编译主图未见融合attention节点，GraphOp内部仍不透明。compiled_abi=NOT_BOUND、compiled_precision=NOT_VALIDATED、device_execution=NOT_RUN；下一步由验证入口绑定语义到物理槽位，核对真实输出。当前未提供新OMC SHA或细分边界，不能把上一版边界、数值和时延当成新结果。

上一packed-spatial的API均值为279.693～298.002ms、单次V1推理245.989ms、Conv8.664ms；V2对照258.682ms，未见收益，继续V1。子图数量仍为5不代表节点放置、接口尺寸或时延完全相同。此次重点看Conv前CPU Permute／FP32边界和重复mask是否改善；完整词表CPU ArgMax仍是后续优化项。详见[结果](results.md)。

## 绑定并验证新模型

沿用现有custom_pypto环境和REPO_DIR／AI_RUN_DIR／CPRO_DDK_ROOT。无需再次导出编译或重建native。下面复用已完成的优化测试槽；脚本检查旧任务完成证据后才复用，活动或退出不明的任务会拒绝。保留原组合版pytorch-result作为数值基线。

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --config "$AI_RUN_DIR/runtime/runtime.json" \
  --compiled "$AI_RUN_DIR/draft/tests/pytorch-optimized-omc" \
  --source "$AI_RUN_DIR/draft/source/prepared" \
  --air-export "$AI_RUN_DIR/draft/tests/pytorch-optimized-air" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-run" --apply

python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --recover "$AI_RUN_DIR/draft/tests/pytorch-optimized-run" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-result"
```

启动后只recover同一执行。若回收显示PENDING，稍后再执行recover，不重复apply。完成且输出已回收后，比对Top1差异、hidden／新增KV相对L2、未写KV及重复稳定性；沿用原门槛，NUMERICAL_FAILED仍如实保留。若出现新的明显数值恶化，先定位，不将其当作已接受基线。

## 读取边界并用V1记录时延

以下步骤使用刚完成的新accuracy结果。边界检查仅在服务器读OMC及DDK schema；profile新增一次明确的厂商调用。allow-numerical-failure仅允许已完成数值失败结果做诊断，不把精度改为通过。

```bash
python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --inspect-boundaries "$AI_RUN_DIR/draft/tests/pytorch-optimized-omc" \
  --accuracy-result "$AI_RUN_DIR/draft/tests/pytorch-optimized-result" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-boundaries.json"

python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --accuracy "$AI_RUN_DIR/draft/tests/pytorch-optimized-result" \
  --modelmanager-itf V1 --allow-numerical-failure \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-profile-run" --apply

python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --recover "$AI_RUN_DIR/draft/tests/pytorch-optimized-profile-run" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-profile-result"
```

回传accuracy摘要、新边界报告，以及本次model/op CSV。核对新OMC SHA、PID和model归属后，再与上一版V1比较；逻辑字节数不是实测拷贝量，算子时间加和不是模型墙钟，单次profile也不是50ms预热稳态验收。
