# 当前阶段测试命令

目标为C16／cap192／K15常驻同步API预热后平均≤50ms，加载与初始catchup另计。用户已接受最差hidden相对L2 2.61%、Top1 7/172差异作为性能基线；原NUMERICAL_FAILED和golden保留。最近有设备时延的4d版本仍是profile184.568ms、算子汇总78.986ms，105.582ms差额尚未归因。

用户已完成 `packed-spatial-controls` 的编译和只读依赖检查：模型SHA为63b94d9f05ec42a9ef8521fcc03fc800cec9b155f011251850aca28f7a9dcb6b。mask回流已消除；中间CPU只剩两个Mul，接收NPU返回的两个浮点[1,16,1]，输出两个[1,16,64]。仍为CPUCL3／NPUCL2，新模型尚未执行设备数值和时延测试。

新候选 `packed-spatial-rope-table` 把RoPE的FP32位置乘频率预计算成表，用INT32位置Gather代替运行时乘法，再沿用Sin／Cos。表覆盖cache_capacity+max(context_rows,K+1)行，包含满缓存时的填充context位置；当前208×64×4=53248字节。17／14接口及原始权重不变，新增确定性FP32常量。主机220例六层回放全部14输出逐字节相同；实际head_dim128的24组几何、594组控制、八档完整形状捕获及权重／模拟流程检查通过。实际编译分配和设备收益待验证。

## 现在执行：导出、编译，再只读检查

服务器更新代码后，沿用已有REPO_DIR、AI_RUN_DIR、CPRO_DDK_ROOT、CPRO_DDK_ID。已完成测试槽位按类型复用，已收到的模型身份和观察记录在[结果](results.md)。以下步骤顺序执行，每一步成功后再继续；不需要重建native runner。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/export_incremental_draft.py" \
  --prepared "$AI_RUN_DIR/draft/source/prepared" \
  --frontend pytorch --optimization packed-spatial-rope-table \
  --context-rows 16 --cache-capacity 192 --max-draft-tokens 15 \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-air" --replace

python3 "$REPO_DIR/targets/cpro/scripts/compile_incremental_draft.py" \
  --air-export "$AI_RUN_DIR/draft/tests/pytorch-optimized-air" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --tbe-diagnosis "$AI_RUN_DIR/common/tbe-diagnosis" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-omc" --replace

python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --inspect-boundaries "$AI_RUN_DIR/draft/tests/pytorch-optimized-omc" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-boundaries.json"

python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --analyze-boundaries "$AI_RUN_DIR/draft/tests/pytorch-optimized-boundaries.json" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-dependencies.json"
```

回传编译摘要和最后两条摘要。先确认新policy、17／14接口、模型SHA，再检查是否还有中间CPU分区及两个频率Mul回流。编译器可能重新编号，按操作、端口和依赖比较。NPU主体合并为一个分区是预期，不能由源码查表就宣称已经达成。

本轮只读检查省略--accuracy-result，避免旧结果与新模型错配；accuracy_binding=NOT_PROVIDED、source_numerical_validation=NOT_EVALUATED是预期状态。若中间回流消除，再执行完整accuracy与固定输入、同频率3次预热＋10次同步API测量。预计算仅在CPU源数学中已证等价；编译后常量精度、Gather／Sin／Cos分配及实际性能仍需检查。末尾CPU ArgMax是后续独立实验。
