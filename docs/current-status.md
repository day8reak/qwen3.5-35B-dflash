# 当前阶段：修复 RoPE 相位精度并回放完整 Draft

2026-11-05：离线复算已返回。四条 Sin/Cos 在 case 0／2 都与“先把相位转 FP16，再计算三角函数并转 FP16”的模拟逐位一致；case 17 的模拟残差也仅为 1.5e-5～6.1e-5。原 FP32 相位在 case 17 被舍入约 0.059～0.062 弧度，支持相位降精度导致偏差。Q 投影仍有独立残差，完整 Draft 精度尚未通过，性能优化暂缓。证据见[结果页](results.md)。

当前候选在编译副本中，把四条 Sin/Cos 改成从原 FP32 相位表预计算的系数查表：先算三角函数，再转 FP16，沿用原位置索引和下游乘加。源 AIR、模型权重、输入、golden 和容差保留。系数以能精确表示 FP16 值的 FP32 常量载体存储，不代表设备内部使用 FP32。NumPy 与参考框架的三角函数舍入仍须通过原完整模型门槛检验。

## DDK 服务器执行

只编译一个完整候选，并回放原 isolated／feedback 各19例。无需重导出 AIR、重新生成参考或再次跑165组探针；不需要 Torch/TorchAir。保留本次编译副本，后续 ABI 绑定需要它。

```bash
git -C "$REPO_DIR" pull --ff-only
export TOP1_DIR="$AI_RUN_DIR/draft/tests/top1-integration"
export ROPE_DIR="$TOP1_DIR/rope-fp32"

python3 "$REPO_DIR/targets/cpro/scripts/compile_incremental_draft.py" \
  --air-export "$TOP1_DIR/export" --rope-fp32-table \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --tbe-diagnosis "$AI_RUN_DIR/common/tbe-diagnosis" \
  --output "$ROPE_DIR/compiled"
```

编译成功后运行原数据回放。工具校验原源图、权重、几何和语义 IO；若新 OMC 输入槽位顺序变化，按语义名称重排原输入文件，逐个核对字节哈希。

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --replay "$TOP1_DIR/output-init/result" \
  --compiled "$ROPE_DIR/compiled" \
  --source "$AI_RUN_DIR/draft/source/prepared" \
  --air-export "$TOP1_DIR/export" --ddk-root "$CPRO_DDK_ROOT" \
  --config "$AI_RUN_DIR/runtime/runtime.json" \
  --output-initialization untouched --output "$ROPE_DIR/run" --apply

python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --recover "$ROPE_DIR/run" --output "$ROPE_DIR/result"
```

若返回仍在运行，只回收同一次执行。贴编译摘要和最后回收摘要即可，重点比较原七处 Top1 差异、hidden 与 appended KV 误差，同时保留未写区检查。Q 投影尚未修复，误差下降不等于完整精度通过。

本地16项检查已通过：完整相位表位置覆盖、图与文件绑定、输入槽位重映射和原数据保护；使用合成数据及模拟编译入口，尚无该候选的真实 OMG 或设备结果。
