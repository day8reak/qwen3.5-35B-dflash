# 当前阶段：定位 hidden 最早的数值偏差

2026-11-05：完整Draft仍未通过精度门槛。`untouched` 回放消除了首例额外零值，但hidden相对L2仍约1.34%–2.61%；离线head已用device hidden复现全部7处Top1翻转。按当前安排，先定位并改善精度，性能优化暂缓。已有结果见[结果页](results.md)。

本轮检查第0层 `input RMSNorm → Q投影 → Q RMSNorm`。两个小图从当前AIR裁取，保留原计算节点和三份权重：一个只输出Q归一化结果，另一个暴露五个检查点。输入复用empty-context原始FP16 embedding，比较容差不变。**裁剪和增加输出可能改变编译，探针结果不能直接当作整图首错位置。**

## 用户现在执行什么

更新代码后在DDK服务器执行；需要已保留的export、prepared和output-init/result，不需要Torch、TorchAir或重新导出整Draft。

```bash
export TOP1_DIR="$AI_RUN_DIR/draft/tests/top1-integration"
export HIDDEN_DIR="$TOP1_DIR/hidden-first-query"

python3 "$REPO_DIR/targets/cpro/tools/probe_draft_hidden.py" \
  --air-export "$TOP1_DIR/export" \
  --source "$AI_RUN_DIR/draft/source/prepared" \
  --result "$TOP1_DIR/output-init/result" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --tbe-diagnosis "$AI_RUN_DIR/common/tbe-diagnosis" \
  --output "$HIDDEN_DIR/compiled"
```

需要两个case均为`COMPILED`。只编译小图，不替换整Draft OMC。随后更新共享运行器，使tensor probe支持显式`untouched`输出模式；这是运行器构建，不是重编整模型：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$AI_RUN_DIR/runtime/native" --replace

python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --compiled "$HIDDEN_DIR/compiled" --ddk-root "$CPRO_DDK_ROOT" \
  --config "$AI_RUN_DIR/runtime/runtime.json" \
  --output "$HIDDEN_DIR/run" --apply
```

启动后回收同一请求，不重复启动：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --recover "$HIDDEN_DIR/run" --output "$HIDDEN_DIR/result"

python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --analyze "$HIDDEN_DIR/result" > "$HIDDEN_DIR/diagnosis.json"
cat "$HIDDEN_DIR/diagnosis.json"
```

## 如何读结果

- 先看`normal_vs_tapped_terminal_all_calls_bitwise_equal`和`repeat_bitwise_stable`。若不同，先排查探针对编译或输出可见性的影响。
- `first_nonexact_checkpoint`表示首次非逐位一致；`first_failed_checkpoint`才表示首次超过原容差。每个检查点均比较十次输出，并报告源码节点、误差和CPU/NPU归属。
- 同时列出原参考的gamma舍入边界，供判断差异来源；这组备选计算不替换golden，不改变验收。

若两图稳定且全部检查点通过，下一步扩到第0层attention／残差；若某点失败，围绕该点验证精度表达。当前本地检查已完成，真实DDK编译和设备数值尚待本轮回传。贴编译摘要及diagnosis.json文本即可，无需打包。

完整构建流程见[执行说明](run-end-to-end.md)。
