# 当前阶段测试命令

当前目标为 C16／cap192／K15 常驻同步 API 预热后平均 ≤50ms，计入 API 内部调度、同步和搬运，加载与初始catchup另计。用户已接受上一版最差hidden相对L2 2.61%、Top1 7/172差异作为性能基线；原NUMERICAL_FAILED和golden保留。

2026-11-03，用户回传packed-spatial结果：API均值279.693～298.002ms，单次厂商模型推理245.989ms，Conv 8.664ms；前版分别约736～774ms、627.647ms、69.038ms。Top1差异及hidden/KV误差摘要与前版相同，未写KV与重复稳定性保持；不等于所有输出已逐位比较。约70ms算子汇总与245.989ms整模型时间的差额仍未归因，50ms尚未达到。完整记录见[结果](results.md)。

## 先隔离运行接口差异

已知Target工具入口用V2和外置权重，Draft工具入口用V1和内嵌权重；尚不能据此断言V1慢。下一次只将同一Draft OMC通过V2从文件恢复，保留同一profile样本、完整输入、单次调用、频率和优先级。无需重新导出、编译OMC、构建native或上传大模型。V2兼容性和性能尚未实测，失败会保留日志，不回退执行V1。

沿用现有custom_pypto环境。REPO_DIR为仓库，AI_RUN_DIR为已有运行根目录；需要已完成的draft/tests/pytorch-optimized-result和它绑定的fixture，板端原OMC缓存保持。该命令新增一次明确的V2对照调用，不是恢复或重跑已完成的V1任务。

```bash
git -C "$REPO_DIR" pull --ff-only

python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --accuracy "$AI_RUN_DIR/draft/tests/pytorch-optimized-result" \
  --modelmanager-itf V2 --allow-numerical-failure \
  --output "$AI_RUN_DIR/draft/tests/pytorch-interface-profile-run" --apply

python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --recover "$AI_RUN_DIR/draft/tests/pytorch-interface-profile-run" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-interface-profile-result"
```

启动与回收摘要应标明model_manager_interface=V2，回收不再指定接口。返回两张原始CSV，核对本次PID/model归属后，与保留的pytorch-optimized-profile-result（V1）对照模型inference、Conv、末尾MatMul及CPU ArgMax。若算子耗时接近而整模型明显下降，优先排查接口/恢复路径；若仍有大差额，继续查图内CPU/NPU边界。两者都不是根因已证明或50ms稳态验收；使用V2作为正式执行路径前仍需完整输出精度与固定输入重复测试。

## 读取新候选边界

旧版CPUCL=3／NPUCL=2以及14.9MB logits到CPU ArgMax的报告不能直接作为新图报告。下面只在服务器读取现有OMC和所选DDK schema，不调用OMG或设备；CPRO_DDK_ROOT沿用100.520.020.010对应路径。

```bash
python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --inspect-boundaries "$AI_RUN_DIR/draft/tests/pytorch-optimized-omc" \
  --accuracy-result "$AI_RUN_DIR/draft/tests/pytorch-optimized-result" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-boundaries.json"
```

回传graphop_counts、CPU子图和最大边界张量。逻辑尺寸不是实测拷贝量，也不能直接换算为差额时长。本轮代码保留V1默认入口、原模型和原数值门槛，10组主机检查通过；设备执行仍由用户完成。
