# 当前阶段测试命令

目标：C16／cap192／K15，模型常驻并预热后，单次Draft proposal的同步`Process`平均时延≤50ms，包含API内部调度、同步和搬运。加载与初始KV补齐单独统计。当前整模型profiling为518.990ms，尚未达标；最终用相同输入、缓存长度和频率配置，3次预热后测10次，报告均值与最大值。

## 1. 复用FP32小图，上板验证精度与时延

用户已回传5个小图全部编译成功。内置RmsNorm（hidden与head形状）、Softmax、残差Add均为1个NPUCL；RMS原语链为2个NPUCL＋1个CPUCL。下一步复用这些OMC，对每个小图加载一次，预热3次，再测10次同步`Process`并比较全部输出。原语RMS作为分段路径的对照。

在已加载项目环境的DDK服务器执行：

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$AI_RUN_DIR/runtime/native" --replace
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --compiled "$AI_RUN_DIR/draft/tests/vector" \
  --config "$AI_RUN_DIR/runtime/runtime.json" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$AI_RUN_DIR/draft/tests/vector-run" --apply
```

看到`VECTOR_VALIDATION_STARTED`后回收同一任务：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --recover "$AI_RUN_DIR/draft/tests/vector-run" \
  --output "$AI_RUN_DIR/draft/tests/vector-result"
```

若返回`VECTOR_VALIDATION_PENDING`，稍后重复回收命令。结果保存在`vector-result/result.json`：每项包含数值结果、10次API耗时、均值、最大值及输出逐字节重复性。失败看对应`<case>.stderr`；已确认结束后可复用相同测试目录。只重建runner，不重编小图或完整Draft。

小图预设逐元素容差为`atol=1e-5, rtol=1e-4`，检查有限值；输入含零行、大值、小值、Softmax掩码及残差抵消。每个实测调用都保留快照，文件写入与外部输出拷贝不计入API耗时。这是小图诊断时延，不能相加预测完整Draft时延。编译归属和输出一致也不单独证明内部指令精度。

验证工具已通过16项本地检查，包括参考绑定、全部输出比较、native连续调用、归档与恢复；本地使用假执行器及模拟传输，所选OHOS SDK构建与板端验证等待上述命令回传。完整Draft的原精度阈值保持不变。

## 2. 后续接入顺序

先验证FP32 RMS／Softmax／残差的NPU实现，合并每层反复切换的计算区域；接入时保留原有FP16舍入边界。随后优化固定16行context投影的无效计算，再检查词表head和ArgMax。每次改图都复查独立Draft精度、连续KV、整模型时延与端到端接通。

目前70.968ms的入口Conv本身已经超过50ms目标，需要和分段问题一起处理。27／26是静态子图数量，不能直接当成运行时切换次数；整模型与算子合计相差376.550ms的部分尚未全部归因。

保留当前`draft/tests/accuracy`、`accuracy-result`及原始profiling CSV用于同输入比较。约1%误差仍作为阶段性工程基线，原始失败记录与比较阈值保留。Target精度及verify开发继续独立处理。
