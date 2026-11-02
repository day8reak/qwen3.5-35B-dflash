# 当前阶段测试命令

目标：C16／cap192／K15，模型常驻并预热后，单次Draft proposal的同步`Process`平均时延≤50ms，包含API内部调度、同步和搬运。加载与初始KV补齐单独统计。当前整模型profiling为518.990ms，尚未达标；最终用相同输入、缓存长度和频率配置，3次预热后测10次，报告均值与最大值。

## 1. 定位原语RMS的数值错误

原语RMS已完成3次预热、10次测量：平均1.5722ms，最小1.232ms，最大1.928ms，10次输出逐位一致。但输出相对L2误差为**88.12%**、最大绝对误差11.5117，全部有限值，数值失败。这远超此前完整Draft约1%的阶段性误差，当前探针实现不能接入；也不能用该探针误差替换原完整Draft的精度结论。

内置RmsNorm hidden在HIAI `Load`阶段返回1，runtime为`800.631.120.010`；其余3项未启动，打包成功且锁已释放。该加载问题与原语RMS的数值错误分别定位。

沿用已有`REPO_DIR`、`AI_RUN_DIR`。更新脚本后，从本地现有快照分析误差，不连接HDC、不编译、不运行模型：

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --analyze "$AI_RUN_DIR/draft/tests/vector-result"
```

工具重新比较全部10次快照，并将第1次输出按零输入行、大值行、小值行、普通输入行分组，报告误差、失败元素数、最差位置与实际/参考值。`least_squares_scale`是实际输出对参考的拟合倍数，配合`cosine`用于识别缩放偏差，不据此直接认定根因。原始结果和阈值保持不变。

先回传分析输出，完成本轮定位前保留`vector-run`和`vector-result`，避免下一批复用固定目录时删除所需快照。

## 2. 执行剩余3个小图

完成上述定位、现有快照可清理后，复用现有OMC与native，按下列顺序单独测试Softmax、残差Add、head形状的RmsNorm。沿用`CPRO_DDK_ROOT`及HDC配置。`--cases`按给定顺序执行；每个小图加载一次，预热3次，再测10次。某个小图执行失败仍停止该请求，回收时保留已完成项。

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --compiled "$AI_RUN_DIR/draft/tests/vector" \
  --config "$AI_RUN_DIR/runtime/runtime.json" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --cases softmax_fp32 residual_fp32 rms_builtin_heads \
  --output "$AI_RUN_DIR/draft/tests/vector-run" --apply
```

沿用固定`vector-run`目录，旧请求已确认结束才会复用，已回收的原语RMS小结写入`draft/results/`。收到`VECTOR_VALIDATION_STARTED`后回收；若仍在运行，对同一目录继续回收：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --recover "$AI_RUN_DIR/draft/tests/vector-run" \
  --output "$AI_RUN_DIR/draft/tests/vector-result"
```

`PASSED`仅覆盖`validated_cases`列出的本次小图。内置RmsNorm hidden的加载问题仍待定位；原语RMS为2个NPUCL＋1个CPUCL，可作为分段对照，不能当成全NPU RMS。

小图预设逐元素容差为`atol=1e-5, rtol=1e-4`，检查有限值；输入含零行、大值、小值、Softmax掩码及残差抵消。每个实测调用都保留快照，文件写入与外部输出拷贝不计入API耗时。这是小图诊断时延，不能相加预测完整Draft时延。编译归属和输出一致也不单独证明内部指令精度。

回收与选择执行已通过13项本地主机检查；本次数值摘要与离线分组分析另通过11项检查，包含非有限值、跨轮最大误差和本地只读分析。后续小图数值与时延仍以板端回传为准。

## 3. 后续接入顺序

先验证FP32 RMS／Softmax／残差的NPU实现，合并每层反复切换的计算区域；接入时保留原有FP16舍入边界。随后优化固定16行context投影的无效计算，再检查词表head和ArgMax。每次改图都复查独立Draft精度、连续KV、整模型时延与端到端接通。

目前70.968ms的入口Conv本身已经超过50ms目标，需要和分段问题一起处理。27／26是静态子图数量，不能直接当成运行时切换次数；整模型与算子合计相差376.550ms的部分尚未全部归因。

保留当前`draft/tests/accuracy`、`accuracy-result`及原始profiling CSV用于同输入比较。约1%误差仍作为阶段性工程基线，原始失败记录与比较阈值保留。Target精度及verify开发继续独立处理。
