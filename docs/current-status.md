# 当前阶段测试命令

目标：C16／cap192／K15，模型常驻并预热后，单次Draft proposal的同步`Process`平均时延≤50ms，包含API内部调度、同步和搬运。加载与初始KV补齐单独统计。当前整模型profiling为518.990ms，尚未达标；最终用相同输入、缓存长度和频率配置，3次预热后测10次，报告均值与最大值。

## 1. 取回已完成的原语RMS结果

5个小图已编译，native runner已构建。板端原语RMS正常退出；内置RmsNorm的hidden形状在HIAI `Load`阶段返回1，runtime为`800.631.120.010`。后面3个小图未启动，输出打包成功，任务已结束并释放锁。该错误发生在推理前，仅凭返回码1还无法确定底层原因。

沿用已有`REPO_DIR`、`AI_RUN_DIR`、`CPRO_DDK_ROOT`和HDC环境配置，更新Python脚本后回收原`vector-run`。此步不启动模型，也无需重编OMC或native：

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --recover "$AI_RUN_DIR/draft/tests/vector-run" \
  --output "$AI_RUN_DIR/draft/tests/vector-result"
```

新版会取回打包数据，比较原语RMS已完成的10次输出，显示`cases[].passed`、`mean_ms`和`max_ms`。整组仍为`FAILED`，命令仍返回1；若原语RMS数值通过，`numerical_validation=PARTIAL_PASSED`，`validated_cases`仅包含它。详细结果存于`vector-result/result.json`。如果仍无`cases`，先保留原执行目录处理回收错误。

## 2. 执行剩余3个小图

取得上一步的原语RMS结果后，复用现有OMC与native，按下列顺序单独测试Softmax、残差Add、head形状的RmsNorm。`--cases`按给定顺序执行；每个小图加载一次，预热3次，再测10次。某个小图执行失败仍停止该请求，回收时保留已完成项。

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

本次部分回收与选择执行通过13项本地主机检查，包含真实shell的顺序与失败停止检查。设备精度与时延以上述命令回传为准。

## 3. 后续接入顺序

先验证FP32 RMS／Softmax／残差的NPU实现，合并每层反复切换的计算区域；接入时保留原有FP16舍入边界。随后优化固定16行context投影的无效计算，再检查词表head和ArgMax。每次改图都复查独立Draft精度、连续KV、整模型时延与端到端接通。

目前70.968ms的入口Conv本身已经超过50ms目标，需要和分段问题一起处理。27／26是静态子图数量，不能直接当成运行时切换次数；整模型与算子合计相差376.550ms的部分尚未全部归因。

保留当前`draft/tests/accuracy`、`accuracy-result`及原始profiling CSV用于同输入比较。约1%误差仍作为阶段性工程基线，原始失败记录与比较阈值保留。Target精度及verify开发继续独立处理。
