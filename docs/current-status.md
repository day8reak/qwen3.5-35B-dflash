# 当前阶段测试命令

目标：C16／cap192／K15，模型常驻并预热后，单次Draft proposal的同步`Process`平均时延≤50ms，包含API内部调度、同步和搬运。加载与初始KV补齐单独统计。当前整模型profiling为518.990ms，尚未达标；最终用相同输入、缓存长度和频率配置，3次预热后测10次，报告均值与最大值。

## 1. 编译FP32小图，检查NPU支持

已收到现有OMC的27个CPUCL／26个NPUCL子图报告。先编译RMS原语链、内置RmsNorm（hidden与head形状）、Softmax和残差Add五个小图，保留FP32输入输出，不指定CPU策略。测试不需要模型权重，不启动板端任务；当前Draft AIR／OMC保持可运行。

在已加载项目环境的DDK服务器执行：

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/tools/probe_draft_vector.py" \
  --ge-root "$CANN_ROOT" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$AI_RUN_DIR/draft/tests/vector" --replace
```

汇总在`draft/tests/vector/result.json`，每项的`placement.json`记录CPU/NPU归属及边界dtype。`COMPILED`表示小图编译和归属读取完成；`NPU_GRAPH_PRESENT`只表示存在NPU子图，仍需确认其覆盖范围，并上板比较数值、测耗时。它不证明内部FP32运算或Vector指令已经验证。

失败日志保存在对应小图的`build.log`、`export.log`或`omc/`内；修复后复用同一个`vector`目录。`--replace`仅替换该工具已结束的测试，不清理正在运行的任务。此脚本已完成本地CPU与CLI验证，所选DDK编译等待本轮回传。

## 2. 后续接入顺序

先验证FP32 RMS／Softmax／残差的NPU实现，合并每层反复切换的计算区域；接入时保留原有FP16舍入边界。随后优化固定16行context投影的无效计算，再检查词表head和ArgMax。每次改图都复查独立Draft精度、连续KV、整模型时延与端到端接通。

目前70.968ms的入口Conv本身已经超过50ms目标，需要和分段问题一起处理。27／26是静态子图数量，不能直接当成运行时切换次数；整模型与算子合计相差376.550ms的部分尚未全部归因。

保留当前`draft/tests/accuracy`、`accuracy-result`及原始profiling CSV用于同输入比较。约1%误差仍作为阶段性工程基线，原始失败记录与比较阈值保留。Target精度及verify开发继续独立处理。
