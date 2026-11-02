# 当前阶段测试命令

目标：C16／cap192／K15，模型常驻并预热后，单次Draft proposal的同步`Process`平均时延≤50ms，包含API内部调度、同步和搬运。加载与初始KV补齐单独统计。现有整模型profiling为518.990ms，目标尚未达到。

## 1. 当前FP32小图结论

| 小图 | 同步Process均值 | 当前结论 |
|---|---:|---|
| RMS原语链 | 1.5722ms | L2误差88.12%；2个NPUCL＋1个CPUCL |
| Softmax | 1.7261ms | L2误差0.1434%；原逐元素阈值失败 |
| 残差Add | 0.8412ms | 整体L2误差1.3361%；大值抵消行708.78% |
| 内置RmsNorm hidden／heads | 未执行推理 | 两者均在HIAI Load返回1 |

已执行的小图均为3次预热、10次计时，输出逐位稳定；耗时排除加载、外部输出拷贝和快照写盘。不能由这些小图耗时相加预测完整Draft时延。

本地使用相同输入生成配方复算：`float32(float16(float16(x) + float16(residual)))`重现了用户回传Add的整体和四组误差、失败元素数及20个具体值。这强烈支持有效FP16舍入，尚不定位舍入发生在运行链路的哪一层，也不证明内部指令类型。该路径不能当成保留FP32计算的替换。

当前`vector-run`、`vector-result`已是Softmax／Add／head RMS这一批。原语RMS当前只有此前回传结论，不能用新批次快照分析它。保留当前两目录供下面的本地分析使用。

## 2. 确认全张量舍入特征

沿用现有`REPO_DIR`、`AI_RUN_DIR`，更新脚本后分析现有快照：

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --analyze "$AI_RUN_DIR/draft/tests/vector-result"
```

不连接HDC、不编译或运行模型。工具仍以原阈值重比全部10次输出，并分析第1次快照。`outputs[].precision.rounding_models`比较完整Add张量与四种模型：FP32相加、仅输出舍入、仅输入舍入、输入和输出均舍入。查看`exact_fp32_bit_matches`是否等于`elements=32768`。`fp16_roundtrip_exact_elements`只描述输出数值能否精确表示为FP16，不单独证明内部计算精度。分析不会修改原始失败状态或阈值。

## 3. 显式FP32权重的两项对照

用户回传的所选OMG帮助明确：`--weight_data_type`支持FP16（默认）和FP32，仅作用于原始FP32权重且未指定`compress_conf`的场景。此前省略该选项不等于保留FP32权重。帮助没有列出直接控制中间算术精度的选项；`--mode=6`虽标为精度模拟JSON，当前未取得其调用和解释方式，不作为已验证手段。

内置RmsNorm含FP32常量`gamma`；先显式请求FP32权重编译，检验常量转换与加载失败是否有关，不预设它就是根因。另跑没有浮点权重的残差Add作为对照，不能预期权重选项自动修复动态输入的算术精度。

完成上面的旧结果分析后，复用固定目录，只编译这两个小图：

```bash
python3 "$REPO_DIR/targets/cpro/tools/probe_draft_vector.py" \
  --ge-root "$CANN_ROOT" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --weight-data-type FP32 \
  --cases residual_fp32 rms_builtin_heads \
  --output "$AI_RUN_DIR/draft/tests/vector" --replace
```

看到`COMPILED`后，复用已有native运行；无需重新构建runner：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --compiled "$AI_RUN_DIR/draft/tests/vector" \
  --config "$AI_RUN_DIR/runtime/runtime.json" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --cases residual_fp32 rms_builtin_heads \
  --output "$AI_RUN_DIR/draft/tests/vector-run" --apply
```

收到`VECTOR_VALIDATION_STARTED`后回收；尚未完成则继续回收同一目录：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --recover "$AI_RUN_DIR/draft/tests/vector-run" \
  --output "$AI_RUN_DIR/draft/tests/vector-result"
```

回收有已完成小图时，再运行第2节`--analyze`命令。此次对照保持源图、输入配方、阈值及3次预热／10次测量不变。回收结果和固定目录复用后的简要记录会保留`compiler_options.weight_data_type=FP32`，避免与默认编译混淆。

- Add若仍复现FP16舍入，不能再归因于权重转换；后续需核对NPU算子实现／执行库配置。
- RmsNorm若仍在Load失败，记录该阶段，结束只改此选项的尝试；若能加载，还必须通过`y`和`rstd`数值检查。
- 只有实际编译归属与数值均满足要求，才考虑接入Draft；此对照未改变完整Draft的AIR、OMC或精度策略。

本地舍入分析7项、编译选项传递／子集准备／结果兼容10项检查通过。所选DDK实际编译和板端执行待用户回传。

## 4. 后续接入

先解决FP32区域的有效精度与RmsNorm加载问题，再验证每层分段是否减少。接入时保留原有FP16舍入边界，复查独立Draft精度、连续KV与整模型时延。约1%的阶段性误差接受不替代新算术路径验证。

随后优化固定16行context投影的无效计算，以及词表head和ArgMax。入口Conv曾记录70.968ms，需要与分段问题一起处理；27／26是静态CPUCL／NPUCL子图数，整模型与算子合计相差376.550ms的部分仍未全部归因。

完整Draft暂未改图。最终在相同输入、缓存长度和频率配置下，3次预热后测10次proposal，报告均值和最大值，并完成端到端接通检查。Target精度和verify开发继续独立处理。
