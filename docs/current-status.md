# 当前阶段测试命令

目标：C16／cap192／K15，常驻预热后，单次Draft proposal同步`Process`平均≤50ms，包含API内部调度、同步和搬运；加载与初始KV补齐另计。现有整模型profiling为518.990ms，目标尚未达到。

## 1. 当前结论

显式`weight_data_type=FP32`对照已结束：Add仍未通过数值检查，head RmsNorm仍在HIAI Load返回1。该权重选项没有修复动态输入Add的精度，停止重复此项测试。完整Draft精度策略保持原状。

| 小图 | 同步Process均值 | 当前结论 |
|---|---:|---|
| RMS原语链，默认权重选项 | 1.5722ms | L2误差88.12%；2个NPUCL＋1个CPUCL |
| Softmax，默认权重选项 | 1.7261ms | L2误差0.1434%；原逐元素阈值失败 |
| Add，默认权重选项 | 0.8412ms | 整体L2误差1.3361%；大值抵消行708.78% |
| Add，显式FP32权重 | 0.6651ms | 整体L2与最大绝对误差和默认编译相同；跨编译输出逐位比较尚未完成 |
| 内置RmsNorm hidden／heads | 未执行推理 | 默认编译两者均Load失败；显式FP32权重的heads仍失败 |

已执行小图均为3次预热、10次计时，各批次内输出逐位稳定；耗时排除加载、外部输出拷贝和快照写盘。不同批次均值不作为已确认加速，不能相加预测完整Draft时延。

用户对默认权重批次的全张量分析已经确认：Add的32768个输出全部逐位匹配`FP32(FP16(FP16(x)+FP16(residual)))`，FP32直接加法匹配0个。Softmax的106496个输出都能精确表示为FP16。

显式FP32权重的Add主图随后确认存在两个输入`CastT(float32→float16)`、一个输入输出均为float16的NPUCL GraphOp，以及输出`CastT(float16→float32)`。输入精度损失已经定位到编译产物的NPU子图边界；内部指令是否还使用其他精度仍未知。`NO_DUMPS`只表示没有独立IR dump，不推翻主图证据。

```text
x FP32        → graphop_data_cast_0 → FP16 ┐
                                          ├→ NPUCL (FP16输入/输出) → FP32输出
residual FP32 → graphop_data_cast_1 → FP16 ┘                       netoutput_cast_0
```

## 2. 查找实现选择依据

沿用现有`REPO_DIR`、`AI_RUN_DIR`、`CPRO_DDK_ROOT`。主图检查已完成，不必重复。下面只读当前Add的编译日志及所选OMG目录中的配置文本，不重新编译、构建runner或连接板端。使用`rg`筛选；每份文件最多显示规定条数，无匹配时返回1，不代表测试失败：

```bash
git -C "$REPO_DIR" pull --ff-only
rg -n -i -m 120 \
  'graphop_data_cast|netoutput_cast|precision|implementation|select.?kernel|custom_ascendc|add.*(dtype|support|kernel)|sum.*(dtype|support|kernel)' \
  "$AI_RUN_DIR/draft/tests/vector/residual_fp32/omc/compile.log"

rg -n -i -m 12 \
  --glob '*.json' --glob '*.ini' --glob '*.cfg' \
  --glob '*.md' --glob '*.txt' --glob '*.py' --glob '*.h' \
  'npucl_implementation|execute_device_config|keep_origin_dtype|must_keep_origin_dtype|precision_mode' \
  "$CPRO_DDK_ROOT/tools/tools_omg"
```

需查明配置的正式schema、可选实现及其FP32输入支持。`CheckSupport`行是候选查询，不等于最终选中；关键字或枚举值本身也不是有效配置证明。若本目录没有相关文本，请提供该DDK附带的配置说明或样例路径，不能从无匹配推断不支持。不要盲试实现编号：现有精度策略中的7表示`custom_ascendc_lib`，并非“强制FP32”。

## 3. FP32 Add的可行路径

当前所选OMG帮助没有列出已验证的中间算术保FP32开关。`weight_data_type`控制权重；源图和输入输出声明FP32也已被本次结果证明不足以保证实际算术。其他编译器的`precision_mode`或`keep_dtype`不能直接当成本版OMG的有效参数。

当前完整Draft采用已检查的CPU FP32路径。要迁移到NPU，必须选中并验证FP32实现，保证进入加法前的有效输入精度也得到保留。不能直接删掉两个输入Cast而继续使用现有FP16接口子图；这会造成接口不匹配。仅增加输出FP32 Cast也无法恢复已丢失的输入信息。

公开[AscendC Add文档](https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/basic_arithmetic/Add.html)列出X90支持`Add<float>`；但[华为算子工程工具文档](https://developer.huawei.com/consumer/cn/doc/doccenter-capabilities/cannkit-creating-operator-project-msopgen)同时注明KirinX90暂不支持float类型输入。这分别描述计算API与工程接口，不能据此宣称当前DDK已打通自定义FP32输入链路。自定义实现需先核对该DDK的接口和注册支持，不能只在kernel内换成float就视为完成。

单独Add验证保留大值抵消样本：`1040.5068359375 + (-1040.496826171875)`的FP32结果是`0.010009765625`，输入先转FP16后为`1.0`。原阈值与完整张量比较继续保留，不以约1%的整图误差接受替代此项验证。

## 4. 后续接入

确认有效精度后再减少每层CPU/NPU分段，并考虑融合残差与RMS以减少调度边界；独立Draft精度、连续KV和整模型时延均需复查。

随后优化固定16行context投影的无效计算，以及词表head和ArgMax。入口Conv曾记录70.968ms；27／26是静态CPUCL／NPUCL子图数，整模型与算子合计相差376.550ms的部分仍未全部归因。最终在相同输入、缓存长度和频率配置下，3次预热后测10次proposal，报告均值和最大值。Target精度和verify开发继续独立处理。
