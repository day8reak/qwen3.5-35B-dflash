# 当前更新与执行命令

[首页](README.md) · [环境配置](environment-setup.md) · [验证记录](validation.md)

更新日期：2026-10-27。当前主线是完整40层target、6层DFlash和可选融合算子。

## 当前状态

| 项目 | 状态 |
|---|---|
| PyTorch完整结构 | 全尺寸meta模型及层级FX图；不加载权重 |
| GE完整结构 | target/draft完整节点、连边、C++和recipe；融合可选 |
| 融合算子探针 | 5例AIR成功、OMC失败：GDN平台初始化异常；conv状态宽度不符；RMS维推注册缺失 |
| 真实权重AIR导出 | 旧双图支持完整FP16；新GE target支持显式缓存 |
| 状态执行 | 主机fake prefill→decode接续通过；真实板端待验证 |
| 尚未完成 | W4A8映射、常驻缓存与native DFlash调度、完整模型设备精度和性能 |

## 用户现在执行什么

**执行下面这一轮平台排查即可。** 上轮四组全部失败，单改库顺序没有解决问题。本轮 `--diagnose` 最多12组、`--jobs 4` 并行执行，覆盖Python包来源、库顺序和平台更新参数；每组30秒超时，完成一组立即显示结果。

目标仍是 **CPro/X90（`kirinx90`）**。文档的 `Ascend310P3` 是原工程平台；日志的 `Ascend310B1` 是当前DDK调用 `te_set_version("kirinx90")` 后报告的内部名称。OMG命令一直使用 `--platform=kirinx90`，这轮只检查两个平台API是否使用不同名称。

```bash
cd "$REPO_DIR"
git pull --ff-only
export TBE_DIAG_JOB="$(mktemp -d "$AI_RUN_DIR/tbe-diagnose-XXXXXXXX")"
export GE_MODEL_TOOL="$REPO_DIR/targets/cpro/scripts/direct_model_air.py"
"$MODEL_PYTHON" "$GE_MODEL_TOOL" check-tbe-platform \
  --ddk-root "$CPRO_DDK_ROOT" --python "$MODEL_PYTHON" \
  --diagnose --jobs 4 --timeout-seconds 30 \
  --output "$TBE_DIAG_JOB/diagnosis"
cat "$TBE_DIAG_JOB/diagnosis/report.txt"
```

回传 `diagnosis/report.txt`。测试维度如下，按组合运行：

| 维度 | 对照项 |
|---|---|
| Python包 | 当前环境；当前DDK自带的唯一 `te-*.whl` 解压到本次运行目录 |
| 动态库 | OMG目录优先；TBE目录优先 |
| 平台更新参数 | `kirinx90`；初始化后查询到的SOC名；空字符串（包装API省略SOC时使用的值） |

全部先以 `kirinx90` 初始化。DDK wheel副本只用于独立诊断进程，不执行pip安装；来源不完整或不唯一会记录失败/跳过。`DIAGNOSIS_COMPLETE` 表示报告收集完成，某组更新成功也还需真实OMG编译验证。现有FP16 `GE_CUSTOM_BINDINGS` 保留，本轮不重建AIR或运行设备。

| 已定位 | 下一步要核对 |
|---|---|
| 两种库顺序下，初始化均报告 `Ascend310B1`，随后传 `kirinx90` 更新均返回 `error` | 初始化/更新API的名称处理是否不同；这是待验证假设 |
| 四组均使用当前环境的TBE Python包 | 当前DDK自带wheel与已安装包的行为是否一致 |
| 安装版GDN用 `half*` 读初态、写末态 | 已增加源码绑定的FP16状态选项，真实编译和运行待验证 |
| conv输入 `x2` 要求宽度4，文档写3 | 安装版状态顺序、输出与控制参数；不能直接给生产缓存补一个槽位 |
| `AdnRmsNorm` 无可用维推函数 | 当前DDK的注册与实现是否配套 |

原文档的FP32末态描述保留，安装源码覆盖另存 `.sources.json` 并记录SHA256。Conv/RMS仍未修复，不要直接使用本绑定启用这两个融合算子。后续编译现场会保存在 `cases/<用例>/compile-work/`。

首次生成安装版GDN状态绑定时执行；本次已成功生成的无需再运行：

```bash
"$MODEL_PYTHON" "$GE_MODEL_TOOL" bindings-from-document \
  --target kirinx90 --ddk-id "$CPRO_DDK_ID" \
  --gdn-initial-state-dtype float16 \
  --gdn-state-source "$CPRO_DDK_ROOT/ddk/ascendc/ops/impl/custom/chunk_gated_delta_rule.cpp" \
  --output "$GE_CUSTOM_BINDINGS"
```

## 首次生成与编译（当前失败批次无需重跑）

**先在黄区一次性编译5个融合算子用例，确认DDK能识别接口。** 无需模型权重；先按[环境配置](environment-setup.md)设置 `REPO_DIR`、`MODEL_PYTHON`、`AI_RUN_DIR`、`CANN_ROOT`、`CPRO_DDK_ROOT` 和 `CPRO_DDK_ID`。

下面更新代码、生成接口绑定，再导出AIR并编译OMC。`--target kirinx90` 对应当前CPro编译平台，`--gdn-initial-state-dtype float16` 选择文档Cpro的FP16初态。

```bash
cd "$REPO_DIR"
git pull --ff-only

export FUSED_PROBE_JOB="$(mktemp -d "$AI_RUN_DIR/fused-probes-XXXXXXXX")"
export GE_CUSTOM_BINDINGS="$FUSED_PROBE_JOB/bindings.json"
export FUSED_PROBE_RUN="$FUSED_PROBE_JOB/probes"
export GE_MODEL_TOOL="$REPO_DIR/targets/cpro/scripts/direct_model_air.py"

"$MODEL_PYTHON" "$GE_MODEL_TOOL" bindings-from-document \
  --target kirinx90 --ddk-id "$CPRO_DDK_ID" \
  --gdn-initial-state-dtype float16 \
  --output "$GE_CUSTOM_BINDINGS"

"$MODEL_PYTHON" "$GE_MODEL_TOOL" probe-fused-ops \
  --bindings "$GE_CUSTOM_BINDINGS" --ge-root "$CANN_ROOT" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$FUSED_PROBE_RUN"

cat "$FUSED_PROBE_RUN/report.txt"
```

全部编译成功后，在同一终端继续[板端批量执行命令](board-guide.md#融合算子调用探针)，无需等待确认。完成后回传 `$FUSED_PROBE_JOB/comparison/report.txt`；编译失败则先回传 `$FUSED_PROBE_RUN/report.txt`。

## 完整模型构图（按需）

需要完整模型结构时选择下面一条路径；`sequence-length=8`只限定输入长度，模型层数、专家和词表均为完整尺寸。

**PyTorch：** 生成完整meta模型和层级FX结构，不需要检查点或TorchAir。

```bash
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli build-full-pytorch \
  --sequence-length 8 --output "$AI_RUN_DIR/torch-full-$(date +%Y%m%d-%H%M%S)"
```

**GE：** 生成完整target/draft构图源码，默认不使用融合算子，不需要检查点或GE库。

```bash
"$MODEL_PYTHON" "$REPO_DIR/targets/cpro/scripts/direct_model_air.py" build-full-structure \
  --sequence-length 8 --expert-dispatch selected --fused-ops none \
  --output "$AI_RUN_DIR/ge-full-$(date +%Y%m%d-%H%M%S)"
```

这些是未绑定真实权重的结构产物。实际导出AIR：按 [PyTorch](pytorch-guide.md)或 [GE](ge-guide.md)指南继续；无需重跑旧探针。

## 文档融合接口

上面的 `GE_CUSTOM_BINDINGS` 可复用于GE融合构图。绑定来自用户文档，不代表kernel已安装或已在目标芯片验证；融合构图命令见[GE指南](ge-guide.md)。

完整W4A8 + FP16 DFlash先按24–28GiB可用RAM规划；这不是实测峰值，也不适用于当前FP16权重导出，见[内存预算](environment-setup.md#内存预算文档-w4a8--dflash)。
