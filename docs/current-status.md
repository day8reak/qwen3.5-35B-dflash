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

**生成匹配安装源码的GDN状态绑定，然后读取TBE底层平台错误。** 不再重读旧日志，不编译模型、不上板。`--gdn-state-source` 校验并记录安装源码的FP16初态/末态指针；`--python` 使用本次报错的 `custom_pypto` 解释器（当前 `MODEL_PYTHON`）。

```bash
cd "$REPO_DIR"
git pull --ff-only
export FUSED_NEXT_JOB="$(mktemp -d "$AI_RUN_DIR/fused-platform-XXXXXXXX")"
export GE_MODEL_TOOL="$REPO_DIR/targets/cpro/scripts/direct_model_air.py"
export GE_CUSTOM_BINDINGS="$FUSED_NEXT_JOB/bindings.json"
"$MODEL_PYTHON" "$GE_MODEL_TOOL" bindings-from-document \
  --target kirinx90 --ddk-id "$CPRO_DDK_ID" \
  --gdn-initial-state-dtype float16 \
  --gdn-state-source "$CPRO_DDK_ROOT/ddk/ascendc/ops/impl/custom/chunk_gated_delta_rule.cpp" \
  --output "$GE_CUSTOM_BINDINGS"
"$MODEL_PYTHON" "$GE_MODEL_TOOL" check-tbe-platform \
  --ddk-root "$CPRO_DDK_ROOT" --python "$MODEL_PYTHON" \
  --output "$FUSED_NEXT_JOB/platform"
cat "$FUSED_NEXT_JOB/platform/report.txt"
```

回传 `platform/report.txt`。平台检查会在独立子进程导入TBE并调用一次底层 `cce.te_update_version`，保留原始返回值、模块来源和加载库；30秒超时。不修改DDK、不安装包、不切换SoC。`FAILED` 也会生成报告；`RAW_UPDATE_SUCCEEDED` 仅表示这一次调用成功，尚未复现完整OMG工作进程，先不要重跑五例编译。

| 已定位 | 下一步要核对 |
|---|---|
| GDN包装层隐藏了底层平台函数的非success返回值 | 直接记录原始返回值；Conda安装路径本身不证明版本混用 |
| 安装版GDN用 `half*` 读初态、写末态 | 已增加源码绑定的FP16状态选项，真实编译和运行待验证 |
| conv输入 `x2` 要求宽度4，文档写3 | 安装版状态顺序、输出与控制参数；不能直接给生产缓存补一个槽位 |
| `AdnRmsNorm` 无可用维推函数 | 当前DDK的注册与实现是否配套 |

原文档的FP32末态描述保留，安装源码覆盖另存 `.sources.json` 并记录SHA256。Conv/RMS仍未修复，不要直接使用本绑定启用这两个融合算子。后续编译现场会保存在 `cases/<用例>/compile-work/`。

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
