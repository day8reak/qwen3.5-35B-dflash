# 当前更新与执行命令

[首页](README.md) · [环境配置](environment-setup.md) · [验证记录](validation.md)

更新日期：2026-10-27。当前主线是完整40层target、6层DFlash和可选融合算子。

## 当前状态

| 项目 | 状态 |
|---|---|
| PyTorch完整结构 | 全尺寸meta模型及层级FX图；不加载权重 |
| GE完整结构 | target/draft完整节点、连边、C++和recipe；融合可选 |
| 融合算子探针 | 3个算子、5个独立GE用例；生成到回收的主机联调通过，真实调用待验证 |
| 真实权重AIR导出 | 旧双图支持完整FP16；新GE target支持显式缓存 |
| 状态执行 | 主机fake prefill→decode接续通过；真实板端待验证 |
| 尚未完成 | W4A8映射、常驻缓存与native DFlash调度、完整模型设备精度和性能 |

## 用户现在执行什么

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
