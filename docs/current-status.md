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

**平台排查已收敛，现在只重编译GDN prefill/decode两例。** `--tbe-diagnosis` 使用刚完成的12组报告，为本次OMG进程生成平台名称适配；`--cases` 跳过已知接口不匹配的conv/RMS。保留已生成的FP16状态 `GE_CUSTOM_BINDINGS`。

```bash
cd "$REPO_DIR"
git pull --ff-only
export FUSED_PROBE_JOB="$(mktemp -d "$AI_RUN_DIR/fused-gdn-XXXXXXXX")"
export FUSED_PROBE_RUN="$FUSED_PROBE_JOB/probes"
export GE_MODEL_TOOL="$REPO_DIR/targets/cpro/scripts/direct_model_air.py"
"$MODEL_PYTHON" "$GE_MODEL_TOOL" probe-fused-ops \
  --bindings "${GE_CUSTOM_BINDINGS:?使用已生成的安装版FP16状态绑定}" \
  --ge-root "$CANN_ROOT" --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --cases gdn_prefill,gdn_decode \
  --tbe-diagnosis "${TBE_DIAG_JOB:?设置刚完成的平台诊断目录}/diagnosis" \
  --output "$FUSED_PROBE_RUN"
cat "$FUSED_PROBE_RUN/report.txt"
```

回传 `report.txt`，包含实际编译结果和名称适配是否被工作进程调用。两例均编译成功后，可继续[板端批量执行](board-guide.md#融合算子调用探针)，脚本只执行本次选中的用例。

**目标始终是CPro/X90：OMG仍传 `--platform=kirinx90`。** 当前DDK初始化该目标后报告内部名称 `Ascend310B1`；适配仅把TBE底层更新接口的参数改为经本次报告验证的当前名称，不改初始化目标。代码副本放在本次运行目录，不改DDK或当前Python安装。文档的 `Ascend310P3` 是原工程平台。

| 当前结论 | 后续验证 |
|---|---|
| 12组中，更新传 `kirinx90` 均失败；传内部名称/空值均成功且SOC保持 | 此次用内部名称适配重跑真实OMG |
| 两种库顺序结果相同；当前环境与DDK wheel所列模块哈希一致 | 本轮保留OMG库优先，不重装Python包 |
| 安装版GDN初态和末态均用FP16 | 用已有源码绑定重新生成AIR并编译 |
| conv状态宽度要求4、RMS维推注册缺失 | 分别处理；本轮不重复编译 |

编译现场保留在 `cases/<用例>/compile-work/`；平台适配与调用记录在该用例的 `tbe-compat/`。API测试成功还不代表OMC编译或板端精度通过。

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
