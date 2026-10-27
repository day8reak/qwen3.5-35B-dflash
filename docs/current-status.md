# 当前更新与执行命令

[首页](README.md) · [环境配置](environment-setup.md) · [验证记录](validation.md)

更新日期：2026-10-27。当前主线是完整40层target、6层DFlash和可选融合算子。

## 用户现在执行什么

**先跑通全部五类自定义算子的调用。** `--all-ops` 复用已有GDN OMC，补编conv、RMS、CacheUpdate和分页Attention，共覆盖9个用例；编译失败不阻止其它用例上板。

沿用现有环境、`FUSED_PROBE_RUN`、`CANN_ROOT`、`CPRO_DDK_ROOT` 和 `TBE_DIAG_JOB`。更新代码后执行：

```bash
cd "$REPO_DIR"
git pull --ff-only
"$MODEL_PYTHON" targets/cpro/scripts/run_fused_probes.py --all-ops
```

脚本只构建一次runner，自动上传、查询版本、执行和回收；结果放在 `$AI_RUN_DIR/fused-all-*`。conv使用DDK日志要求的宽度4候选，输出顺序尚未确认；RMS仍使用原名 `AdnRmsNorm`，缺注册会明确列为失败。

回传最后打印的 `report.txt` 总表；各组详细结果位于 `board-*/comparison/report.txt`。所有日志保留。当前先检查调用覆盖，GDN数值异常保留记录，全部调用确认后再集中对齐精度。

## 当前状态

| 项目 | 状态 |
|---|---|
| PyTorch完整结构 | 全尺寸meta模型及层级FX图；不加载权重 |
| GE完整结构 | target/draft完整节点、连边、C++和recipe；融合可选 |
| 自定义算子探针 | 五类9例代码已齐；GDN两例用户回传加载/执行通过，但数值明显偏离；其余待批量验证 |
| 真实权重AIR导出 | 旧双图支持完整FP16；新GE target支持显式缓存 |
| 状态执行 | 主机fake prefill→decode接续通过；真实板端待验证 |
| 尚未完成 | W4A8映射、常驻缓存与native DFlash调度、完整模型设备精度和性能 |

## 已确认与剩余问题

| 项目 | 结论 |
|---|---|
| GDN两例 | 用户回传加载/执行通过；decode核心输出全零，prefill核心输出relL2约0.999；未通过精度验收 |
| 编译目标 | 始终为 `kirinx90`；`Ascend310B1` 是DDK初始化后返回的内部名称，`Ascend310P3` 是原工程平台 |
| GDN状态 | 安装版初态/末态为FP16；本次runner修正了末态只接受FP32的旧校验 |
| conv / RMS | 状态宽度4与文档3不一致 / 当前维推注册缺失，分别待处理 |
| CacheUpdate / Attention | 新增prefill/decode独立探针；Attention包含动态K/V端口及可选端口占位，待真实编译/上板 |
| 尚未验证 | 全部五类调用覆盖、正确数值及NPU执行位置；完整模型与连续状态验证另行进行 |

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
