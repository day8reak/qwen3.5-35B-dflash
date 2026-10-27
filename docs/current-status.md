# 当前更新与执行命令

[首页](README.md) · [环境配置](environment-setup.md) · [验证记录](validation.md)

更新日期：2026-10-27。当前主线是完整40层target、6层DFlash和可选融合算子。

## 用户现在执行什么

**Attention重试遇到导出器C++的const接口错误，现已修正；conv/RMS仍待解决。** 更新后只重编Attention两例：

```bash
cd "$REPO_DIR"
git pull --ff-only
"$MODEL_PYTHON" targets/cpro/scripts/run_fused_probes.py --all-ops \
  --probe-run "$AI_RUN_DIR/fused-all-9759eb4f9b0b" \
  --retry-cases attention_prefill,attention_decode
```

`--probe-run` 是上次批量运行的目录；沿用 `CANN_ROOT`、`CPRO_DDK_ROOT`、`TBE_DIAG_JOB` 和HDC环境。脚本复用GDN/Cache四个OMC，连同新编译成功的Attention上板；conv/RMS不重复编译，其失败继续保留在总表。回传最后打印的报告，完整注册检查在新目录的 `inspection/report.txt`。脚本不修改DDK。

## 当前状态

| 项目 | 状态 |
|---|---|
| PyTorch完整结构 | 全尺寸meta模型及层级FX图；不加载权重 |
| GE完整结构 | target/draft完整节点、连边、C++和recipe；融合可选 |
| 自定义算子探针 | 五类9例已批量尝试；GDN/CacheUpdate四例加载和执行通过，conv/RMS编译失败，Attention导出失败 |
| 真实权重AIR导出 | 旧双图支持完整FP16；新GE target支持显式缓存 |
| 状态执行 | 主机fake prefill→decode接续通过；真实板端待验证 |
| 尚未完成 | W4A8映射、常驻缓存与native DFlash调度、完整模型设备精度和性能 |

## 已确认与剩余问题

| 项目 | 结论 |
|---|---|
| GDN两例 | 用户回传加载/执行通过；decode核心输出全零，prefill核心输出relL2约0.999；未通过精度验收 |
| 编译目标 | 始终为 `kirinx90`；`Ascend310B1` 是DDK初始化后返回的内部名称，`Ascend310P3` 是原工程平台 |
| GDN状态 | 安装版初态/末态为FP16；本次runner修正了末态只接受FP32的旧校验 |
| conv | 宽度4已通过初始维推；生成的y2为 `[1,8192,1,S+3]`，随后格式配置失败；不能据此断言DDK完全没有kernel |
| RMS | 本次仍找不到 `AdnRmsNorm` 维推函数，尚未进入板端执行 |
| CacheUpdate | prefill/decode两例编译、加载和执行通过；无数值golden，不代表精度通过 |
| Attention | 端口顺序校验已修正；最新重试在C++构建时暴露 `GNode` const不匹配，已按CANN 9.1头文件声明修正，待真实重试 |
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
