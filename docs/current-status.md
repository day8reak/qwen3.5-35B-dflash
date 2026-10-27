# 当前更新与执行命令

[首页](README.md) · [环境配置](environment-setup.md) · [验证记录](validation.md)

更新日期：2026-10-27。当前主线是完整40层target、6层DFlash和可选融合算子。

## 用户现在执行什么

**当前是Docker重进且旧run产物已删除。** 按[环境指南的四步顺序](environment-setup.md#docker-重进后的配置恢复)执行：恢复conda/DDK → 首次补七项export → 保存仓库旁配置 → 加载并运行。DDK脚本已经导出的变量不用重复配置。

已有配置时，恢复 `custom_pypto` conda/DDK环境后，在仓库根目录直接执行；以下 `python3` 使用当前已激活的模型环境：

```bash
git pull --ff-only &&
source targets/cpro/scripts/env.sh &&
  python3 "$REPO_DIR/targets/cpro/scripts/run_fused_probes.py" --all-ops --fresh
```

不再引用已经删除的bindings、TBE诊断或OMC；安装版FP16绑定、平台诊断和全部探针自动重建，编译成功者上板。保留权重无需重新准备。

**后续重试：** 等本次fresh打印出新的批次目录后，才使用下面命令。将占位路径换成该新目录；已经删除的历史批次不能复用。Attention导出器的const接口修正已提交，Conv/RMS仍待解决。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_fused_probes.py" --all-ops \
  --probe-run '/实际路径/本次新生成的fused-all批次' \
  --retry-cases attention_prefill,attention_decode
```

脚本复用新批次内已成功编译的OMC，只重编指定缺项；TBE诊断自动从新批次记录继承，无需另export `TBE_DIAG_JOB`。未重试的失败继续保留在总表。回传最后打印的报告；如该批次包含Conv/RMS失败，其注册检查另写到 `inspection/report.txt`。脚本不修改DDK。

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
| RMS | 编译时仍找不到 `AdnRmsNorm` 维推函数；新增调用摘录表明它用于ND，原Cpro用内置 `npu_rms_norm`，实际GE映射待核对；本仓库可选 `rms` 融合绑定仍依赖Adn |
| CacheUpdate | prefill/decode两例编译、加载和执行通过；无数值golden，不代表精度通过 |
| Attention | 端口顺序校验已修正；最新重试在C++构建时暴露 `GNode` const不匹配，已按CANN 9.1头文件声明修正，待真实重试 |
| 尚未验证 | 全部五类调用覆盖、正确数值及NPU执行位置；完整模型与连续状态验证另行进行 |

新增调用源码的[核对与勘误](qwen35-moe-operators-and-quantization-analysis.md#调用源码核对与勘误2026-10-27)确认：Attention converter把Python `pse_shift`实参接到GE端口16，端口3留空，当前探针方向正确；Conv的输出rank、GDN的Meta/安装版状态dtype和Cache的Functionalize调用名仍有差异，不能凭前端声明判定安装版行为。

## 完整模型构图（按需）

需要完整模型结构时选择下面一条路径；`sequence-length=8`只限定输入长度，模型层数、专家和词表均为完整尺寸。

**PyTorch：** 生成完整meta模型和层级FX结构，不需要检查点或TorchAir。

```bash
python3 -m qwen35_moe_dflash.cli build-full-pytorch \
  --sequence-length 8 --output "$AI_RUN_DIR/torch-full-$(date +%Y%m%d-%H%M%S)"
```

**GE：** 生成完整target/draft构图源码，默认不使用融合算子，不需要检查点或GE库。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/direct_model_air.py" build-full-structure \
  --sequence-length 8 --expert-dispatch selected --fused-ops none \
  --output "$AI_RUN_DIR/ge-full-$(date +%Y%m%d-%H%M%S)"
```

这些是未绑定真实权重的结构产物。实际导出AIR：按 [PyTorch](pytorch-guide.md)或 [GE](ge-guide.md)指南继续；无需重跑旧探针。

## 文档融合接口

本次fresh批次的 `bindings.json` 可复用于GE融合构图，需要时在构图命令前设置其路径。绑定包含文档端口与安装源码的GDN状态声明，不代表全部kernel已安装或已在目标芯片验证；融合构图命令见[GE指南](ge-guide.md)。

完整W4A8 + FP16 DFlash先按24–28GiB可用RAM规划；这不是实测峰值，也不适用于当前FP16权重导出，见[内存预算](environment-setup.md#内存预算文档-w4a8--dflash)。
