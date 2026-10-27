# 当前更新与执行命令

[首页](README.md) · [环境配置](environment-setup.md) · [验证记录](validation.md)

更新日期：2026-10-27。当前主线是完整40层target、6层DFlash和可选融合算子。

## 用户现在执行什么

**按用户决定，模型所需自定义算子保留GDN、CacheUpdate、Attention三类六例，Conv状态和RMS走当前基础算子路径。** `fused-all-f6b81ead7ae8` 已有六例调用和 `board-1` 数值诊断；源图及OMC未变，当前无需再次上板，先读取现有报告检查GDN精度：

```bash
git pull --ff-only &&
cat "$AI_RUN_DIR/fused-all-f6b81ead7ae8/board-1/comparison/report.txt"
```

优先核对GDN核心输出和末态的误差、非零数量，确认旧报告中的decode全零和prefill大偏差是否仍存在。六例调用通过及 `DIAGNOSTIC_COMPLETE` 均不等于精度通过。

只有后续确需重新执行这六例时，才使用下面入口：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_fused_probes.py" --model-ops \
  --probe-run "$AI_RUN_DIR/fused-all-f6b81ead7ae8"
```

命令复用现有六例OMC，构建runner并上板执行、回收诊断；Conv/RMS不再作为此模式的必需项。旧九例 `--all-ops` 保留为可选诊断，原Conv两例和AdnRmsNorm编译失败仍按失败记录，不因退出当前依赖范围而改为通过。

原工程中 `LinearAttentionConvstateUpdate` 用于Cpro，`AdnRmsNorm` 用于ND，Cpro的RMS调用内置 `npu_rms_norm`。本仓库当前选择现有基础算子实现Conv历史拼接/截取和RMS，生产GE只启用 `--fused-ops gdn`。CacheUpdate/Attention的独立探针通过尚不等于已接入生产GE模型；完整模型及数值验证继续保留。

若再次进入Docker，恢复conda/DDK后执行 `source targets/cpro/scripts/env.sh` 即可；配置含自动生成的HIAI头文件及三库路径，详见[环境恢复指南](environment-setup.md#docker-重进后的配置恢复)。上述命令要求保留当前批次，保留权重无需重新准备。

## 当前状态

| 项目 | 状态 |
|---|---|
| PyTorch完整结构 | 全尺寸meta模型及层级FX图；不加载权重 |
| GE完整结构 | target/draft完整节点、连边、C++和recipe；融合可选 |
| 自定义算子探针 | 当前选择GDN/CacheUpdate/Attention三类6例，用户回传导出、编译、加载及执行通过；原Conv/RMS三例失败保留为可选诊断 |
| 真实权重AIR导出 | 旧双图支持完整FP16；新GE target支持显式缓存 |
| 状态执行 | 主机fake prefill→decode接续通过；真实板端待验证 |
| 尚未完成 | W4A8映射、常驻缓存与native DFlash调度、完整模型设备精度和性能 |

## 已确认与剩余问题

| 项目 | 结论 |
|---|---|
| GDN两例 | 新批次两例调用完成、数值诊断已生成；旧报告的decode核心输出全零、prefill核心输出relL2约0.999，尚未证实已解决；未通过精度验收 |
| 编译目标 | 始终为 `kirinx90`；`Ascend310B1` 是DDK初始化后返回的内部名称，`Ascend310P3` 是原工程平台 |
| GDN状态 | 安装版初态/末态为FP16；本次runner修正了末态只接受FP32的旧校验 |
| conv | 当前走基础算子的历史拼接/截取路径；原Cpro自定义Conv两例在rank-4 y2维推后格式配置失败，历史结论不变 |
| RMS | 当前走基础算子RMS路径；原ND用的 `AdnRmsNorm` 探针缺维推函数，保留失败；不再尝试其他融合RMS候选 |
| CacheUpdate | 新批次prefill/decode两例导出、编译、加载和执行通过，数值诊断已生成；无数值golden，不代表精度通过 |
| Attention | 新批次prefill/decode两例导出、编译、加载和执行均通过，先前导出器问题不再阻止调用；数值仅为DIAGNOSTIC_COMPLETE，尚未验收 |
| 尚未验证 | 当前六例的正确数值及NPU执行位置；CacheUpdate/Attention生产GE接入、基础Conv/RMS的完整模型目标验证与连续状态验证仍待完成 |

新增调用源码的[核对与勘误](qwen35-moe-operators-and-quantization-analysis.md#调用源码核对与勘误2026-10-27)确认：Attention converter把Python `pse_shift`实参接到GE端口16，端口3留空，当前探针方向正确；Conv的输出rank、GDN的Meta/安装版状态dtype和Cache的Functionalize调用名仍有差异，不能凭前端声明判定安装版行为。

## 完整模型构图（按需）

需要完整模型结构时选择下面一条路径；`sequence-length=8`只限定输入长度，模型层数、专家和词表均为完整尺寸。

**PyTorch：** 生成完整meta模型和层级FX结构，不需要检查点或TorchAir。

```bash
python3 -m qwen35_moe_dflash.cli build-full-pytorch \
  --sequence-length 8 --output "$AI_RUN_DIR/torch-full-$(date +%Y%m%d-%H%M%S)"
```

**GE：** 生成完整target/draft构图源码，仅选择GDN融合，不需要检查点或GE库。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/direct_model_air.py" build-full-structure \
  --sequence-length 8 --expert-dispatch selected --fused-ops gdn \
  --bindings "$AI_RUN_DIR/fused-all-f6b81ead7ae8/bindings.json" \
  --output "$AI_RUN_DIR/ge-full-$(date +%Y%m%d-%H%M%S)"
```

这些是未绑定真实权重的结构产物。当前生产GE融合接口支持 `gdn,conv_state,rms` 三个角色，本次只选择 `gdn`，Conv状态和RMS保留基础算子。CacheUpdate/Attention尚未通过该接口接入生产图。实际导出AIR：按 [PyTorch](pytorch-guide.md)或 [GE](ge-guide.md)指南继续；无需重跑旧探针。

## 文档融合接口

本次fresh批次的 `bindings.json` 可复用于GE融合构图，需要时在构图命令前设置其路径。绑定包含文档端口与安装源码的GDN状态声明，不代表全部kernel已安装或已在目标芯片验证；融合构图命令见[GE指南](ge-guide.md)。

完整W4A8 + FP16 DFlash先按24–28GiB可用RAM规划；这不是实测峰值，也不适用于当前FP16权重导出，见[内存预算](environment-setup.md#内存预算文档-w4a8--dflash)。
