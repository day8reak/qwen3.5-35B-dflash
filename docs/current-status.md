# 当前更新与执行命令

[首页](README.md) · [环境配置](environment-setup.md) · [验证记录](validation.md)

更新日期：2026-10-27。当前主线是完整40层target、6层DFlash和可选融合算子。

## 用户现在执行什么

**最新批次已恢复并完成九例尝试。** `fused-all-f6b81ead7ae8` 中GDN、CacheUpdate、Attention共六例调用通过；Conv两例和AdnRmsNorm仍编译失败。保留该批次，先读取其失败日志与所选DDK的注册文本：

```bash
git pull --ff-only &&
python3 "$REPO_DIR/targets/cpro/scripts/run_fused_probes.py" --inspect-only \
  --probe-run "$AI_RUN_DIR/fused-all-f6b81ead7ae8"
```

命令自动在 `AI_RUN_DIR` 下创建新的 `fused-inspect-*` 目录，并直接打印摘要。它支持单次探针目录和all-ops批次；只读取、复制现有证据，不编译、不构建runner、不连接设备。回传打印的摘要即可，完整日志和匹配的安装文本路径也会打印。`INSPECTED` 仅表示检查完成，不改变原三例失败结论。

此前 `--fresh` 漏掉了Conv/RMS失败后的自动检查，现已补齐；以后新批次出现这类失败时也写入 `inspection/report.txt`。已有成功重试的历史失败不再触发检查。Conv/RMS实际绑定或安装问题解决后，才需要[选择性重试](ge-guide.md)，无需重复fresh。

若再次进入Docker，恢复conda/DDK后执行 `source targets/cpro/scripts/env.sh` 即可；配置含自动生成的HIAI头文件及三库路径，详见[环境恢复指南](environment-setup.md#docker-重进后的配置恢复)。只有运行产物再次丢失时才用 `--all-ops --fresh` 重建；保留权重无需重新准备。

## 当前状态

| 项目 | 状态 |
|---|---|
| PyTorch完整结构 | 全尺寸meta模型及层级FX图；不加载权重 |
| GE完整结构 | target/draft完整节点、连边、C++和recipe；融合可选 |
| 自定义算子探针 | 用户回传新批次 `fused-all-f6b81ead7ae8`：五类9例已重建尝试；GDN/CacheUpdate/Attention六例导出、编译、加载及执行通过，conv/RMS三例编译失败 |
| 真实权重AIR导出 | 旧双图支持完整FP16；新GE target支持显式缓存 |
| 状态执行 | 主机fake prefill→decode接续通过；真实板端待验证 |
| 尚未完成 | W4A8映射、常驻缓存与native DFlash调度、完整模型设备精度和性能 |

## 已确认与剩余问题

| 项目 | 结论 |
|---|---|
| GDN两例 | 新批次两例调用完成、数值诊断已生成；旧报告的decode核心输出全零、prefill核心输出relL2约0.999，尚未证实已解决；未通过精度验收 |
| 编译目标 | 始终为 `kirinx90`；`Ascend310B1` 是DDK初始化后返回的内部名称，`Ascend310P3` 是原工程平台 |
| GDN状态 | 安装版初态/末态为FP16；本次runner修正了末态只接受FP32的旧校验 |
| conv | 新批次两例仍编译失败：宽度4已通过初始维推，y2为 `[1,8192,1,S+3]`，随后格式配置失败；不能据此断言DDK完全没有kernel |
| RMS | 编译时仍找不到 `AdnRmsNorm` 维推函数；新增调用摘录表明它用于ND，原Cpro用内置 `npu_rms_norm`，实际GE映射待核对；本仓库可选 `rms` 融合绑定仍依赖Adn |
| CacheUpdate | 新批次prefill/decode两例导出、编译、加载和执行通过，数值诊断已生成；无数值golden，不代表精度通过 |
| Attention | 新批次prefill/decode两例导出、编译、加载和执行均通过，先前导出器问题不再阻止调用；数值仅为DIAGNOSTIC_COMPLETE，尚未验收 |
| 尚未验证 | Conv/RMS三例调用、全部用例的正确数值及NPU执行位置；完整模型与连续状态验证另行进行 |

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
