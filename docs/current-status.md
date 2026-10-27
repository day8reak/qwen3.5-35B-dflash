# 当前更新与执行命令

[首页](README.md) · [环境配置](environment-setup.md) · [验证记录](validation.md)

更新日期：2026-10-27。用户明确时间紧迫：先推进完整40层target的普通prefill/decode，再接6层DFlash；最终目标不变。

## 用户现在执行什么

**三填充诊断已完成，不需要再续跑或重跑这批探针。** 用户回传 `gdn-diagnostic-138b60d3eba7` 为 `DIAGNOSIS_COMPLETE`：原 `gdn-diagnostic-c0cbac5ac9cf` 的零填充结果复核完成，另两轮执行完成，三轮host输入回读SHA256一致。保留这两个诊断目录和原 `fused-all-f6b81ead7ae8` 批次，后续直接分析已有证据。

同一OMC/输入的GDN输出随预填充值变化。零填充时prefill/decode核心relative L2约为0.00392/0.0564，末态约0.146/0.155；A5填充时核心升至3.44/11.19；5A填充时出现约203.25的误差，与FP16位模式 `0x5A5A` 的值一致。有效token区域已排除padding。当前证据不能区分部分写、输出累加和缓存可见性；清零不是已验证修复，数值验收仍未通过。上述结果来自用户报告，原始远端文件尚未在本地重新核验。

整模工作与该问题并行推进，下一里程碑是**带真实量化权重的target OMC编译与一次普通prefill/decode**。当前没有可诚实宣称“一键完整上板”的命令：直接GE权重路径只支持FP16，W4A8尚未接入；约34.66B文本参数的FP16数据约69.3GB，不能作为32GB板端方案。`--expert-dispatch selected`减少参与计算的专家，不减少常驻权重。现有stateful图的 `past_length` 静态，单步native执行已实现，但可复用decode图/常驻状态及对应部署入口仍待接通。整模编译准备不以所有可选探针通过为前提；实际整模执行必须具备这些资源与接口条件。

当前优先级：接通量化权重与算子映射、整模编译入口和target状态执行；并行处理阻塞运行的GDN读写问题。暂不增加Conv/RMS候选、重复平台诊断或新的性能矩阵。需要实机验证时给单一入口命令。

模型范围仍保留GDN、CacheUpdate、Attention三类六例，Conv/RMS使用基础算子。只有确需重新执行全部六例时才使用 `run_fused_probes.py --model-ops`；旧九例 `--all-ops` 保留为可选诊断，原Conv/RMS失败不改为通过。

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
| GDN两例 | 三填充诊断已完成，输出依赖预填充值；零填充改善但末态relL2仍约0.146/0.155，未通过精度验收，原因未定位 |
| 编译目标 | 始终为 `kirinx90`；`Ascend310B1` 是DDK初始化后返回的内部名称，`Ascend310P3` 是原工程平台 |
| GDN状态 | 安装版初态/末态为FP16；本次runner修正了末态只接受FP32的旧校验 |
| conv | 当前走基础算子的历史拼接/截取路径；原Cpro自定义Conv两例在rank-4 y2维推后格式配置失败，历史结论不变 |
| RMS | 当前走基础算子RMS路径；原ND用的 `AdnRmsNorm` 探针缺维推函数，保留失败；不再尝试其他融合RMS候选 |
| CacheUpdate | prefill/decode调用通过，比较仅为SHAPE_ONLY_UNVERIFIED_CONTRACT，无数值精度通过结论 |
| Attention | prefill/decode调用通过，比较仅为SHAPE_ONLY_UNVERIFIED_CONTRACT；先前导出器问题不再阻止调用，精度仍未验证 |
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
