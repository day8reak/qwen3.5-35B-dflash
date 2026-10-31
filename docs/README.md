# Qwen3.5-35B-A3B · DFlash / CPro X90

将 Qwen3.5-MoE 目标模型与 DFlash 草稿模型接到 CPro/X90。支持 **PyTorch 建模导出**和**直接 GE 构图**两条路径，部署使用 `AIR → OMG → OMC → HIAI`。

当前主线复用原40层target prefill64／decode1与完整六层draft S64，不重编模型。原prefill驱动的板端P8/G8闭环已经执行通过，精度与加速未验收。用户已允许混合接口推进常驻：V2 prefill一次后关闭，随后V2 decode与V1 draft各加载一次、在板端交替Run／Process。新混合常驻完整请求已由用户实机跑通：prefill1／decode8／draft7，各resident模型加载1次，总墙钟88.680秒。下一步定位与旧版第2个token起的差异；精度、重复稳定性和独立权重释放仍待验收。当前命令见[执行入口](current-status.md)，总阶段及天数见[交接计划](dflash-delivery-plan.md)。

| 从这里开始 | 内容 |
|---|---|
| [模型结构](model-architecture.md) | 混合注意力、MoE、DFlash 与三类缓存 |
| [当前更新与待执行命令](current-status.md) | 已完成什么、现在需要运行什么 |
| [DFlash执行总计划与交接](dflash-delivery-plan.md) | 从当前闭环到精度／长上下文／批量verify／2倍性能／交付，含天数、失败分支和后续助手提示词 |
| [机器可读执行状态](../specs/dflash-execution-state.json) | 当前阶段、依赖、通过条件、证据与唯一下一动作 |
| [环境配置](environment-setup.md#docker-重进后的配置恢复) | 按顺序source已有环境、补七项export、保存配置、执行全新算子探针 |
| [PyTorch 执行](pytorch-guide.md) | 参考推理、TorchAir 导出、独立融合模型 |
| [GE 构图执行](ge-guide.md) | 准备权重、基础双图、可选融合与状态图 |
| [板端执行与对比](board-guide.md) | 编译 runner、上传 OMC、运行、回收与比较 |
| [项目结构与端到端流程](project-structure.md) | 主要代码文件的职责及数据如何流转 |

补充资料：[权重下载](weights.md) · [验证范围](validation.md) · [算子接口原文](qwen35-moe-operators-and-quantization-analysis.md)。已有结论的探针和旧排障过程已归档到仓库外，本仓库保留生产实现与必要回归。
