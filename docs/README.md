# Qwen3.5-35B-A3B · DFlash / CPro X90

将 Qwen3.5-MoE 目标模型与 DFlash 草稿模型接到 CPro/X90。支持 **PyTorch 建模导出**和**直接 GE 构图**两条路径，部署使用 `AIR → OMG → OMC → HIAI`。

当前按用户要求先接通两套完整模型的真实端到端流程：短提示词由原decode1从零状态逐token读入，复用原S64 draft，多轮串行verify，不重编译模型。六层 `residual-fix` 已在固定fixture通过15/15 Top1，端到端直接复用其bundle；原40层量化target采用[原PyTorch与直接GE双路线](target-parallel-tracks.md)并行，接真实8层features、64→1状态与串行verify。最新命令统一见[当前更新与执行命令](current-status.md#用户现在执行什么)。真实特征多轮精度、语言质量和最终加速仍待验收。

| 从这里开始 | 内容 |
|---|---|
| [模型结构](model-architecture.md) | 混合注意力、MoE、DFlash 与三类缓存 |
| [当前更新与待执行命令](current-status.md) | 已完成什么、现在需要运行什么 |
| [环境配置](environment-setup.md#docker-重进后的配置恢复) | 按顺序source已有环境、补七项export、保存配置、执行全新算子探针 |
| [PyTorch 执行](pytorch-guide.md) | 参考推理、TorchAir 导出、独立融合模型 |
| [GE 构图执行](ge-guide.md) | 准备权重、基础双图、可选融合与状态图 |
| [板端执行与对比](board-guide.md) | 编译 runner、上传 OMC、运行、回收与比较 |
| [项目结构与端到端流程](project-structure.md) | 主要代码文件的职责及数据如何流转 |

补充资料：[权重下载](weights.md) · [验证范围](validation.md) · [算子接口原文](qwen35-moe-operators-and-quantization-analysis.md)。已有结论的探针和旧排障过程已归档到仓库外，本仓库保留生产实现与必要回归。
