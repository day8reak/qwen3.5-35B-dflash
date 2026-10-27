# Qwen3.5-35B-A3B · DFlash / CPro X90

将 Qwen3.5-MoE 目标模型与 DFlash 草稿模型接到 CPro/X90。支持 **PyTorch 建模导出**和**直接 GE 构图**两条路径，部署使用 `AIR → OMG → OMC → HIAI`。

当前先复用原工程的完整AIR，以用户提供的 **40层、64/1动态档位** 契约编译target，再接6层DFlash。执行命令见[当前更新](current-status.md)。自建PyTorch/GE路径和量化缓存准备器保留作为备选；旧tiny仅用于回归。完整模型的编译回传、状态执行、设备精度和最终加速尚未验收。

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
