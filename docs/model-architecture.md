# 模型结构

模型由 40 层 Qwen3.5-35B-A3B Target 和 6 层 DFlash Draft 组成。Draft 根据 Target 的中间特征生成候选 token，Target 验证候选并提交结果。

## Target 和 Draft 交互图

- [Target 40 层交互图](assets/target-architecture-interactive.html)：查看 Prefill／Decode、层间连接、GDN／全注意力、MoE 和状态流。
- [Draft 6 层交互图](assets/draft-architecture-interactive.html)：查看共享 context、逐层注意力与 MLP、残差连接、独立 KV 缓存及候选输出。

下载 HTML 后用浏览器打开即可离线交互，支持展开细节和导出 SVG；GitHub 文件预览不执行 HTML。

## Target

40 层按「3 层 Gated DeltaNet＋1 层全注意力」重复排列，每层包含 MoE。每层依次执行归一化、注意力与残差，再执行归一化、MoE 与残差；输出直接传给下一层，最后经过最终归一化和词表输出头。

Prefill 处理输入序列，Decode 逐 token 生成。GDN 层维护卷积状态与递归状态，全注意力层维护 K/V 缓存；各层状态在后续调用中反馈给同一层。

## Draft

从 Target 的第 1、6、11、16、22、27、32、37 层（从 0 起计）提取特征，拼接并投影成六层共享的 context。候选块经过六层注意力和 SwiGLU MLP，hidden 逐层传递，每层维护独立的 K/V 缓存。前五层使用滑窗因果注意力，最后一层允许块内双向注意力。

当前归档版本每次最多接收 16 行新增 context，缓存容量为 192 行。候选块由 1 个 anchor 和 15 个 MASK 组成，经过完整词表输出头产生 15 个候选，实际接受数量由 Target 验证决定。缓存只提交已确认的 context，候选块 K/V 仅供本次计算。

## 核心参数

| 项目 | Target | Draft |
|---|---|---|
| 层数 | 40：30 层 GDN＋10 层全注意力 | 6 |
| Hidden／词表大小 | 2048／248320 | 2048／248320 |
| 注意力头 | 全注意力 Q/KV：16/2，head_dim 256；GDN Q/K：16、V：32，head_dim 128 | Q/KV：32/8，head_dim 128 |
| FFN | 256 个路由专家，Top-8，另有共享专家 | SwiGLU，中间维度 6144 |
| 状态 | 30 份卷积状态＋30 份递归状态＋20 份 K/V | 6 层各一对 K/V，共 12 份 |

配置见 [Target](../specs/target-config.json) 和 [Draft](../specs/draft-config.json)。运行方式见[完整模型执行](run-end-to-end.md)，精度与性能结论见[当前结果](results.md)。
