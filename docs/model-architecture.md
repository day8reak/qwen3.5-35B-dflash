# 模型结构

当前部署使用Qwen3.5-35B-A3B文本主干与独立DFlash草稿模型，不包含视觉编码器或原checkpoint的MTP路径。

## Target

| 项目 | 配置 |
|---|---|
| 层数 | 40：30层Gated DeltaNet＋10层全注意力 |
| 层顺序 | 每4层为3层GDN＋1层全注意力 |
| Hidden / 输出词表 | 2048 / 248320 |
| MoE | 每层256个路由专家，Top-8，另有共享专家 |
| 全注意力 | Q/KV头16/2，head_dim=256 |
| GDN | Q/K头16、V头32，head_dim=128，卷积宽4 |
| 当前产物 | 原量化prefill64和decode1 OMC，外置权重 |
| 状态反馈 | 80份：30卷积状态＋30递归状态＋20 K/V缓存 |

每层执行：RMSNorm → GDN或全注意力 → 残差 → RMSNorm → MoE → 残差。40层之后经过最终Norm与词表输出头。

## Draft

| 项目 | 配置 |
|---|---|
| 模型 | 完整训练六层DFlash，内嵌权重OMC |
| 注意力 | 前5层窗口注意力，末层全注意力 |
| Hidden / MLP中间维度 | 2048 / 6144 |
| Q/KV头 | 32/8，head_dim=128 |
| Target特征层（从0开始） | 1、6、11、16、22、27、32、37 |
| 输入特征 | 每token拼接8层输出，宽16384，再投影到2048 |
| 一次生成 | block_size=16：1个anchor＋15个候选 |
| 当前容量 | 静态S64；模板后的prompt＋生成预算≤64 |

当前测试每轮最多验证前2个候选；生成15个候选不等于接受15个。Target决定最终token，三类状态只沿已提交的token推进。Prefill/decode使用HIAI V2，Draft使用HIAI V1。

参数来源：`specs/target-config.json`、`specs/draft-config.json`；部署继续复用原量化权重，未缩减层数或词表。
