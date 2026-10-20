# CPro/X90 目标计划与未完成门槛

当前要求：HarmonyOS HIAI，PyTorch → AIR → OMG → OMC；权重、激活可量化，其他浮点计算 FP16。蓝区 6.1.1.0 是开发基线，不能替代内部 DDK 兼容性验证。当前量化为候选，无质量或性能晋升。

## 图边界（ABI v2）

| 图 | 输入顺序 | 输出顺序 |
|---|---|---|
| target | INT32 `[1,S]` IDs；INT32 `[1,S]` mask | INT32 `[1,S]` Top1；FP16 `[1,S,16384]` features |
| draft | FP16 `[1,S,16384]` features；INT32 IDs；INT32 mask | INT32 `[1,15]` Top1 |

ND、batch 1、右侧 padding 的 0/1 mask。`len(prompt)+max_new_tokens-1 <= S`；每个 S 独立产物。普通模式跳过草稿执行。HIAI 持久张量以实际 dtype 计算字节数，FP16 直接复制，不在 host 解码成 float。公开兼容 API 不暴露 dtype；manifest 和 OMC SHA256 负责锁定，运行时检查 dims/format/bytes。旧 ABI v1 产物被主机命令拒绝。

## 分类、验证和缺口

| 区域 | 当前实现 | 分类/剩余门槛 |
|---|---|---|
| 草稿与调度 | 复用 4B 代码，锁定 35B 草稿配置/权重；拒绝、修正、bonus、EOS | 算法参考；4B 权重维度不兼容，不能复用 |
| 目标 decoder | upstream text MoE/GDN；显式导出层循环 | 小模型参考通过；全权重待验证 |
| 非量化 FP16 | 图级改写主动 FP32 运算，再捕获并审计 dtype | 用户明确要求的数值改变；FP16 与原始 BF16/FP32 质量比较待做 |
| MoE | 所有专家静态枚举，动态 top-8 mask | 结构映射不冻结样例路由；性能差，不宣称稀疏加速 |
| DOPT | 所有融合专家暴露为 Linear；group128 INT4 + 16-bit 激活校准 | 近似候选；全权重校准/独立评测待做 |
| 激活量化数值 | vendor 输入量化器用 FP32 运算，输出回 FP16 | 避免本次实际复现的 FP16 缩放溢出；身份记录此策略，板端仍需比较 |
| 量化参考 | 主机桥接真实 Cython quantizer，普通区域为 FP16 | 进程内数值参考，不是 AIR 或自定义 NPU kernel |
| 压缩 AIR | 调用 DOPT get_air_model；检查原生量化 MatMul 数量和打包字节 | 缺配套 NpuQLinear / npu_quant_matmul_v4 / TorchAir；真实导出未完成 |
| OMG | framework=1，隔离短路径，payload 哈希、失败清理 | 真实负例已进入 AIR parser；没有有效 AIR 转换成功证据 |
| HIAI | 公共真实头文件 ARM64 编译；主机模拟生命周期/故障 | 内部库链接、模型兼容、板端执行待完成 |

压缩检查使用每个预定矩阵的原生量化节点和 INT4 字节预算。若内部 DDK 使用另一种 op/schema 或带 padding 的布局，先核实后做明确适配；不自动接受回退成密集 FP16 的“量化图”。

## 容量与性能

Meta 架构统计 target text 参数 `34,660,610,688`；目标候选量化参数 `33,617,346,560`。草稿连同复制的 embedding/head 共 `1,403,024,896` 参数。估算双模型权重、FP16 scale/offset 为 `22,751,793,408` 字节（21.19 GiB），未计 LUT、布局、激活、缓存、加载副本、HIAI workspace。18G free 不足；假设可用 32GiB 也必须实测峰值。内存报告只来自用户，不能把整机32G当成进程可用32G，Hugetlb 不是额外 RAM。

当前重算前缀、计算全部专家、两个 OMC 各持有 embedding/head；尚无增量 KV/GDN commit/rollback，也没有速度目标达成证据。后续应由完整板端基线识别热点，再开展有边界的稀疏/缓存优化。

`hardware_reference_policy=use`；没有匹配 X90 内部 DDK/固件/功耗的实测画像，没有能力/工作量哈希或依赖延迟账本。未委派性能算子，没有 roofline、理论余量或 DFlash 加速比结论。

## 后续门槛

用户明确板机不可访问，因此不连接其他设备冒充验证。板端入口是 `qwen35_dflash_hiai --probe`；device gate 保持 `UNAVAILABLE_BY_USER_ACCESS_CONSTRAINT` / pending。

先用匹配导出环境跑 tiny AIR→OMC，再进行完整量化参考与转换。完整链路应包括 tokenizer、target/draft、HIAI、解码；量化后的普通/DFlash token 与停止原因零差异，且与同源同精度的主机参考比较。原始模型与量化模型的质量另设 held-out 门槛，尚未冻结。性能需最终 artifact 在同一 X90 上各十次稳定、正确的测量。对象编译、CPU 和 fake HIAI 不能推进这些完整模型门槛。
