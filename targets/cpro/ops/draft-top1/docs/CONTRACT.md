# StableArgMax contract

用户确认本次先做通用 Top1，TopK 留独立接口扩展。输入为连续 ND FP16 `x[..., V]`，沿最后一维归约，输出 INT32 `indices[...]`。静态 rank 1–8、各维正数、总元素数 ≤ 2147483647；rank1 输出为标量。每个编译模型的 shape 固定，不宣称一个 OMC 动态切换形状。物理 storage shape 必须与 origin shape 逐维相同。

数学定义：`indices[r] = min {i | x[r,i] = max_j x[r,j]}`。所有有限 binary16 均在输入域内，包括 ±0、subnormal、±65504。正负零相等。NaN/Inf、空维度、其他 dtype/布局/轴、TopK 不在当前合同内。主机参考拒绝非有限值；kernel 不扫描或返回非有限错误码，调用方须保证输入域。

Qwen 验收配置为 `[1,15,248320] → [1,15]`：输入7,449,600字节，输出60字节。不会裁剪词表、改变 LM head、隐式降精度、添加索引扰动或修改原 NaN 失败记录。真实 Draft logits 的有限性仍需在整模型接入时验证。

kernel 每块最多8192个FP16，`ReduceMax(calIndex=true)` 联合得到值和局部索引。局部索引从第二个half槽位按 uint16 位模式读取，绝不数值转换成half索引；加上INT32范围内的块偏移后合并，相等保留小索引。NPU scalar流水执行合并，不涉及主机CPU回退。

X90配置为BlockDim=1。UB预算32864字节：输入16384、独立scratch16384、归约结果32、输出64。host读取所选平台UB容量并检查；不假定另一芯片容量。所有GM读写使用DataCopyPad按有效字节搬运，ReduceMax仅归约实际count。每16行写一次索引，最后一次仅写剩余索引字节；Qwen为单次60字节写回。无GM临时workspace。

验收是每个输出INT32逐字节相同，10次真实执行全部通过并稳定。主机shim只验证控制流、范围和索引规则，不证明硬件的ReduceMax tie/subnormal语义、DMA或同步。310P需独立配置与设备验证，当前AddConfig仅kirinx90。
