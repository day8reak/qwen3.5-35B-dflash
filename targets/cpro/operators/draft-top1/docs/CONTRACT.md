# Contract

候选名 Qwen35DraftTop1。仅替代有限 FP16 logits 的完整词表 ArgMax。独立验证后才讨论整模型接入；不改 LM head、词表、Draft KV、输出 ABI 或原 NaN 失败记录。

| 项目 | 固定值 |
|---|---|
| 输入 | little-endian contiguous ND float16 [1,15,248320]，7,449,600 bytes |
| 输出 | little-endian contiguous ND int32 [1,15]，60 bytes |
| 数学定义 | y[r] = min {i : x[r,i] = max_j x[r,j]} |
| 输入域 | 所有有限 binary16，包括 ±0、subnormal、±65504 |
| 非有限输入 | 不在候选合同内；fixture/host 参考拒绝。未宣称设备图带有限性守卫 |
| 精度 | 不降精度，不裁剪，不采样，不对值添加索引扰动 |
| shape/batch | 仅固定尺寸；错误 shape、dtype、字节数拒绝 |
| 权重 | 无 |
| 验收 | 每行 INT32 逐字节相同，10/10实测输出相同 |

+0 与 -0 相等，选较早索引。全局索引范围0–248319，不能经单个 FP16 数值转换。主机 C++ 参考读取 FP16 位模式为有序整数键，与直接 NumPy ArgMax、FP32精确扩展后的PyTorch ArgMax独立交叉检查；它不是设备内核。

候选设备逻辑为 UB 分块的值/索引联合归约，块偏移以 INT32 合并。实际块尺寸、core 数、局部索引存储格式、同步和尾块必须由所选 DDK 核对后固定。相等保留较小全局索引。按行调度是候选，尚未实现或性能验证。

输出只有60字节，禁止越界对齐写回；不能假设runtime额外分配64字节。不同核的32字节DMA/64字节缓存行可能重叠。需验证精确尾写API或采用有独立workspace且单writer的输出方案，不能无保护并发写相邻INT32。输入也不允许超范围尾读。

公开接口线索：[X90 ReduceMax](https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/reduction_compute/ReduceMax.html)、[标量访存与cacheline](https://asc.gitcode.com/guide/technical_appendix/concepts_and_terms/memory_access/scalar_read_write.html)。必须以选定DDK内容和设备验证确认可用性。
