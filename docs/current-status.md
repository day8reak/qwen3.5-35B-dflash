# 当前进展

目标为C16／cap192／K15常驻预热后平均≤50ms。完整Draft仍为103.720ms／NUMERICAL_FAILED，host embedding准备未计入API。

21项Top1编译矩阵、9个OMC的61例设备语义／时延验证均已完成，无需重复启动或回收这批任务。用户报告610次测量调用全部成功，输出重复稳定；原完整比较标准下45例通过、16例失败。

| 已测表达 | 设备结果 |
|---|---|
| 4个小NPU ArgMax | 各11／14通过；仅first_nan、last_nan、multiple_nan失败；非NaN场景全部通过 |
| CPU ArgMax [7275,512] | 原有限fixture通过，API均值7.4633ms；这是局部归约，不是最终完整词表Top1 |
| tiled／flat／485宽完整组合 | 仅NaN两行失败，API均值分别19.8443／19.5646／19.6726ms；全量局部ArgMax与ReduceMax仍在CPU |
| transposed完整组合 | 同样两行失败，API均值40.027ms；全量局部归约仍在CPU |

NaN由探针主动注入，不能据此推断真实Draft logits有NaN。正常推理预期logits有限；本地通用PyTorch Top1在归约前拒绝非有限hidden／logits，但这不是当前设备增量图已执行该检查的证据。原报告的完整NaN合同失败保留；小NPU原语仍可作为已测非NaN输入的候选，不能仅因NaN差异否定其正常数据用途。已有数据也不足以保证所有真实输入永远有限。

上述时延为预热3次后10次同步API墙钟时间，非纯NPU时间，不能相加或直接扣减完整Draft时延。完整组合的NaN丢失发生在哪一段尚未隔离。当前没有已验证的完整词表纯NPU Top1，也没有改变完整Draft。

## 纯NPU替代方案调研

2026-11-04按用户要求检索官方资料，优先考虑能同时保留最大值和索引的方案，避开当前CPU ReduceMax：

1. **分块TopK(k=1)**：每块同时输出value和index，再对块最大值做TopK(1)，选择对应局部索引并以INT32合成全局索引。比如248320=485×512，数学上覆盖全部词表。官方[Ascend C TopK](https://asc.gitcode.com/api/SIMD-API/adv_api/sort_operations/TopK.html)明确列出X90支持half输入与INT32索引接口；这不是当前CPro GE TopK前端可用或自动选到NPU的证据，且k=1也可能走排序路径。需先核对所选DDK的前端、后端注册及索引选择／合成支持。
2. **比较／选择树**：相邻候选两两比较，值与INT32索引一起选择，相等取左。按连续词表顺序分组可保留最小全局索引，248320项最多18级；有限输入不改变logits数值。它可避开ArgMax／ReduceMax，但大量Slice／Compare／Select以及INT32处理的实际落点和融合开销未验证。[Where语义](https://onnx.ai/onnx/operators/onnx__Where.html)仅说明条件选择，不证明X90支持。
3. **Ascend C值／索引联合归约**：官方[ReduceMax](https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/reduction_compute/ReduceMax.html)列出X90 half／float、可同时返回值和索引、并列选最小索引。按UB容量分块，half局部索引不超过65535，再以INT32合并。它是内核API，区别于本次落CPU的GE ReduceMax；接入涉及自定义算子，暂停的AddCustom路径本轮未恢复。

MaxPoolWithArgmax作为低优先级线索：[MindSpore文档](https://www.mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.MaxPoolWithArgmaxV2.html)中Ascend索引为uint16，不能直接当作248320词表的INT32全局索引；文档也没有验证当前X90／CPro组合。

下一步先审查TopK与比较选择树所需算子的当前DDK支持，形成一次完整候选批次后再上机；本页不安排新一轮同类形状／NaN探测。数学等价、官方API支持、选定DDK编译落点和实测收益分别记录。详细证据见[结果](results.md)。
