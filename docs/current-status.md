# 当前进展

目标为C16／cap192／K15常驻预热后平均≤50ms。完整Draft仍为103.720ms／NUMERICAL_FAILED，host embedding准备未计入API。

最后内置路线已结束：用户报告 `top1_bits_payload` 在OMG的NPU预构建阶段失败。选中的Bitcast配置是INT32→INT4，与请求的FP16入口不匹配；后两项树按依赖跳过，未编译。没有新的OMC或设备结果，不能据此推断所有X90 Bitcast均不可用。

已按用户要求交付[ops自定义Top1源码](../targets/cpro/ops/draft-top1/README.md)，采用通用`StableArgMax`：有限连续FP16，静态rank1–8沿最后维归约，INT32输出，同值取小索引。Qwen `[1,15,248320]` 为验收配置。已包含kernel、host tiling/注册、Torch schema/Meta和GE converter；TopK与310P执行适配暂不实现。

主机源码控制流/Torch导出63项、参考工具35项、工程准备11项、文本回传9项通过；真实pt2已导出。所选DDK编译、AIR/OMC、设备数值和性能仍待用户执行，不是已验证纯NPU实现。完整Draft未改变。

2026-11-05按两台服务器重新分工：DDK服务器只把三个host/kernel文件复制到现有allops，沿用其编译安装；单算子使用 `stable_arg_max_ge.py` 走GE C++→AIR→OMG/OMC，再用已有validate/recover工具上板，无需Torch/TorchAir。权重服务器保留Torch Meta/converter及TorchAir导出。默认一份OMC复用七组Qwen输入，逐例3次预热/10次精确比较。无框架流程27项及NumPy/PyTorch参考交叉检查19组通过；GE/OMG调用测试使用mock，真实所选DDK和设备仍待用户执行。具体命令见上述README。

后续用户OMG日志确认StableArgMax注册/dispatch和host tiling已生效（15/248320/8192，block_dim=1），worker在te_update_version初始化失败。新GE入口漏传已有TBE兼容参数，现已补 `--tbe-diagnosis`，复用此前的 `common/tbe-diagnosis`。新增 `--reuse-export <旧目录>` 核验并复用AIR，仅重做OMG，保持kirinx90和旧失败现场；host/kernel不变。41项主机流程检查通过，修正后的实际编译/设备结果仍待用户执行。

用户已启动修正后的OMG命令但终端无输出；代码确认新入口漏接既有progress回调。已补stderr阶段提示与30秒OMG等待观察，43项host检查通过；当前运行保留，另终端tail原compile.log，实际编译结果仍待回传。无需为进度显示重跑。

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

## 12项内置算子结果与自定义Top1评估

用户回传`2a42b15`候选批次：**10项编译成功、2项失败，纯NPU候选为0**。6个成功图仅CPUCL，4个是NPU／CPU混合。设备执行和数值仍为NOT_RUN；本批不安排重跑或设备计时，保留已有诊断。

| 已测表达 | 所选DDK结果 | 对下一步的意义 |
|---|---|---|
| 4种独立TopK及直接／分块／15分支完整组合 | TopKV2落为CPU `TopKD`；15分支的局部TopK也全在CPU | 本批的分块大小和拆分方式没有解决归约回退 |
| Select控制 | 浮点值／条件分支为NPU任务，INT32 `Select`在CPU | 仅替换首层Cast不能使现有整数索引树成为纯NPU |
| INT32 Gather和索引合成 | `GatherV2D`、`Mul`、`Add`均在CPU | 只把局部归约移到NPU仍不够 |
| 小树与完整词表树 | NPUCL编译`CastT tree_0_index`失败：`kernelId not registered: 196620` | 源码该节点为BOOL→INT32；失败图没有有效落点 |

日志里的裸`dataType 3`不按HIAI序列化枚举解读。上述是当前前端、形状、dtype和默认选择策略的证据，不代表X90硬件普遍不支持INT32。用户摘要未提供本批OMC哈希，未独立读取服务器产物。

**评估结论：为“纯NPU＋精确INT32输出”值得投入一个独立自定义Top1原型。**把FP16值归约、同值最小索引选择和INT32全局索引合并封装在同一个NPU算子内部，可避开已观察到的GE整数节点回退。官方X90 `ReduceMax(calIndex=true)`是算法基础；实际选定DDK的注册、代码生成和运行支持仍须验证。

候选边界固定为`FP16[1,15,248320] → INT32[1,15]`，完整词表、不改LM head。输入逻辑7,449,600字节，输出60字节；这不是已测传输量。按实际UB容量分块，half块内索引不超过65535，按API约定读取索引位表示，在核内用INT32加块偏移并比较合并。首版可按核数分配行、核内遍历连续块，减少跨核同步；是否需要二阶段并行由实测决定。

单个FP16无法精确承载全局索引：主机穷举0–248319，首个失真值是2049。两个FP16小整数能精确存储高低位，但最终纯NPU INT32重建尚无已证实路径；FP32虽能精确表示这些索引，当前编译链的FP16降精度已有记录，不能仅靠前端声明保证。

建议把后续工作限定为同一个Top1验证包：先验证所选DDK中的FP16输入／INT32输出、自定义NPU任务和实际Load／执行，再验证完整词表的并列、尾块、65535／65536、大索引及subnormal，最后与**等价的完整词表CPU Top1**配对计时并做整模型回归。原61例NaN合同保持独立记录。算法范围清楚，最大不确定性是自定义接入；此前AddCustom JIT的`te_update_version()`失败说明这部分仍有风险。不能将小NPU原语0.4–0.8ms、局部CPU对照7.4633ms或混合图19–40ms直接当成新算子的预期收益，也不能据此承诺完整Draft≤50ms。

上述为先前评估。最终内置失败已触发自定义Top1授权，当前主机合同与环境准备见页首；设备实现尚待所选DDK证据。详细记录见[结果](results.md)。
