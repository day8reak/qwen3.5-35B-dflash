# 当前进展

目标为C16／cap192／K15常驻预热后平均≤50ms。完整Draft仍为103.720ms／NUMERICAL_FAILED，host embedding准备未计入API。

当前操作：用户要求最后再试一次内置实现，失败后转独立自定义Top1。新增`--top1-bitcarrier`固定三项，源码已完成主机验证；所选DDK编译和设备执行均未运行。

```bash
git pull --ff-only origin main
python3 "$REPO_DIR/targets/cpro/tools/probe_draft_vector.py" \
  --top1-bitcarrier --ddk-root "$CPRO_DDK_ROOT" \
  --output "$AI_RUN_DIR/draft/tests/top1-last-builtin" --replace
```

该方案把INT32 token的原始字节拆成两个FP16存储字，只做Select／结构搬运，末尾Bitcast恢复INT32；不把token数值转成FP16，也不改变输出ABI。所用[固定版本GE Bitcast声明](https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/python/torchair/_ge_concrete_graph/auto_generated_ge_raw_ops.py#L68735)有`type`与`keep_dim`属性；形状按[两字合并的Bitcast语义](https://www.tensorflow.org/api_docs/python/tf/bitcast)构造。两份公开接口资料均不证明所选CPro DDK支持，实际编译是第一道检查。

| 顺序 | 用例 | 检查范围 |
|---|---|---|
| 1 | `top1_bits_payload` | 0–248319每个ID在Select左右分支各选两次；覆盖全部65536种低16位、广播、拼接与Bitcast |
| 2 | `top1_bits_tree_offset65500` | 15×485奇数尾项归约，输出带65500偏移，穿过65535／65536边界 |
| 3 | `top1_bits_tree_full` | 完整FP16[1,15,248320]→INT32[1,15]，有限logits、并列选最小索引 |

前置用例编译失败、落CPU或存在未归属任务的主图计算时，自动跳过依赖项并保留诊断，结束这条内置路线。三项都通过图层面纯NPU检查后，才进入同批三项设备验证；验证器重新读取哈希绑定的落点，不接受只挑最后一项。每项预热3次、测量10次，所有INT32输出逐字节一致。位片段可能呈现FP16 NaN／Inf／subnormal，但不是logits；编译器转换或Select未原样搬运都会破坏索引，不能豁免此类失败。

主机已通过全词表位表示检查、19种宽度57次树回放、直接PyTorch ArgMax交叉检查、30次伪设备快照精确比较及损坏拒绝；旧38个探针的图、C++和fixture不变。尚无新NPU落点、设备数值或收益证据。若最后路线不成立或没有实际收益，按用户授权转固定边界的自定义Top1，不再扩展内置形状矩阵。整模型接入仍需单独验证。

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

该评估阶段只记录证据，自定义实现／编译／设备执行尚未开始，暂停的残差AddCustom路径未恢复。用户后续已授权“最后再试一次内置，失败后写Top1算子”，当前步骤见页首。详细记录见[结果](results.md)。
