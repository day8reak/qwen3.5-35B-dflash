# Draft direct 输出的只读定位

最新用户实机结果已执行完整六层 draft：编译精度检查 153/153，16 个输出 dtype 均保留，HIAI Process 返回成功。Top1 仍与参考不一致，数值未通过。

direct 观察仍存在三个零值归一化矛盾。下列端口的读回字节全零，输出对象、指针和容量在 Process 前后未变化，三个输入回读哈希匹配：

| 端口 | 语义 | 主图生产者 |
| --- | --- | --- |
| 2 | context_fc | SubGraph_1:3 |
| 6 | layer_0_qk_fp16 | SubGraph_5:0 |
| 8 | layer_0_probability_fp32 | SubGraph_6:0 |
| 11 | layer_0_post_attention | SubGraph_7:2 |
| 14 | layer_0_output / layer_1_input | SubGraph_9:2 |

同一 SubGraph_7 和 SubGraph_9 的其他输出有非零结果。填充值为零时，全零字节既可能没有被写回，也可能被写成零；上述记录不能区分两者，不能据此确定 CPU 算术、Cast、内存复用或缓存问题。

直接读取已经保留的编译和执行文件：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/inspect_draft_outputs.py" --math
```

默认任务目录为 `$AI_RUN_DIR/full-draft/direct-trace`。也可以用 `--job` 指定该目录。`--full` 增加原始有界元数据。命令打印报告，不写已有文件，不调用 OMG/HDC，不读取投影权重或多 GB 打包模型，也不重新运行 CPU 模型。

报告包含主 NetOutput 与 GraphOp 的选定端口描述符、主图消费者、明确归属的 CPU GraphDef 路径，以及原始源节点在 CPU 图中的出现位置和消费者。schema 来自编译记录并核对 SHA；不解码不透明 NPU task，不根据相似名称推断原始值或优化行为。Cast/view 路径是连接记录，不证明数值等价。

`--math` 额外验证已有 board/native 传输凭据，读取已取回的输出和一个 layer1 input-norm FP16 权重。它比较由 layer0 输入、attention 和 MLP down 重建的残差与观察值，并列出完整残差、仅 MLP down、仅 post-attention 三种 layer1 norm 输入假设的误差。概率行和也单独报告。缺少 visible mask 时，QK 到 masked-score 的关系明确不可检查。

所有结果保持诊断性质，没有新增数值通过阈值，没有重写 golden、权重或模型算术。原 direct 输出本身若不可靠，重建关系也可能不可靠；任一假设更接近观察不能单独证明运行时根因。

目前 153/153 检查仅证明锁定的 FP32 CPU 节点、描述符与规定边关系。边界 Cast 被记录但不作为 required arithmetic 检查，不能把该结果当作所有 FP16 输入来源、舍入边界或最终输出快照正确的证明。

## 已取回的数学结果与残差候选

用户已运行上述只读检查。`layer_1_input_norm` 与同一板端 `layer_0_mlp_down` 经原 layer1 gamma/RMS 计算的结果 **32768 个元素逐字节完全一致**；使用完整两次残差和重建时 relative L2 为 `0.45476676183344134`。这是当前 direct 图有效数据路径缺少残差项的强证据，仍不能据此确定原生产图的首个故障位置，因为两图的终端结果已不同。

两处残差输出有 `output_tensor=true`、`reuse_input_index=2`。该序号不是独立的启用标志：[官方接口](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/API/basicdataapi/atlasopapi_07_00455.html)与[官方 schema](https://gitee.com/ascend/metadef/blob/cba623c83840fd0ac1e365a6fb80a34b20bd6a2d/proto/ge_ir.proto)另有 `reuse_input` 布尔值。描述符的32字节尾部也符合 [GE 内部内存大小计算](https://gitee.com/ascend/metadef/blob/cba623c83840fd0ac1e365a6fb80a34b20bd6a2d/graph/utils/tensor_utils.cc)，不能单凭这个大小差改宿主分配或 OMC 字段。

针对残差路径新增独立、可恢复的候选：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_draft_residual.py" \
  --tbe-diagnosis "$AI_RUN_DIR/fused-all-f6b81ead7ae8/diagnosis"
```

它复用已验证的 prepared 权重、固定输入及两路原 golden；不重新执行 CPU 模型。新 AIR、编译、bundle 和板端结果位于 `full-draft/residual-fix`，原生产、packed 和 direct 文件均保留。候选仅导出原 Top1、hidden 两路，避免再次加入中间观察端口。

仅12处残差 Add 的 GE 发射采用 `FP16 -> FP32` 的两个输入 Cast、FP32 Add、再 Cast 回原 FP16 结果；源图名称、权重、参考与比较判据不变。IEEE RN-even 下有限 FP16 操作数的加法保持相同结果，包括负零及溢出；NaN payload 等价不作断言。附近指数的和可由 FP32 精确表示，距离更远时较小操作数不足以改变 FP16 舍入；单元验证覆盖所有63488个有限值对12个边界锚点、40万随机对及独立 Torch/FP64 参考。

六层 recipe 增加36节点，FP32 CPU要求由153增至165。新导出契约含 `-residual-cpu-fp32-add-v1`，`draft/residual-lowering.json` 记录12处选择。新加法必须全部通过实际 CPUCL 检查，才允许 bundle 和执行；编译或执行失败后重试复用相应已完成阶段。主机等价检查及静态精度检查不是实机数值通过，最终以同输入的 Top1 和 hidden 结果为准，完整语言质量、接受率、状态及速度仍未关闭。
