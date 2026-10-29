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
