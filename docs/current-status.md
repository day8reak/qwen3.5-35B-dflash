# 当前阶段测试命令

目标为C16／cap192／K15常驻同步API预热后平均≤50ms，加载与初始catchup另计。用户已接受原组合版最差hidden相对L2 2.61%、Top1 7/172差异作为性能基线；原NUMERICAL_FAILED和golden保留。

最新 `packed-spatial-4d` 已完成用户侧精度回收和profile。accuracy API均值189.147～200.515ms，较上一版同类均值下降28.3%～36.5%；精度差异摘要保持。单次profile推理184.568ms、算子汇总78.986ms，两者差105.582ms；加载234.684ms另计。目标尚未达到，混合accuracy用例与单次profile不代替固定输入预热基准。

按用户决定，先排查105.582ms差额，暂缓Top1改写。现有native计时仅包住Process，输入准备、输出落盘及KV提交不在其中。最新OMC仍有3个CPU／2个NPU子图：中间CPU子图接收第一个NPU子图的BOOL[1,208]，还接收两个经Cast送来的FP32[1,16,1]。其上游须从完整主图追踪；不把差额直接认作传输或等待。

## 分析已有边界，无需重新运行模型

在已更新代码的服务器上，沿用现有REPO_DIR和AI_RUN_DIR。输入是此前 `--inspect-boundaries` 写出的**完整JSON文件**，包含main_graphs；终端打印的摘要不足以追踪Cast上游。这个步骤只读取JSON，不依赖DDK、不重新加载OMC、不接触设备。

```bash
python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --analyze-boundaries "$AI_RUN_DIR/draft/tests/pytorch-optimized-boundaries.json" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-dependencies.json"
```

回传命令输出即可。其中npu_to_cpu保留每条连接的生产子图、端口、Cast路径和消费端；intermediate_cpu_outputs展示中间CPU子图送往后续NPU的输出。输出路径按同类已完成报告复用，原边界文件保持。

若status=PARTIAL，保留unresolved条目，不根据节点名字补猜连接。结果只证明显式图依赖，不能直接给出拷贝量、等待时间或子图内部每个输出的依赖。现有profile中ArgMax时间早于末尾MatMul，不能用两个设备的start time直接计算空闲区间。

下一项模型实验优先尝试前移由标量决定的mask／position计算，消除中间NPU→CPU→NPU依赖。先核对完整路径，再选择最小改动；确认编译分区、完整数值和KV保持后，以相同输入、频率、3次预热＋10次测量比较同步API。当前没有新模型需要导出、编译或上板。详见[结果](results.md)。
