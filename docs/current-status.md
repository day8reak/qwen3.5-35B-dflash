# 当前阶段测试命令

目标为C16／cap192／K15常驻预热后平均≤50ms。最新完整Draft的model inference为103.720ms，完整Draft仍为NUMERICAL_FAILED，host embedding准备未计入API。

## 当前进展：核对ArgMaxExt2前端IR输入合同

补output_type后的三个原生探针仍然编译失败。现在OMG明确报`Input size 1 is wrong, should be 2`，随后报`Input[1] DataType FLOAT is wrong`。第二输入不存在，后者不能作为FP16 logits不受支持的证据。错误暴露了探针构图问题：NPU后端opinfo的一输入／dimension属性定义不能直接充当OMG前端IR原型。此前主机回放只验证构图的数学意图，没有验证厂商前端接口。

当前保留这些失败表达供对照，导出前拒绝单输入ArgMaxExt2；CLI在替换测试槽之前拦截已知无效的三个原生case，避免清除本轮失败图。没有猜第二输入名称、轴常量形状或ReduceMax前端定义。分块日志的local_value infershape success后，winning_tile输入shape仍为空，需要一起检查。

## 只读已有DDK、失败图和成功编译的CPU节点

同步main后执行这一条。无需CANN编译、重编模型或上板，也不用替换测试槽。标准输出分列frontend_sources、cases及reference三类证据。

```bash
python3 "$REPO_DIR/targets/cpro/tools/probe_draft_vector.py" \
  --inspect-top1-ir "$AI_RUN_DIR/draft/tests/top1-probes" \
  --reference-compiled "$AI_RUN_DIR/draft/tests/pytorch-optimized-omc" \
  --ddk-root "$CPRO_DDK_ROOT"
```

检查器按所选DDK读取通用头文件／前端源码中的ArgMaxExt2、ReduceMax、GatherV2D声明；按保存的编译环境核对DDK路径，解码已有失败IR。另从完整Draft的CPU子图提取已编译ArgMaxExt2及其直接输入生产者，读取前核对packing的schema、分区范围及CPU图owner。完整OMC仅流式计算哈希，权重分区不会载入内存；不执行厂商代码。

头文件文本不等于实际加载的注册，CPU节点是已编译示例而非完整前端schema或NPU支持证明；失败图shape可能不完整。常量数据保留字节数／哈希，不猜轴值。目录／文件／输出限额、缺失声明、schema不可读或DDK不匹配会显式报告；PARTIAL不表示平台不支持。先根据这些现有证据修正端口、轴和归约shape，再选择下一次小图编译。

主机run `20261104T024333Z-2592-afd370`：69项检查通过，使用合成ELF/schema、IR、OMC和头文件验证读取、边界／来源检查、只读保证、失败重试保留证据及CLI。六种旧图与fixture字节保持，三个旧GE表达仍可发射；这些测试不代表所选DDK或设备验证通过。完整Draft与暂停的AddCustom路径保持。详细记录见[结果](results.md)。
