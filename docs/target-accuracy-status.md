# Target 精度状态

更新：2026-11-12。最新回传 **146 项数值，51 项 relative-L2 >1%**，高危结论不变。Prefill／Decode 各 73 项；原状态为 142 METRICS_RECORDED、2 PASSED、2 FAILED。全部 stable=TRUE，六个 GDN raw state 的 dtype 不保持。ConvState 12 项仍没有数值。

[交互结构图](assets/target-architecture-interactive.html)保留 Prefill／Decode、两次层内残差、层间连接和逐边界指标；[最新 CSV](../specs/target-boundary-metrics-20261112.csv)合并旧表与本次补齐项，重叠项保留已归档的小数精度。[算子族风险 CSV](../specs/target-operator-accuracy-risk.csv)同步更新。CSV 的 relative_l2 是比值，乘 100 才是百分比。结构图可直接导出当前展开状态的 SVG；[旧静态 SVG](assets/target-architecture-accuracy.svg)保留为历史快照。

本次回传表未附 OMC／执行清单，尚未在本地主机核验本轮产物绑定。此前 90/158 编译成功、68 失败是全批重编前的记录，不再用它解释本次数值是否执行；需要对应 compile.json 和 fixture 才能关联具体产物。浮点项没有冻结验收容差，不改写为 PASSED。

## 与此前结论的区别

已回传的异常指标在本次显示精度下未见变化。新增四项均低于 1%，没有新增高危边界：

| 补齐项 | 阶段 | relative-L2 |
|---|---|---:|
| L0 InputNorm | Prefill | 0.1212968% |
| L3 Q Norm | Prefill | 0.1259383% |
| L0 Conv1d | Prefill | 0.0263842% |
| L0 Conv1d | Decode | 0.0291362% |

普通 Norm 现有 20/20 项、Conv1d 6/6 项数值；图中这四个原灰色节点已更新。146 项中，36 项数值零误差、59 项非零且不超过 1%、51 项超过 1%。`stable=TRUE` 仅表示重复一致。

### 高危算子重点定位表

更新于 2026-11-12，数值来自最新回传汇总的[逐边界指标](../specs/target-boundary-metrics-20261112.csv)。P0／P1 合计 **51 个 relative-L2 >1% 的边界输出**；另外单列 3 个 Decode 状态 dtype 问题和 12 个编译阻塞。数值行的数量是当前范围内的“问题输出数／已测输出数”；C0 的数量是“编译失败数／选中 case 数”，都不是整图内核数量。下表 relative-L2 **统一为百分比**，量化投影的原始比值范围是 483.04～11278.20。

| 优先级 | 算子／定位区域 | 异常范围与数量 | relative-L2 | 首先检查什么 | 对照与边界 |
|---|---|---|---|---|---|
| **P0** | **QuantBatchMatmulV4444 所在 QLinear 链路，含 DynamicQuant** | L0–L2 qkv/z/b/a/out，L3 q/k/v/o；Prefill **19/19**、Decode **19/19** | **48,304%～1,127,820%** | 先比 DynamicQuant 的 INT8 输出与 per-token scale；再用相同量化输入、packed W4、权重 scale 隔离 MatMul；核对 group256、转置、布局与槽位 | 38/38 边界异常；DynamicQuant 尚无独立数值，不单独认定其内核错误 |
| **P0** | **CacheUpdate** | L3 Decode，K/V **2/2** | **99.69%～99.81%** | 分别比较当前写入区和未写历史区；核对 block/offset、RefData/alias、缓存初始化与输出回读 | Prefill **2/2 PASSED**；问题集中在本批 Decode，不能推广为全部调用失败 |
| **P1** | **RoPE 组合子图** | L3 Q/K；Prefill **2/2**、Decode **2/2** | **17.56%～19.62%** | 核对 position、cos/sin、partial rotary 维度、rotate_half、head/序列布局和实际输入绑定 | 两阶段全部已测输出异常；Slice/Concat/Mul/Add 的具体责任待拆分 |
| **P1** | **ChunkGatedDeltaRule：core** | L0–L2 Prefill，**3/3** | **1.56%～7.12%** | 核对 Q/K 归一化、g/beta、初态、chunk/effective_length，逐 chunk 比较首个偏差 | Decode core **0.115%～0.175%**；当前严重偏差集中在 Prefill |
| **P1** | **ChunkGatedDeltaRule：raw state** | L0–L2 Prefill，**3/3**，dtype 全不保持 | **12.10%～44.60%** | 比较写回前 raw state；核对实际输出 dtype、state 布局和累加精度，分开检查后续 FP16 转换 | 不用最终 writeback 零误差替代 raw state 验证，也不把全部误差归因于 dtype |
| **P1** | **AdnFusedInferAttention** | L3 Prefill，**1/1** | **99.67%** | 核对 QKV/NZ 布局、mask/scale、有效长度、padding、block table，按 query 检查 | Decode **0.251%**；Prefill 目前只有一个已测输出，需控制变量复现 |
| **P2** | **ChunkGatedDeltaRule：raw state dtype** | L0–L2 Decode，dtype **3/3** 不保持 | **0.065%～0.429%** | 核对参考与 OMC 实际输出 dtype、Cast 位置，以及多步状态反馈的误差变化 | 数值未超过 1%；与 Prefill 合计 6/6 raw state dtype 不保持，不计入上述 51 项 |
| **C0** | **LinearAttentionConvstateUpdate** | L0–L2、两阶段、两个输出：**12 项无数值；此前小图编译失败** | **未执行** | 对照成功整图的 RefData、双输出/卷积消费者及 hybrid 路径，与小图 Data、单输出及 single 路径的差异 | 原 T64 整图编译成功；当前是小图构建阻塞，没有精度结论 |

所有数值项使用各自捕获的 golden 输入，不能统一归因于上游 QKV 误差传播。表内是**重点定位边界，不是已经定责的内核 bug**；原始 `METRICS_RECORDED`／`FAILED` 状态保持不变。MoE 专家 `GroupedMatmulV555`、Router、共享专家和 LM head 尚无本批精度证据，继续列为待覆盖，不纳入已观察异常数量。

## QLinear 输入与权重检查

2026-11-12：已提供[两类单模块候选测试命令](target-batch-rerun.md)：38 个 QLinear 显式 FP16 scale／per-token 适配，Cache K/V 两阶段各有浮点索引控制组和整数索引候选组，共 46 图。只完成主机逻辑检查，实机精度待回传；以上指标仍为修改前结果。旧 Cache 小图未包含位置计算，不能把其 Decode 误差直接归因于浮点索引。Const 没有 Cast 节点也不能单独证明编译后物理字节仍是 FP32。

先检查全部 38 个量化投影已有的 AIR、OMC、编译日志，不重新导出或编译。静态代码推导自洽不等于实际加载的内核与输入契约已经验证；本次优先排查 INT8 激活、packed W4、weight scale 和 per-token scale。

在 35 服务器现有 Python 环境执行，`OMC_BATCH` 指向**产生本次数值的实际 OMC 批次**；以下默认值沿用全批重编命令。检查目录须为空或不存在，保留既有输入和编译产物。

```bash
export REPO_DIR=/home/w00949577/qwen3.5-35B-dflash
export AI_RUN_DIR=/home/w00949577/qwen3.5-35B-dflash-run
export CPRO_DDK_ROOT=/home/l00886453/ddk_0806/image/hiai_ddk/ddk_internal
export OMC_BATCH="${OMC_BATCH:-$AI_RUN_DIR/target/tests/all-boundaries/omc-rebuild}"
export QLINEAR_CHECK="$AI_RUN_DIR/target/tests/qlinear-contracts"

python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" inspect-qlinear \
  --batch "$OMC_BATCH" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$QLINEAR_CHECK"

cat "$QLINEAR_CHECK/summary.txt"
```

如已知本轮上板准备目录，在同一次命令中加 `--fixture "$EXECUTION/fixture"`，同时核对该 fixture 的模型、槽位和输入字节是否属于当前批次。未提供时明确保持未检查，不用相同文件名或尺寸推断同一组数据。

| 产物 | 用途 |
|---|---|
| `ports.csv` | 按 case／阶段列出 AIR、保留 IR、OMC 中可见端口的 dtype、shape、format、Cast／TransData／Transpose／Reshape 链；编译日志完整参数另列 |
| `report.json` | 常量完整字节哈希与导出时原权重哈希对照、权重对应端口、512 字节前缀的按类型解码、编译器身份和解析限制 |
| `summary.txt` | 全部 QLinear 的输入类型、权重绑定结果及未解析项，便于回传 |

重点看：

1. `weight_port_bindings`：x2 是否接到匹配 `W_q` 的常量、scale 是否接到匹配 `scale` 的常量，以及中间是否有转换。`EXACT_SOURCE_WEIGHT_BYTES` 只说明转换前源权重逐字节相同；没有匹配不能自行归因，应查看常量是否可见、被折叠或发生了 Cast。
2. `SOURCE_AIR` 对照 `COMPILER_BUILD_ARGUMENTS`：实际构建参数是否把两路 scale 变成了 FP16；保留完整参数序号及 AIR 端口名，不能把 optional 输入压缩后的序号误当成同一端口。日志交错损坏会标 UNPARSED，不猜测缺失字段。
3. scale 方向是否与模型维度吻合，packed 权重是否沿途经过 TransData／Transpose。检查依据是已观察的描述与转换链，I4ZN 字节前缀仅作对照，不能靠“看起来像 int4”证明逻辑 K/N 排列正确。
4. 如 OMC 封装内部节点，`OPAQUE_OR_FUSED` 表示不可见；编译参数也不是设备 GM 内容。若这些检查无差异，下一步采集 DynamicQuant 的真实 INT8／per-token 输出，并以相同输入、权重与 scale 比较 MatMul，继续验证实际数值。

命令不改源码、权重、scale 或 dtype。输入数据只有捕获的外部输入及源常量可读；**没有把这次检查冒充真实 MatMul 四输入 dump，也不宣称已经找到根因**。局部文件缺失会给出 PARTIAL 并继续其余 case。

## 覆盖边界

各小图使用独立捕获的 310P golden 输入，不能把 GDN／Cache／RoPE 等异常统一归因于上游 QKV。writeback／residual 零误差不证明其上游正确，也不证明 RefData 别名或多步状态反馈正确。MoE 专家 `GroupedMatmulV555`、Router、共享专家、LM head 和完整 40 层仍待覆盖。

[全部 AIR／OMC／上板／比较命令](target-batch-rerun.md) · [ConvState 编译对照](target-compile-triage.md) · [整图 dump 对比](target-dump-compare.md)
