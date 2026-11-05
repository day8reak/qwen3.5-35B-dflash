# 当前阶段：优先优化 Draft 常用调用

2026-11-05：用户要求优先找性价比高的路线。按实际调度覆盖、已测热点、改动与验证成本排序。当前RoPE模型C16／cap192／K15，OMC SHA为 `b43ae9cfcd50999277829a21adc24099170d4a7c4856b70729385b59d362a828`。原数值结果保持FAILED，进一步精度排查按用户决定暂缓。[完整结果](results.md)。

## 当前 profiling

| 项目 | 耗时 ms | 源码对应关系 |
|---|---:|---|
| 同步模型 inference | 65.822 | 一次profiling调用，非预热后十次均值 |
| MatMul_30 | 19.807 | 按源码推断为完整词表head |
| Conv2D | 8.788 | 上下文特征投影 |
| Transpose_1／Transpose | 5.359／0.414 | 投影前的空间打包 |
| StableArgMax | 0.416 | 完整词表Top1 |

表格来自用户响应case18采集请求的粘贴内容，原始CSV／本次请求摘要未在本地绑定。NPU时间戳晚于API结束，不能据此推算关键路径、等待时间或净收益；算子duration合计61.837ms也不是关键路径。

## 第一优先：在输入准备时完成特征重排

当前入口把FP16 `[1,16,16384]` 通过两次转置打包为 `[1,1024,16,16]`，再做原来的4×4、stride4 Conv。每次仅512KiB，重排本身不涉及浮点算术。正常有新增上下文的Draft调用都经过这条路径。

拟保留Conv的权重、形状与计算边界，把空间打包移到端侧CPU准备输入的过程中。目标是免去图内两个转置，尽量合并到已有输入拷贝，避免新增一套独立的全量拷贝。

主机已逐字节核对现有 `IncrementalDraft.project_features` 的Conv输入：valid=0／1／2／14／16均一致；全部262144个元素的索引映射通过，覆盖全部16位存储模式。只证明重排等价，没有执行Conv或完整模型。本地主机CPU的NumPy拷贝计时不作为设备性能预测。

入口重排实验只生成两份投影前缀，每份约64MiB原权重，输出均为FP16 `[1,16,2048]`。使用已完成回放的case2／16／17（valid2／16／14），共享这两份模型，不复制完整Draft。源码图逐元素验证两次转置的索引映射，Conv属性、权重及输出布局保留。

服务器首轮baseline已编译成功（NPUCL=1），packed在OMG形状检查失败，尚未执行设备。定位到入口Data仍保留旧的三维 `input_desc`；上一版只改了 `output_desc`。现已同步修正两侧描述，并在源码前缀与编译器准备后的AIR各检查一次。30项新增主机检查通过，使用固定版本TorchAir的真实Data生成函数复现旧问题；DDK、OMC检查和进程归属边界使用桩，实际重编仍待服务器验证。

端侧CPU使用预分配512KiB缓冲，候选每次做字节重排后SetInput；基线每次直接SetInput。两者都预热3次、记录10次，分别记录打包、SetInput、同步API以及从打包开始到Execute结束的总墙钟。当前小实验仍有一次临时缓冲到输入的拷贝，其成本已计入，尚未实现直接写入最终输入缓冲。快照和模型加载不计入总墙钟。

在服务器更新代码，沿用已选DDK、现有目录和环境变量，保留baseline，只重编已知失败的packed：

```bash
cd "$REPO_DIR"
git pull --ff-only origin main
TOP1_DIR="$AI_RUN_DIR/draft/tests/top1-integration"
ROPE_DIR="$TOP1_DIR/rope-fp32"
PACK_DIR="$AI_RUN_DIR/draft/tests/feature-pack"

python3 "$REPO_DIR/targets/cpro/tools/probe_draft_feature_pack.py" \
  --air-export "$TOP1_DIR/export" \
  --result "$ROPE_DIR/kv-reproduction/result" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --tbe-diagnosis "$AI_RUN_DIR/common/tbe-diagnosis" \
  --output "$PACK_DIR/compiled" --retry-failed
```

`--retry-failed`核对原模型、源AIR、权重、输入、golden、DDK和成功前缀，确认无活动进程且OMG有非零退出码后，保存上次失败报告到 `previous-failure.json`，重置失败候选目录。baseline OMC和fixture复用；未知完成状态或文件身份变化会拒绝重置。首次运行、输出目录尚不存在时去掉这个参数。

需要保留原export的特征投影权重和已完成回放的fixture。编译结果应为COMPILED，两份模型均完成；失败保留诊断并停止，不继续启动设备。编译器副本只在确认打包完成后清理，保留两份小OMC和必要报告。不会编译新的完整Draft。

编译完成后，重建共享runner以支持端侧打包计时，再执行六组对照：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$AI_RUN_DIR/runtime/native" --replace

python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --compiled "$PACK_DIR/compiled" \
  --config "$AI_RUN_DIR/runtime/runtime.json" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$AI_RUN_DIR/draft/tests/feature-pack-run" --apply
```

启动确认后，使用同一执行目录回收；这个命令只回收，不重新执行模型：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --recover "$AI_RUN_DIR/draft/tests/feature-pack-run" \
  --output "$AI_RUN_DIR/draft/tests/feature-pack-result"
```

重点看结果中的 `feature_pack`：`coverage_complete=true`、`baseline_equivalence=PASSED` 表示三组输入十次输出均与设备基线逐字节一致且有限、稳定。各组 `mean_saved_ms`／`median_saved_ms` 比较包含CPU准备的总墙钟，正数代表候选较快；还需看波动范围。原CPU参考容差没有放宽，`numerical_validation=FAILED` 可能与“设备基线等价通过”同时出现，应分别解读。前缀收益足够且稳定后才进入完整Draft改图与验证；5.773ms是旧两算子的duration之和，不是已实现净收益。

## 后续排序

| 顺序 | 方向 | 判断标准与成本 |
|---|---|---|
| 2 | 8.788ms的特征投影 | 复用约64MiB投影权重做少量前缀对照；保持完整点积和原舍入边界，先确认选用的表达／布局是否更快 |
| 3 | 19.807ms完整词表head | 覆盖每轮proposal、耗时最大，但逻辑FP16权重约1.017GB；先查实际布局和实现，不能因热点大就直接重写自定义内核 |
| 暂缓 | 零新增KV的专用proposal | 当前调度把最后一批追加与proposal合并，正常多轮生成很少满足valid=0；独立快路径收益覆盖有限 |
| 暂缓 | 继续改StableArgMax | 已测仅0.416ms，优先级低 |

当前head输出logits约7.45MB，远小于完整权重；单独融合Top1减少中间写回，不能据此推断能解决19.8ms。暂不以缩词表、量化或减少层数作为默认路线。没有匹配的X90实测能力画像，不给出带宽极限或硬件无优化空间结论。

零新增KV原型已保存在本次本地实验目录，未发布、未编译、未接入生产，不占服务器运行盘。若以后采用KV-only catchup＋proposal拆分，必须比较两次调用总耗时、缓存交接和模型常驻成本；测试中6/19例为零追加不代表实际请求频率。现阶段无需执行该原型的编译命令，保留当前已完成profiling和基线OMC即可。
