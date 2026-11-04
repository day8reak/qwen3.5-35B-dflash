# 当前阶段测试命令

目标为C16／cap192／K15常驻预热后平均≤50ms。继续记录同步API，并单列输入准备／提交开销，不能把移到Process之外的工作当作消失。加载与初始catchup另计。已接受的性能基线仍保留原NUMERICAL_FAILED、golden和门槛。

`packed-spatial-rope-table` 已由用户完成编译产物的只读检查：OMC为2808047355字节，SHA256 `a045f7629920fd962d21c8ce694b1ee694a3be61942c4cac6a556f00cc631027`。CPUCL3／NPUCL2变为CPUCL2／NPUCL1，中间CPU两个Mul和回流依赖消失，`intermediate_cpu_partitions=[]`。当前结构是前置CPU控制→一个NPU主体→末尾CPU ArgMax；仅剩logits这条NPU→CPU路径。NPU分区20输入／14输出是内部边界，不是公共输入数量。

该模型随后在MODEL_LOAD失败，尚未进入推理。执行cc7399ff087d40be8e647bcdd2f46a7c已回收，锁已释放。日志候选报告单个NPU分区权重2806299136字节（2.614 GiB），随后memcpy_s失败并终止AllocateWeightMemory。公开Secure C默认单次destMax上限为0x7fffffff；大权重块触及拷贝上限是当前优先假设，实际设备库和本次拷贝参数仍未核实。见[结果](results.md)。

用户随后确认：显式--limit-partition-size true仍生成2806299136字节的单NPU权重块，被打包前检查拒绝。所选OMG帮助说明true本来就是默认值，不能继续靠这个开关实现2GiB拆分。

## 现在执行：固定外置权重格式后重新导出

新增显式候选packed-spatial-host-embedding，沿用相位表和原FP16权重。调用前复制[anchor, MASK×K]对应行，以float16[1,K+1,2048]的block_embeddings替换anchor ID输入；仍17输入／14输出。完整embedding表不再作为NPU常量，LM head、六层计算和KV规则保留。K15输入块65536字节，embedding原表1017118720字节；旧NPU块减表约1789180416字节，仅为估计，须查看新编译结果。

用户本次导出停在air-validation：export.air为1790366439字节，weight/文件数0。去掉embedding后，TorchAir自动选择了内嵌Const；本项目需要小AIR＋带原参数名的FileConstant来做编译快照与权重绑定。已固定本进程内的存储格式选择，沿用安装版本的原生FileConstant生成和原字节写盘，结束或异常后恢复，不修改共享安装。64MiB元数据限制保留。已有内嵌AIR不能仅凭arg节点名可靠恢复原参数文件名，本次需重新导出，不能用--finalize-air绕过。

更新代码后，沿用已有REPO_DIR、AI_RUN_DIR、CPRO_DDK_ROOT、CPRO_DDK_ID。固定测试槽位上一任务已完成，可用--replace；原prepared权重、公共可运行模型及失败结果保留。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/export_incremental_draft.py" \
  --prepared "$AI_RUN_DIR/draft/source/prepared" --frontend pytorch \
  --optimization packed-spatial-host-embedding \
  --context-rows 16 --cache-capacity 192 --max-draft-tokens 15 \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-air" --replace
```

先确认status=PYTORCH_DRAFT_AIR_EXPORTED、weight_storage.policy=native-torchair-fileconstant-v1、weight_storage.restored=true、weight_files>0。weight-export.json记录安装版本源码哈希、原生格式选择、各文件名与尺寸；不支持的导出器接口会停止并保留错误，不静默回到内嵌格式。

导出通过后再编译：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/compile_incremental_draft.py" \
  --air-export "$AI_RUN_DIR/draft/tests/pytorch-optimized-air" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --tbe-diagnosis "$AI_RUN_DIR/common/tbe-diagnosis" \
  --limit-partition-size true \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-omc" --replace
```

true保留以启用本项目打包前的权重大小检查，不再把它当作新增拆分策略。所有GraphOp权重大小须可读且≤2147483647字节，否则在partition-weight-check停止，不发布可执行候选。此阈值是诊断条件，不是设备拷贝参数的实测结果。

编译成功后顺序执行只读检查：

```bash
python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --inspect-boundaries "$AI_RUN_DIR/draft/tests/pytorch-optimized-omc" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-boundaries.json"

python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --analyze-boundaries "$AI_RUN_DIR/draft/tests/pytorch-optimized-boundaries.json" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-dependencies.json"
```

本阶段回传编译与依赖摘要，核对每个权重块及intermediate_cpu_partitions为空。之后重建共享native runner以支持accuracy plan v2，再执行原isolated／feedback比较；旧runner只支持v1，不能直接用于新输入。新ABI目前仅接入独立accuracy候选，生产resident仍使用原token-ID接口。

新accuracy在服务器准备原FP16 embedding输入文件，独立参考仍按token ID查表。API统计不含该查表、文件读取和输入提交，报告会明确标注；不能用它单独证明端侧完整调用≤50ms。后续端侧接入需复用本地embedding资源、缓存MASK行，并测输入准备＋提交＋同步API。

主机六层缩小尺寸220次回放全部14输出逐字节相同，八档完整尺寸meta图不含embedding参数，完整词表head保持。所选DDK编译、新OMC加载、设备数值和时延均未执行。上一已测4d为profile184.568ms、算子汇总78.986ms，105.582ms差额仍未归因。
