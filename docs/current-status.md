# 当前阶段测试命令

目标为C16／cap192／K15常驻预热后平均≤50ms。完整Draft仍为103.720ms／NUMERICAL_FAILED，host embedding准备未计入API。

## 21项编译矩阵已完成

用户回传21／21编译通过，自动落点检查全部INSPECTED，设备尚未执行。

| 已测表达 | 编译观察 |
|---|---|
| ArgMax [1,512]、[15,485]、[15,512]、[485,512] | 各1个NPUCL、无CPU操作 |
| ArgMax [7275,512]、[7680,485]、[15,485,512]、[512,7275]沿axis0 | 各1个CPUCL，明确含ArgMaxExt2 |
| 9种ReduceMax | 全部CPUCL，明确含ReduceMax |
| 4个完整词表组合 | 均NPUCL2＋CPUCL2，局部ArgMax和ReduceMax仍在CPU |

因此，展平、转置和交换485／512因子均未移除全量CPU归约；回退也发生在独立图，不能只归因于组合分段。存在规模相关的选择差异，但这些点不能给出精确行数／元素数门槛，更不证明硬件普遍不支持ReduceMax。NPU内核未解码，也没有新时延结果。

## 复用现有OMC，一次验证语义和API时延

这次不编译新模型。选择4个小ArgMax、1个大规模CPU ArgMax对照和4个完整词表组合，共9个已有OMC。小ArgMax各跑原fixture及13种新输入：全相等、首／末唯一最大、非零位置并列、全负数、正负零并列、正Inf并列、全负Inf、首／末／多处／全NaN、正子正规数。其余5个控制沿用原fixture，共61个输入case。先执行9个原fixture控制，再执行52个新输入。

每个输入case独立Load一次、预热3次、保留10次同步API调用及输出；所有INT32结果逐字节比较。NaN参考为第一个NaN，否则取第一个最大值；不改golden门槛。报告按9个原模型汇总场景覆盖、失败场景、实际／期望索引与每例API时延；时延不能相加或直接外推完整Draft收益。

沿用[环境配置](environment-setup.md)中的REPO_DIR、AI_RUN_DIR、CPRO_DDK_ROOT和HDC配置，运行配置为现有`runtime/runtime.json`。

先重建共享native运行器：旧探针解析器限制单维≤65536，会拒绝248320词表。新版本允许长轴，保留16 MiB合计输入输出预算，plan仍为V1；Python在上传前拒绝旧运行器。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$AI_RUN_DIR/runtime/native" --replace

python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --top1-semantics \
  --compiled "$AI_RUN_DIR/draft/tests/top1-probes" \
  --config "$AI_RUN_DIR/runtime/runtime.json" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$AI_RUN_DIR/draft/tests/top1-validation" --apply
```

启动后回收同一次执行：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --recover "$AI_RUN_DIR/draft/tests/top1-validation" \
  --output "$AI_RUN_DIR/draft/tests/top1-validation-result"
```

若回收显示仍在运行，稍后只重复recover。SDK执行失败会按原规则停止后续case并回收已完成输出；数值不一致会完整报告，未执行场景不能算通过。新输入在独立验证目录生成，编译目录中的OMC、原fixture和报告保持。

先确认小ArgMax的语义，再决定是否尝试限规模分支及按索引取值来避开ReduceMax。当前没有可提升为完整Draft的全NPU Top1，也不重启暂停的AddCustom路径。

主机315项检查通过，52组新输入与PyTorch交叉一致，610次模拟输出通过原比较／回收入口；真实C++计划解析器接受全部61份计划并拒绝6类非法／超预算输入。主机模拟不替代设备结果，详见[结果](results.md)。
