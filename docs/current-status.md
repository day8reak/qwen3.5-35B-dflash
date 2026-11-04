# 当前阶段测试命令

目标为C16／cap192／K15常驻预热后平均≤50ms。最新完整Draft的model inference为103.720ms，算子合计73.656ms，算术差额30.064ms；差额尚未归因。完整Draft仍为NUMERICAL_FAILED，host embedding准备未计入API，原数值门槛保持。

## 当前进展：修复ArgMaxExt2缺失输出类型属性

用户回传三个原生候选均失败于`ArgMaxExt2Verify: AttrName output_type not exis.`，尚未产生可判断CPU／NPU落点的OMC。上一版探针按NPU opinfo构图，遗漏了OMG IR校验要求的属性；输出描述INT32不能替代output_type。

现给全部ArgMaxExt2节点补充整数属性`output_type=3`（现用GE枚举的INT32），包括分块版的局部及获胜块归约。发射前检查属性与输出描述一致，并优先展示缺属性的具体错误。数值、shape和fixture保持。所选OMG是否接受此补充仍待重编确认。

分块版日志已显示local_value／ReduceMax的infershape success，但winning_tile的input shape打印为空；当前只记录该现象，不能断言已正确推导[15,485]或已落到NPU。越过output_type校验后继续检查实际shape与落点。

用户回传的`tools/tools_omg/master/lib64/config/npu_generated_opinfo.json`包含NPU注册：ArgMaxExt2的FP16输入、INT32输出及dimension属性，ReduceMax的axes／keep_dims属性，以及ArgMin。对应arg_max_d.py、reduce_max_d.py源码存在。记录不证明当前形状会选择NPU实现。

上一轮ArgMaxV2编译为CPUCL1；两个ArgMaxWithValue候选均在infershape函数查找失败，未执行设备测试。新候选使用此次实际发现的接口：

| case | 表达 |
|---|---|
| top1_ext2_fp16 | FP16 [1,15,248320]直接ArgMaxExt2，dimension=2 |
| top1_ext2_flat_fp16 | 展平为[15,248320]后ArgMaxExt2，dimension=1 |
| top1_ext2_tiled_fp16 | [15,485,512]局部ArgMaxExt2＋ReduceMax，再归约485个值，GatherV2D＋INT32乘加恢复索引 |

旧ArgMaxV2也可能映射到同一ArgMaxExt2实现，直接版本作为对照，不预设能改变落点。分块版全词表保留、无padding；ReduceMax按实际CPro注册使用单输入和axes属性。仍需检查NaN、并列最大值及跨块索引的设备行为，即便部分归约落NPU也不等于验证通过。

## 重编修复后的三个小图

同步源码后，沿用已有CANN_ROOT、CPRO_DDK_ROOT、CPRO_DDK_ID及工作目录环境执行。`--replace`复用已完成的Top1探针测试槽；旧结果摘要已记录，完整Draft及其profile不受影响。

```bash
python3 "$REPO_DIR/targets/cpro/tools/probe_draft_vector.py" \
  --cases top1_ext2_fp16 top1_ext2_flat_fp16 top1_ext2_tiled_fp16 \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$AI_RUN_DIR/draft/tests/top1-probes" --replace
```

先检查输出JSON中的编译结果、graphop_counts和precision_boundaries。存在有意义的NPU归约路径后，再进行精确INT32输出验证及同步API计时；编译失败的候选不执行设备测试。原AddCustom路径继续暂停。

此次只读报告的PARTIAL来自源码候选／摘录限额；KIRIN_CONF_JSON_PATH为空、部分公共布局目录缺失，不足以认定内部DDK不可用。检查器现排除池化、NMS、字节码文件，优先注册opFile实现与参数检查正文，不需要为开始这轮编译再重复检查。

主机run `20261104T023316Z-2165-71799e`：233项图回放／发射／模拟绑定和严格比较检查通过，其中新旧六种Top1共54次完整形状回放；另32项修改范围、缺属性拦截和日志优先级检查通过。前一版本的原生图现会在导出前被拒绝；全部fixture字节保持。修复后的所选DDK编译、NPU落点、实机数值和时延均待验证。详细证据见[结果](results.md)。
