# 当前阶段测试命令

目标为C16／cap192／K15常驻预热后平均≤50ms。最新完整Draft的model inference为103.720ms，算子合计73.656ms，算术差额30.064ms；差额尚未归因。完整Draft仍为NUMERICAL_FAILED，host embedding准备未计入API，原数值门槛保持。

## 当前进展：按所选DDK注册测试Top1

用户回传的`tools/tools_omg/master/lib64/config/npu_generated_opinfo.json`包含NPU注册：ArgMaxExt2的FP16输入、INT32输出及dimension属性，ReduceMax的axes／keep_dims属性，以及ArgMin。对应arg_max_d.py、reduce_max_d.py源码存在。记录不证明当前形状会选择NPU实现。

上一轮ArgMaxV2编译为CPUCL1；两个ArgMaxWithValue候选均在infershape函数查找失败，未执行设备测试。新候选使用此次实际发现的接口：

| case | 表达 |
|---|---|
| top1_ext2_fp16 | FP16 [1,15,248320]直接ArgMaxExt2，dimension=2 |
| top1_ext2_flat_fp16 | 展平为[15,248320]后ArgMaxExt2，dimension=1 |
| top1_ext2_tiled_fp16 | [15,485,512]局部ArgMaxExt2＋ReduceMax，再归约485个值，GatherV2D＋INT32乘加恢复索引 |

旧ArgMaxV2也可能映射到同一ArgMaxExt2实现，直接版本作为对照，不预设能改变落点。分块版全词表保留、无padding；ReduceMax按实际CPro注册使用单输入和axes属性。仍需检查NaN、并列最大值及跨块索引的设备行为，即便部分归约落NPU也不等于验证通过。

## 编译三个新小图

同步源码后，沿用已有CANN_ROOT、CPRO_DDK_ROOT、CPRO_DDK_ID及工作目录环境执行。`--replace`复用已完成的Top1探针测试槽；旧结果摘要已记录，完整Draft及其profile不受影响。

```bash
python3 "$REPO_DIR/targets/cpro/tools/probe_draft_vector.py" \
  --cases top1_ext2_fp16 top1_ext2_flat_fp16 top1_ext2_tiled_fp16 \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$AI_RUN_DIR/draft/tests/top1-probes" --replace
```

先检查输出JSON中的编译结果、graphop_counts和precision_boundaries。存在有意义的NPU归约路径后，再进行精确INT32输出验证及同步API计时；编译失败的候选不执行设备测试。原AddCustom路径继续暂停。

此次只读报告的PARTIAL来自源码候选／摘录限额；KIRIN_CONF_JSON_PATH为空、部分公共布局目录缺失，不足以认定内部DDK不可用。检查器现排除池化、NMS、字节码文件，优先注册opFile实现与参数检查正文，不需要为开始这轮编译再重复检查。

主机run `20261104T022117Z-1605-76c12b`：230项图回放／发射／模拟绑定和严格比较检查通过，其中新旧六种Top1共54次完整形状回放；另22项只读检查器测试通过。旧三种图manifest及fixture字节不变。当前三种新图的所选DDK编译、NPU落点、实机数值和时延均待验证。详细证据见[结果](results.md)。
