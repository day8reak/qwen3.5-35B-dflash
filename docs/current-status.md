# 当前阶段测试命令

目标为C16／cap192／K15常驻预热后平均≤50ms。同步API、输入准备／提交、加载与初始catchup分别记录。原数值门槛和golden保持。

## 当前进展：103.720ms，先检查全词表Top1落点

用户已回传host-embedding候选model.csv及完整op表：inference为103.720ms，算子耗时合计73.656ms，算术差额30.064ms；上一轮为184.568／78.986／105.582ms。差额减少75.518ms，但不是已归因的等待或拷贝时间。当前最大项为末尾MatMul 19.960ms、CPU ArgMax 12.353ms、Conv 8.643ms及两次特征Transpose 5.615ms。CPU小算子合计约0.785ms。加载160.281ms、卸载4.859ms另计。

本轮与先前OMC `967e9b640915893890ef95425b0e1279c8245bff0cfc2833f0cbfa56eb023372` 的关联来自连续会话；粘贴表没有模型哈希或回收receipt。CPU／NPU start time不能据此直接拼接关键路径。现有accuracy仍为NUMERICAL_FAILED：Top1 82/86与83/86，hidden／KV原门槛未通过；未写KV保持和重复调用稳定。host embedding准备不包含在native API中，生产resident尚未切换新ABI。

## 编译三个无模型权重的小图

更新本次源码，沿用已有REPO_DIR、AI_RUN_DIR、CPRO_DDK_ROOT、CPRO_DDK_ID和CANN_ROOT。CANN_ROOT为此前GE导出使用的主机CANN目录，不是手机DDK。当前只需要在服务器编译这三个小图；复用现有大模型与profile，不重新跑accuracy或profile。

```bash
python3 "$REPO_DIR/targets/cpro/tools/probe_draft_vector.py" \
  --cases top1_argmax_fp16 top1_with_value_fp16 top1_tiled_fp16 \
  --ge-root "$CANN_ROOT" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$AI_RUN_DIR/draft/tests/top1-probes"
```

三者都输入FP16 `[1,15,248320]`、只输出INT32 `[1,15]`，没有训练权重。依次检查ArgMaxV2、ArgMaxWithValue和485×512两级ArgMaxWithValue＋INT32索引合成。保持完整词表，不用近似TopK或FP16编码全局索引；不会运行已暂停的AddCustom探针。

回传命令末尾JSON，其中包含每个case的编译状态、CPUCL／NPUCL数量和边界dtype。COMPILED仅说明编译成功；仍落CPU的方案不视为达到目标，混合图需查看对应placement.json。若失败，JSON会附已有编译错误摘要，不启动设备。

主机已有27次完整形状回放及128项检查，覆盖跨块并列、65535／65536、末尾token、signed zero、±Inf和NaN；设备选择规则仍未验证。placement合适后再用现有validate_draft_vector.py执行三次预热／十次测量，所有INT32结果要求逐字节一致。探针通过也不能代替完整Draft数值及50ms验收；只有接入后同模型端到端测量才能判断最终收益。历史数据见[结果](results.md)。
