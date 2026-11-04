# 当前阶段测试命令

目标为C16／cap192／K15常驻预热后平均≤50ms。最新完整Draft的model inference为103.720ms，完整Draft仍为NUMERICAL_FAILED，host embedding准备未计入API。

## 当前进展：按所选HIAI前端定义修正Top1探针

用户回传的math_defs.h明确规定ArgMaxExt2第二输入为INT32 axis，ReduceMax第二输入为INT32 axes。两者的数据端口均列出FLOAT而未列FLOAT16。完整Draft的CPU子图同时证实：ArgMaxV2已编译为ArgMaxExt2，接FP32数据和一个4字节INT32常量，属性为output_type=3、keep_dims=false、outmaxval=false、topk=1。常量的实际轴值未解码。

三个原生探针现在保留FP16公共输入，在图内扩展为FP32后接前端算子。ArgMaxExt2使用INT32标量轴，ReduceMax使用INT32向量[2]，分块值输出为FP32[15,485]，全局索引恢复仍用INT32。旧的一输入／轴属性表达会在导出前拒绝。直接版作为同一路径对照；展平、分块是否改变NPU落点，需要实际编译结果。

## 重新编译三个小图

同步main后执行以下命令，替换已完成的top1-probes测试槽。本轮只编译、检查OMC分区与边界描述，不上板。

```bash
python3 "$REPO_DIR/targets/cpro/tools/probe_draft_vector.py" \
  --cases top1_ext2_fp16 top1_ext2_flat_fp16 top1_ext2_tiled_fp16 \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$AI_RUN_DIR/draft/tests/top1-probes" --replace
```

查看每例的model_compilation、placement、graphop_counts及precision_boundaries。编译通过后仍需检查整个归约链的落点；设备比较必须保持所有15行、所有重复的INT32字节完全一致，包括并列、NaN和大索引。FP16扩展为FP32在主机上保持数值，不证明设备归约的NaN规则或内部指令精度。

主机run `20261104T030633Z-3506-c58c14`：243项检查含六图54次完整词表回放、C++发射、模拟厂商元数据下的真实ABI绑定与严格整数比较；另46项检查覆盖旧图／fixture保持、65536种FP16位模式扩展、错误前端合同拒绝及HIAI头文件摘录。所选DDK编译、设备数值和性能验证均NOT_RUN。完整Draft与暂停的AddCustom路径未修改，详细证据见[结果](results.md)。
