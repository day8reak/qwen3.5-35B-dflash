# 当前阶段测试命令

目标为C16／cap192／K15常驻预热后平均≤50ms。同步API、输入准备／提交、加载与初始catchup分别记录。原数值门槛和golden保持；旧NUMERICAL_FAILED基线不能证明新模型精度通过。

## 当前模型：已编译，继续设备验证

用户已完成packed-spatial-host-embedding候选的OMC边界及依赖检查。当前模型1790918688字节，SHA256 `967e9b640915893890ef95425b0e1279c8245bff0cfc2833f0cbfa56eb023372`；CPUCL2／NPUCL1，intermediate_cpu_partitions=[]、unresolved=[]。结构为前置CPU控制→单个NPU主体→CPU ArgMax，已经没有中间CPU回流。唯一NPU→CPU路径是完整logits经Cast交给ArgMax。边界逻辑尺寸不是实测搬运量，旧105.582ms模型／算子差额也不能直接当作已消除。

新OMC约1.668GiB，上一单NPU模型约2.614GiB。旧模型在权重拷贝／MODEL_LOAD失败，新模型能否加载及收益尚待本次验证。现有OMC可直接复用，不需要重新导出、编译或重复边界检查。详细事实和历史见[结果](results.md)。

新公共输入用float16[1,K+1,2048]的block_embeddings替换anchor ID，K15为65536字节，仍17输入／14输出。需要支持accuracy plan v2的runner；生产resident仍使用原token-ID接口，本次测试不切换生产配置。

## 1. 重建共享runner

在现有服务器环境中沿用REPO_DIR、AI_RUN_DIR、CPRO_DDK_ROOT、CPRO_DDK_ID，以及原OHOS_NATIVE_SDK、HIAI_INCLUDE_DIR、HIAI_LIBRARIES或QWEN35_RESOURCES配置。运行配置仍为runtime/runtime.json，默认共享运行器目录为runtime/native。旧accuracy失败已回收、无持有锁；下面复用原固定测试槽位，脚本仍保护未完成任务。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$AI_RUN_DIR/runtime/native" --replace
```

这里只编译ARM64运行器，不编译OMC。成功后runtime/native/result.json应为LINKED，并含qwen35_draft_accuracy的构建记录。若已经用当前代码重建过支持plan v2的runner，可直接进入下一步。

## 2. 验证加载、数值、KV与同步API

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --config "$AI_RUN_DIR/runtime/runtime.json" \
  --compiled "$AI_RUN_DIR/draft/tests/pytorch-optimized-omc" \
  --source "$AI_RUN_DIR/draft/source/prepared" \
  --air-export "$AI_RUN_DIR/draft/tests/pytorch-optimized-air" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-run" --apply
```

看到DRAFT_ACCURACY_STARTED后回收同一次执行，不再次启动模型：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --recover "$AI_RUN_DIR/draft/tests/pytorch-optimized-run" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-result"
```

若返回PENDING，稍后重复同一recover命令。回传完整摘要，核对device_execution、first_failure、case_counts、relative_l2_ranges、feedback_repeats和native_api_latency。EXECUTED_OUTPUTS_RETRIEVED才证明该OMC加载且执行成功；PASSED／NUMERICAL_FAILED另行判断数值。未写KV应保持、重复调用应稳定，偏差须与已接受旧基线对照，不能放宽门槛使其通过。若加载失败，保留自动回收的native.stderr和HiLog诊断，先处理失败。

新fixture在服务器复制原FP16 [anchor, MASK×K]行，独立参考仍从token ID查原表。输入准备、文件读取和提交不在native API计时内；不能据此单独宣称端侧完整调用≤50ms。

## 3. 执行成功后采集同模型profile

仅在上一步已回收完整输出、结果为PASSED或NUMERICAL_FAILED后执行。允许数值失败仅用于性能诊断，原失败标记保留。

```bash
python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --accuracy "$AI_RUN_DIR/draft/tests/pytorch-optimized-result" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-profile-run" \
  --allow-numerical-failure --apply
```

看到DRAFT_OPERATOR_PROFILE_STARTED后回收原执行：

```bash
python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --recover "$AI_RUN_DIR/draft/tests/pytorch-optimized-profile-run" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-profile-result"
```

回传这次model.csv与op.csv。用新OMC的同步API和同轮模型／算子数据评估收益，再决定ArgMax或输入准备的后续优化；不将旧模型数据拼成新模型时延。
