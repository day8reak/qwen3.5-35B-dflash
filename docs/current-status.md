# 当前阶段测试命令

目标为C16／cap192／K15常驻预热后平均≤50ms。同步API、输入准备／提交、加载与初始catchup分别记录。原数值门槛和golden保持。

## 当前进展：设备执行完成，采集同模型profile

用户已回收host-embedding候选的38次accuracy调用，board_attempt为qwen35_draft_accuracy_2486123ea3e3411390d491ff6db887a0，状态EXECUTED_OUTPUTS_RETRIEVED、无持有锁。缺失host_embedding绑定已通过原表和模型来源校验恢复；当前候选加载和推理成功。先前边界报告的OMC为1790918688字节、SHA256 `967e9b640915893890ef95425b0e1279c8245bff0cfc2833f0cbfa56eb023372`，CPUCL2／NPUCL1、无中间CPU回流。本次粘贴的accuracy摘要没有模型digest，profile工具会从已回收fixture校验实际模型身份。

isolated catchup／proposal均值103.958／103.998ms，feedback为106.090／104.994ms。按11／8／11／8次调用加权为104.802ms，上一双NPU轮同类均值为193.772ms，减少45.9%。这是混合输入诊断API结果，不是固定输入预热验收，也不包含服务器embedding查表、输入文件读取和提交。

NUMERICAL_FAILED保留：Top1为82/86与83/86，错误位置、hidden／KV相对L2范围与上一轮摘要一致；未写KV保持、两组重复调用逐位稳定。摘要一致不证明全部输出与旧版逐字节相同。生产resident尚未切换新block_embeddings接口，50ms和数值通过均未达成。历史与详细数据见[结果](results.md)。

## 1. 启动一次诊断profile

沿用现有REPO_DIR、AI_RUN_DIR及板端环境，直接复用已回收的pytorch-optimized-result。无需重新导出、编译、重建runner或再跑accuracy。允许数值失败仅用于性能诊断，原失败状态仍保留。

```bash
python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --accuracy "$AI_RUN_DIR/draft/tests/pytorch-optimized-result" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-profile-run" \
  --allow-numerical-failure --apply
```

## 2. 回收同一次执行

看到DRAFT_OPERATOR_PROFILE_STARTED后执行下面命令。若返回PENDING，稍后重复同一recover；不要为了取CSV重新启动模型。

```bash
python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --recover "$AI_RUN_DIR/draft/tests/pytorch-optimized-profile-run" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-profile-result"
```

回传本轮完整回收摘要、model.csv，以及op.csv中耗时最高的前20行和算子总和。先核对同一模型的inference与CPU／NPU算子数据，再决定剩余运行时差额、CPU ArgMax、head MatMul或特征重排的优化优先级。加载／卸载另计；不能用本轮104.8ms API减去旧模型78.986ms算子总和，也不能将旧105.582ms差额视为已全部消除。
