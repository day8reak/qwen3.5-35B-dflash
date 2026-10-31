# 当前执行计划

<a id="用户现在执行什么"></a>

**用户已同意优先推进混合接口常驻：prefill／decode 使用 V2，内嵌权重的完整六层 draft 使用 V1。** 这取代此前全部V1的要求；三个OMC及权重不变，只重建native runner。当前为待实机验证的新候选，不宣称加载、清理或双模型常驻已经成功。

一次部署和启动后，板端同一线程执行：

1. V2恢复原prefill64，Init后InitWeights，执行一次，关闭并释放该实例。
2. V2恢复decode1并初始化外置权重；V1从完整embedded buffer加载draft。两个实例各加载一次。
3. 板端查原embedding表，交替decode Run和draft Process，执行串行verify、接受／纠正及80状态反馈。
4. 关闭两个实例，只回传最终token、接受率、分阶段时延及内存摘要。循环期间没有逐token服务器交互。

prefill卸载与decode加载仍顺序执行，调度上不主动重叠加载两份约18.8GB的target权重；实际释放情况仍需板端观察。draft保持S64、每次15候选，当前最多考虑前K=2；首个不匹配即纠正。输入仍是token ID文件，板端分词和文字解码尚未包含。

## 新请求与恢复

本轮主机验证已通过：186项Python单元测试、7个完整C++循环／恢复场景和18组native CTest。7个场景中两项恢复测试曾缺少测试环境变量，修正后同名补跑通过，原失败记录保留。均为主机fake验证，实机结果待回传。DDK主机拉取更新后使用以下新目录。

```bash
git -C "$REPO_DIR" pull --ff-only

python3 "$REPO_DIR/targets/cpro/scripts/run_board_serial_dflash.py" \
  --decode-job "$AI_RUN_DIR/target-pytorch-decode1-compile" \
  --runtime-probe "$AI_RUN_DIR/target-decode1-runtime-io-v2" \
  --state-lineage "$AI_RUN_DIR/target-decode1-state-lineage-v7" \
  --prefill-feedback "$AI_RUN_DIR/target-prefill64-decode1-feedback-plan-v2" \
  --source-torchair-pb2 "$AI_RUN_DIR/target-decode1-row-response/ge_ir_by_protoc_3_19_pb2.py" \
  --draft-bundle "$AI_RUN_DIR/full-draft/residual-fix/bundle" \
  --weight-table "$AI_RUN_DIR/original-embedding-tables/qwen3_5_moe_embdedding_weight.bin" \
  --scale-table "$AI_RUN_DIR/original-embedding-tables/qwen3_5_moe_embdedding_scale.bin" \
  --prompt-input-ids /home/w00949577/run_35B/chunksize_64_data/input_ids.bin \
  --prompt-length 8 --max-new-tokens 8 --max-proposals-per-round 2 \
  --output "$AI_RUN_DIR/dflash-board-resident-mixed8-run" \
  --resident --apply --detach
```

等待板端完成后回收同一请求，不重新启动推理：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_board_serial_dflash.py" \
  --recover "$AI_RUN_DIR/dflash-board-resident-mixed8-run" \
  --output "$AI_RUN_DIR/dflash-board-resident-mixed8-recovery"
```

启动状态 `BOARD_SERIAL_DFLASH_STARTED` 不等于推理成功。断连时保留原请求并恢复，不删除锁或重启未知状态的进程。恢复沿用原dump/API配置；旧V1/V2收据仍按其历史配置核验。

## 清理策略的具体范围

新mixed请求在加载之前明确选择 `MANAGER_DEINIT_ONLY_NO_SEPARATE_WEIGHT_UNLOAD`：调用V2 `DeInit`，记录是否正常返回，再销毁manager、built model和tensor对象。它不会先调用失败的DeinitWeights再忽略错误；该API在新策略下未调用，记录called=false、status=-1。DeInit返回void，不伪造status0；若抛异常仍记录清理失败，不重试。

此前Init→InitWeights均成功；单独DeinitWeights在两种顺序下返回3或1的观察全部保留。官方V2索引列有DeInit，但没有取得证明私有InitWeights资源所有权的正文。因此新策略是显式实验，报告保留 `external_weight_release=NOT_INDEPENDENTLY_VALIDATED_MANAGER_DEINIT_ONLY`。正常调用返回和销毁对象不能独立证明板端外置权重内存已释放。

新请求绑定 `execution_mode=resident-mixed`、`model_load_api=target-v2-draft-v1`，三模型API分别为file-v2／file-v2／legacy-buffer；策略为 `v2-target-v1-draft-resident-v1`。失败不自动切换加载方式、不重编模型。Host fake只验证框架调度和错误处理，不验证所选SDK兼容性。

## 保留的结果与验收要求

**P0原prefill板端完整执行已完成。** 旧版prefill1／decode8／draft7，token为 `[4,12,10,8,0,4,32,2]`，接受0／7，总墙钟305.788秒，见[旧成功基线](board-prefill-first-completion.md)。它不证明新常驻、精度或加速。

全V1的file8请求在prefill SDK Load返回1；PID63072 HiLog显示所选HCL拒绝path-only描述符的空buffer，尚未进入OMC恢复、权重解析或Process，见[失败记录](validation.md#resident-v1-hcl-file-load-failure)。旧请求已退出且锁释放，不重复运行。用户工具的V1头文件未显示独立外置权重入口，这不证明所有私有V1扩展均不可用。

新候选验收要看到prefill一次、decode和draft各单次加载及多次调用、全部在板端完成。先对照旧同输入token序列及80状态传递，再解释差异；P0允许不验精度不等于可以忽略API迁移引入的新错误。默认不dump中间张量；时延分别标明加载、运行、关闭，运行包含复制/读回/校验，不称纯NPU时间。内存仅为RSS/HWM及MemAvailable，不称独立NPU显存。

prefill仍是物理64、逻辑8的实验控制。精度、有效接受率、稳定性、长上下文与配对性能按[总计划](dflash-delivery-plan.md)继续。若未来compact请求失败，可用 `collect_board_serial_failure_logs.py --recovery 已完成的失败恢复目录 --output 新诊断目录 --hilog` 只读收集日志；本次旧V1日志已足够，无需再采集。
