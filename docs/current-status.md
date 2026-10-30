# 当前执行计划

<a id="用户现在执行什么"></a>

**当前主线：按用户要求，将 prefill、decode 和 draft 的新执行路径统一为模型管家 V1。** V2 在两次独立加载实验中均 Init／InitWeights成功，但显式DeinitWeights分别返回3和1；原因未定，见[历史观察](../specs/resident-v2-target-cleanup-observed.json)。用户随后提供了官方V2弃用说明，并明确要求全部改V1。此前让用户继续运行V2闭环的指令已经撤回；旧attempt和结果保留用于恢复及对照。

**P0 原 prefill 端侧完整执行已完成。** 旧版用户回传 prefill1／decode8／draft7，token为 `[4,12,10,8,0,4,32,2]`，接受0／7，总墙钟305.788秒。它证明原板端执行闭环，不证明新V1常驻、精度或加速，见[旧成功基线](board-prefill-first-completion.md)。

## 新流程与接口约束

一次部署和启动后，板端同一执行线程完成：

1. 创建独立V1 prefill实例，Load一次，Process一次，UnLoadModel后释放。
2. 创建decode和draft两个独立V1实例，各Load一次。
3. 在两个常驻实例间交替Process，完成候选、decode1 verify、接受／纠正和80状态反馈。
4. 关闭两个实例，回传最终token、接受率、分阶段时延及内存摘要。

模型管家不跨线程共享。同一实例只调用一次Load，失败后不重新Load；不使用异步接口或任务队列。prefill释放后才加载decode/draft；不并行加载卸载。每个实例只加载一个模型，循环期间保留两个实例。

三个既有OMC、权重、六层draft S64及15候选不变；只编译native runner。循环内的embedding查表、状态和特征传递、接受判断均在板端，没有逐token服务器交互。

## 在 DDK 主机启动新 V1 请求

V1改造和主机回归已完成，下一步直接运行以下完整板端命令。`REPO_DIR`、`AI_RUN_DIR`、`BIP`、`SN`和Native SDK沿用现有配置。输出必须使用新目录，旧准备清单不能被重新解释成V1执行。

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
  --output "$AI_RUN_DIR/dflash-board-resident-v1-8-run" \
  --resident --apply --detach
```

历史参数路径中的 `v2` 是既有描述符／状态图检查文件的名字，不表示新循环调用V2。V1重新查询实际shape和buffer字节数；无法查询的V1 tensor名称、dtype不得冒充已经由新SDK观察。外置target权重解析和全部V1实机结果均需这次执行验证。

启动响应 `BOARD_SERIAL_DFLASH_STARTED` 只说明已启动。恢复命令只回收同一attempt，不再次推理：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_board_serial_dflash.py" \
  --recover "$AI_RUN_DIR/dflash-board-resident-v1-8-run" \
  --output "$AI_RUN_DIR/dflash-board-resident-v1-8-recovery"
```

恢复不添加resident／dump／模型参数，沿用旧记录。断连或启动响应丢失时保留目录、恢复原请求；不删除锁或重跑未知状态。无需分开启动和回收时可以省略detach；主机只粗粒度查询完成状态，不参与循环。

## 本次验证内容

- `execution_mode=resident-v1`，prefill1、decode8和至少两次完整六层draft；每次draft输出15个候选，最多考虑前K=2，首个不匹配就纠正。
- decode/draft各一次Load、多次Process，全部V1调用；prefill也是V1且仅执行一次，没有原工具V2子进程。
- 对照旧同模型、同输入、同控制的token `[4,12,10,8,0,4,32,2]`。差异先按API／状态迁移回归定位，不因P0允许乱码就忽略。
- 接受率和prefill／decode／draft时延，另列加载与关闭耗时。时延包含所声明的复制和读回，不称纯NPU推理时间。内存统计为进程RSS/HWM和系统MemAvailable，不是独立NPU显存。
- 失败保留准确阶段、Load／Process／Unload返回码和已经完成的token/计数。回收stage=complete不等于模型成功；清理错误不被吞掉。

默认dump=false，只回收token与小型摘要。需要逐状态证据时在新请求加dump并使用新output；旧请求恢复模式不变。新V1调用失败时停止并保留证据，不自动切回V2，也不重新编译模型或改动原权重。

## 保留的模型及阶段边界

prefill物理长度64、逻辑prompt长度8，沿用 `PREFILL64_PADDED_SHORT_PROMPT_EXPERIMENTAL` 的控制和输入。前8行原始embedding/scale后补零，logits选第7行。前缀和特征不裁剪，P+G不超过原draft容量64。输入仍是token ID文件；板端分词和转文字尚未包含。

P0已完成，P5常驻工程按用户要求提前推进；P1/P2精度、P3历史一致性、P4增量draft、P6批量verify、P7性能、P8交付继续见[总计划](dflash-delivery-plan.md)。原始全网精度、有效接受率、稳定性和配对加速仍未验收。
