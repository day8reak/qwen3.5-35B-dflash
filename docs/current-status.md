# 当前执行计划

<a id="用户现在执行什么"></a>

**下一步：直接运行显式 V1 文件加载候选的完整板端 DFlash；这条候选尚未实机通过。** 原版外置权重 prefill／decode 采用 `FILE_ONLY`：只调用 `SetModelPath`，验证绝对路径，再由 SDK Load 决定兼容性。内嵌权重 draft 采用 `BUFFER_ONLY`：只调用 `SetModelBuffer`，验证完整buffer和生命周期。两条路径不再要求互斥字段同时保留，不交换setter顺序，也不在失败后自动尝试另一加载方式。

用户回传旧 `707d825` 请求在 `prefill-v1-load` 报 `SetModelBuffer did not preserve exact model path`；SDK Load／Process均未调用，生成token为0，锁已释放。该实现遗漏了已有实机互斥setter证据，fake默认双字段共存又掩盖了冲突。`50184b4` 随后恢复embedded draft的buffer-only加载并修正fake默认行为；707d825失败与50184b4主机验证记录保留，不能作为本次文件加载候选的设备PASS，详见[验证记录](validation.md#resident-v1-exclusive-setters-failure)。旧运行目录不覆盖。

本次外置模型候选依据是用户提供的4B `model_run_tool` V1源码：path-only、`OFFLINE`（字面值1），随后Load。它支持开展用户已授权的完整实验，尚不证明当前35B的外置权重解析正确。历史 `bf545c2` 的path-only HCL空buffer失败属于10月23日tiny target／draft bring-up，未证明是当前原版40层prefill／decode OMC，也不能断言运行库身份完全相同。选定SDK头文件和加载实现仍有诊断价值，但不再作为这次完整实验的前置阻塞；不新增load-only试验。

[华为官方固定示例](https://github.com/huaweicodelabs/HiAI-Foundation/blob/83320362c10f2e85a2a1bbb283b3359db204ab2a/app/src/main/jni/classify_sync_jni.cpp#L85-L107)使用builder创建buffer和 `ONLINE`，并非用户的path-only代码。用户另提的[公开4B分支](https://github.com/day8reak/qwen3.5-4B-dflash/tree/90186732d915e8b058b2a053d205991d805eae76)当前使用Ascend310P／AscendCL，也不提供所选Kirin SDK的V1外置加载证明。

**目标仍是按用户要求，将 prefill、decode 和 draft 统一为模型管家 V1。** V2 在两次独立加载实验中均 Init／InitWeights成功，但显式DeinitWeights分别返回3和1；原因未定，见[历史观察](../specs/resident-v2-target-cleanup-observed.json)。用户随后提供了官方V2弃用说明，并明确要求全部改V1。此前让用户继续运行V2闭环的指令已经撤回；旧attempt和结果保留用于恢复及对照。

**P0 原 prefill 端侧完整执行已完成。** 旧版用户回传 prefill1／decode8／draft7，token为 `[4,12,10,8,0,4,32,2]`，接受0／7，总墙钟305.788秒。它证明原板端执行闭环，不证明新V1常驻、精度或加速，见[旧成功基线](board-prefill-first-completion.md)。

本次候选191项Python测试及13组native CTest通过，其中9项集成测试使用实际C++运行框架和fake SDK，覆盖完整debug／compact循环及文件加载失败；这些不代替实机加载与推理。证据见[当前候选验证](../specs/resident-v1-file-buffer-candidate.json)。

## V1 候选设计与接口约束（尚未实机完成）

一次部署和启动后，板端同一执行线程完成：

1. 创建独立V1 prefill实例，Load一次，Process一次，UnLoadModel后释放。
2. 创建decode和draft两个独立V1实例，各Load一次。
3. 在两个常驻实例间交替Process，完成候选、decode1 verify、接受／纠正和80状态反馈。
4. 关闭两个实例，回传最终token、接受率、分阶段时延及内存摘要。

模型管家不跨线程共享。同一实例只调用一次Load，失败后不重新Load；不使用异步接口或任务队列。prefill释放后才加载decode/draft；不并行加载卸载。每个实例只加载一个模型，循环期间保留两个实例。

三个既有OMC、权重、六层draft S64及15候选不变；只编译native runner。循环内的embedding查表、状态和特征传递、接受判断均在板端，没有逐token服务器交互。

新请求明确绑定 `target_lifecycle_policy=legacy-v1-file-target-buffer-draft-v1`；session摘要为 `model_load_api=legacy-v1-mixed`，prefill／target为 `legacy-file`，draft为 `legacy-buffer`。FILE_ONLY检查path精确且buffer为空、长度0；这些检查与SDK加载成功是不同结论。旧请求恢复仍使用其原配置。

## 当前完整 V1 候选命令

更新到包含显式文件加载候选的版本后，使用新output启动。复用三个现有OMC、权重和绑定，只构建native runner；不运行OMG、不重新编译模型。主机调度／加载替身测试不代表本候选已在所选SDK加载成功。

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
  --output "$AI_RUN_DIR/dflash-board-resident-v1-file8-run" \
  --resident --apply --detach
```

参数路径中的 `v2` 是既有描述符／状态图检查文件的名字，不表示新循环调用V2。设计上V1重新查询实际shape和buffer字节数；无法查询的V1 tensor名称、dtype不得冒充已经由新SDK观察。实际Load后才查询和验证这些V1运行时描述符；目前尚无本候选的设备结果。

启动响应 `BOARD_SERIAL_DFLASH_STARTED` 只说明板端进程已启动，不代表生成成功。首次命令使用 `--apply --detach` 后，板端自主运行；等待完成再用以下命令回收同一attempt，不重新启动推理：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_board_serial_dflash.py" \
  --recover "$AI_RUN_DIR/dflash-board-resident-v1-file8-run" \
  --output "$AI_RUN_DIR/dflash-board-resident-v1-file8-recovery"
```

恢复不添加resident／dump／模型参数，沿用该请求记录。断连或启动响应丢失时保留目录、恢复原请求；不删除锁或重跑未知状态。设计上不依赖运行期间的服务器连接，物理断连完成性仍需实际观察。

<a id="历史-v1-请求命令当前阻塞勿重跑"></a>

旧 `dflash-board-resident-v1-8-run`／`dflash-board-resident-v1-8-recovery` 保留为707d825历史请求；其已回传失败且锁已释放，无需再次启动或恢复。新候选使用上述 `file8` 目录，不能用新配置覆盖旧请求。

## 本次候选的验收目标

- `execution_mode=resident-v1`，prefill1、decode8和至少两次完整六层draft；每次draft输出15个候选，最多考虑前K=2，首个不匹配就纠正。
- decode/draft各一次Load、多次Process，全部V1调用；prefill也是V1且仅执行一次，没有原工具V2子进程。
- 对照旧同模型、同输入、同控制的token `[4,12,10,8,0,4,32,2]`。差异先按API／状态迁移回归定位，不因P0允许乱码就忽略。
- 接受率和prefill／decode／draft时延，另列加载与关闭耗时。时延包含所声明的复制和读回，不称纯NPU推理时间。内存统计为进程RSS/HWM和系统MemAvailable，不是独立NPU显存。
- 失败保留准确阶段、Load／Process／Unload返回码和已经完成的token/计数。回收stage=complete不等于模型成功；清理错误不被吞掉。

默认dump=false，只回收token与小型摘要。需要逐状态证据时在新请求加dump并使用新output；旧请求恢复模式不变。新V1调用失败时停止并保留证据，不自动切回V2，也不重新编译模型或改动原权重。

## 保留的模型及阶段边界

prefill物理长度64、逻辑prompt长度8，沿用 `PREFILL64_PADDED_SHORT_PROMPT_EXPERIMENTAL` 的控制和输入。前8行原始embedding/scale后补零，logits选第7行。前缀和特征不裁剪，P+G不超过原draft容量64。输入仍是token ID文件；板端分词和转文字尚未包含。

P0已完成，P5常驻工程按用户要求提前推进；P1/P2精度、P3历史一致性、P4增量draft、P6批量verify、P7性能、P8交付继续见[总计划](dflash-delivery-plan.md)。原始全网精度、有效接受率、稳定性和配对加速仍未验收。
