# 当前执行计划

<a id="用户现在执行什么"></a>

**2026-10-30：用户已回传首次端侧自主多轮完成结果。** 原请求`dflash-board8-run`共16次target、7次draft，考虑7个候选、接受0个、纠正7个；回收成功且锁已释放。P0执行里程碑完成，精度与加速仍未验收。详见[首次实机结果](board-first-completion.md)。成功目录保留，不要求重复运行。

服务器只负责一次部署、启动和最终结果回收。请求启动后，embedding查表、prompt输入、draft、decode1 verify、接受／拒绝、80状态与特征反馈和停止都在板端完成。连接中断后已经启动的端侧请求仍应自行完成，重连回收同一attempt，不重新推理。

## 新执行入口

- 新板端C++调度器复用原decode1工具及完整S64 draft runner，P8/G8，每轮考虑2个候选。
- 两个OMC保持不变；runner构建不等于模型重新编译。不需要S128导出。
- 一次准备embedding表／ABI／runner并复用模型缓存；端侧执行仍检查真实输出和80状态交接。新请求默认精简模式，只回收最终token及小型状态／时延摘要；`--dump`开启详细调试数据保留与回收。
- detached启动、持久checkpoint以及只回收不重跑的恢复入口。
- runner显式采用Release构建；板端合并重复文件读取／哈希，最终回收去掉每个文件重复的远端SHA查询。模型、80状态副本和校验规则保持不变。
- `supervisor.stdout`记录启动哈希和每个子进程开始／结束；`loop.json`的`supervisor_timing`记录墙钟耗时，包含加载／执行／卸载，不是NPU纯推理时间。
- 主机模拟已覆盖接受／纠正／bonus、状态槽位重排、子进程失败与断连恢复。独立集成使用实际C++调度器和实际draft CLI、模拟HIAI／target，完成16次target与5次draft；这不是实机通过。

在DDK主机沿用已经配置好的`REPO_DIR`、`AI_RUN_DIR`、`BIP`、`SN`及Native SDK环境。旧host任务已在第二次draft的依赖库发现处失败，用户选择直接运行新版；保留旧目录，不重跑旧host循环。如果共享板端锁仍被占用，新任务会等待，不能通过删除主机output清锁。`--output`必须是新目录。

```bash
git -C "$REPO_DIR" pull --ff-only

python3 "$REPO_DIR/targets/cpro/scripts/run_board_serial_dflash.py" \
  --decode-job "$AI_RUN_DIR/target-pytorch-decode1-compile" \
  --runtime-probe "$AI_RUN_DIR/target-decode1-runtime-io-v2" \
  --state-lineage "$AI_RUN_DIR/target-decode1-state-lineage-v7" \
  --source-torchair-pb2 "$AI_RUN_DIR/target-decode1-row-response/ge_ir_by_protoc_3_19_pb2.py" \
  --draft-bundle "$AI_RUN_DIR/full-draft/residual-fix/bundle" \
  --weight-table "$AI_RUN_DIR/original-embedding-tables/qwen3_5_moe_embdedding_weight.bin" \
  --scale-table "$AI_RUN_DIR/original-embedding-tables/qwen3_5_moe_embdedding_scale.bin" \
  --prompt-input-ids /home/w00949577/run_35B/chunksize_64_data/input_ids.bin \
  --prompt-length 8 --max-new-tokens 8 --max-proposals-per-round 2 \
  --output "$AI_RUN_DIR/dflash-board8-compact-run" \
  --apply --detach
```

`BOARD_SERIAL_DFLASH_STARTED`只表示请求已发起，**不是完整流程通过**。这条命令返回后没有主机控制器继续驱动token；板端程序自行运行。可以退出这条主机进程／终端，之后用下面命令等待并取回同一请求：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_board_serial_dflash.py" \
  --recover "$AI_RUN_DIR/dflash-board8-compact-run" \
  --output "$AI_RUN_DIR/dflash-board8-compact-recovery"
```

恢复只查询完成状态、回收与校验文件，绝不再发起模型推理。若想验证物理断连后的自主性，在收到`STARTED`后断开HDC连接，恢复连接后再执行回收命令；保留此次操作及时间记录。无需断电或重启设备。若不需要分开启动／回收，第一条命令可省略`--detach`，主机仅定期查询请求是否结束，也不会参与逐token计算。

预期成功：`BOARD_SERIAL_DFLASH_EXECUTED_UNVALIDATED`或恢复等价状态，输出8个token、16次target、至少2次draft及接受率／分阶段时延。精简模式验证最终报告与已结束进程记录，不在主机重放逐状态／候选证据；`--dump`保留原完整校验路径。实际是否经过接受／拒绝分支以实机记录为准，不能把主机模拟分支覆盖当成实机覆盖。

结果：部署记录在`dflash-board8-compact-run/board/checkpoint.json`；恢复后的汇总在`dflash-board8-compact-recovery/result.json`。精简模式仅回收`loop.json`、`progress.json`和小型进程完成记录。当前工具需要临时文件交接；已成功消费的输入／父状态／draft文件及时清理，失败或未知子进程的文件保留在该attempt。不会清理共享模型缓存或旧成功目录。不开dump尚不等于全内存执行。

如需定位精度或状态问题，在**新请求启动命令**加`--dump`。恢复命令不接受该选项，沿用原请求的模式；旧请求未记录dump字段时按原完整模式恢复，不因升级代码丢弃证据。详细模式仍按旧策略回收日志、receipt和最终张量，中间工作文件保留在板端。

普通输出也包含`acceptance`（accepted／considered／rate，分母为实际考虑的候选数）和`phase_latency`。`prompt_ingest`是当前prefill阶段，由decode1逐token输入prompt，标记`DECODE1_SERIAL_PROMPT_INGESTION`；`decode`仅统计生成阶段target调用；`draft`统计draft调用。各项提供calls、total_ms、mean_ms。时延口径为子进程墙钟，含加载、执行、卸载，未提供纯NPU推理时间；`supervisor_timing.total_wall_ms`还包含源检查和调度I/O，不含主机部署／回收。

若HDC断连导致启动响应丢失，保留原目录，优先对原请求执行`--recover`。若失败发生在部署阶段、尚未记录启动请求，需根据checkpoint和原调用日志核对锁与远端命令完成状态；不要直接删锁、删output或重新发起。若板端子进程挂住，本版本保留PID并等待，不自动kill或重跑。当前入口仍按次加载模型，板端保留完整中间张量；本次I/O优化没有实现模型常驻，也没有实机加速比。

旧host任务记录为10次target、1次draft后失败，路径问题已修复并通过39项主机检查，详见[旧runner复用修复](legacy-draft-runner-reuse-fix.md)。旧 `run_serial_dflash_smoke.py` 即使完成，也不计为今天端侧自主目标通过。

用户要求并行减少运行耗时并考虑模型常驻。独立`HiaiOriginalTargetExecutor`已实现，依据所选DDK实际V2声明绑定89／82接口；主机替身连续3次Run、80状态回传及21种失败情形通过，连同既有runner回归共8项CTest通过。它尚未接入worker协议或默认循环，也未在实际DDK／设备验证初始化顺序；当前入口仍按次加载。当前P8流程没有独立prefill模型，8个prompt token仍由decode1输入；未来prefill桥接完成后，采用prefill一次运行并卸载、decode与draft常驻的生命周期。并发卸载／加载必须在状态交接和峰值内存可行后单独验证。见[常驻执行方案](resident-runtime-plan.md)。

首次完整dump实机结果已收到；新增精简模式与时延摘要需独立验证，不能直接继承旧二进制的实机结果。接下来处理普通基线／精度，继续并行完成常驻，再推进跨导出prefill桥接、长上下文、批量verify／回滚与实测加速。

完整后续阶段、预计天数和换助手交接见 [执行总计划](dflash-delivery-plan.md)，机器可读进度见 [执行状态](../specs/dflash-execution-state.json)。
