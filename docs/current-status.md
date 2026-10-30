# 当前执行计划

<a id="用户现在执行什么"></a>

**2026-10-30最新目标：今天在端侧自主完成原40层target＋完整六层draft的多轮端到端DFlash。** 精度暂不作为P0通过条件，端侧自主性必须通过。

服务器只负责一次部署、启动和最终结果回收。请求启动后，embedding查表、prompt输入、draft、decode1 verify、接受／拒绝、80状态与特征反馈和停止都在板端完成。连接中断后已经启动的端侧请求仍应自行完成，重连回收同一attempt，不重新推理。

## 新执行入口

- 新板端C++调度器复用原decode1工具及完整S64 draft runner，P8/G8，每轮考虑2个候选。
- 两个OMC保持不变；runner构建不等于模型重新编译。不需要S128导出。
- 一次准备embedding表／ABI／runner并复用模型缓存；端侧每步保留日志、输入输出哈希、80状态链与完整feature历史。
- detached启动、持久checkpoint以及只回收不重跑的恢复入口。
- runner显式采用Release构建；板端合并重复文件读取／哈希，最终回收去掉每个文件重复的远端SHA查询。模型、80状态副本和校验规则保持不变。
- `supervisor.stdout`记录启动哈希和每个子进程开始／结束；`loop.json`的`supervisor_timing`记录墙钟耗时，包含加载／执行／卸载，不是NPU纯推理时间。
- 主机模拟已覆盖接受／纠正／bonus、状态槽位重排、子进程失败与断连恢复。独立集成使用实际C++调度器和实际draft CLI、模拟HIAI／target，完成16次target与5次draft；这不是实机通过。

在DDK主机沿用已经配置好的`REPO_DIR`、`AI_RUN_DIR`、`BIP`、`SN`及Native SDK环境。等旧任务正常结束后运行；如果共享板端锁仍被占用，新任务会等待，不能通过删除主机output清锁。`--output`必须是新目录。

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
  --output "$AI_RUN_DIR/dflash-board8-run" \
  --apply --detach
```

`BOARD_SERIAL_DFLASH_STARTED`只表示请求已发起，**不是完整流程通过**。这条命令返回后没有主机控制器继续驱动token；板端程序自行运行。可以退出这条主机进程／终端，之后用下面命令等待并取回同一请求：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_board_serial_dflash.py" \
  --recover "$AI_RUN_DIR/dflash-board8-run" \
  --output "$AI_RUN_DIR/dflash-board8-recovery"
```

恢复只查询完成状态、回收与校验文件，绝不再发起模型推理。若想验证物理断连后的自主性，在收到`STARTED`后断开HDC连接，恢复连接后再执行回收命令；保留此次操作及时间记录。无需断电或重启设备。若不需要分开启动／回收，第一条命令可省略`--detach`，主机仅定期查询请求是否结束，也不会参与逐token计算。

预期成功：`BOARD_SERIAL_DFLASH_EXECUTED_UNVALIDATED`或恢复等价状态，生成8个token、16次target、至少2次draft，最终82输出、80状态反馈和每轮日志通过校验。实际是否经过接受／拒绝分支以实机记录为准，不能把主机模拟分支覆盖当成实机覆盖。

结果：部署记录在`dflash-board8-run/board/checkpoint.json`；恢复后的汇总在`dflash-board8-recovery/result.json`，端侧循环及日志在其`results/`。中间输入输出保留在checkpoint记录的板端`board_attempt`；最终回收不复制模型权重，也不逐token往返传输大张量。

若HDC断连导致启动响应丢失，保留原目录，优先对原请求执行`--recover`。若失败发生在部署阶段、尚未记录启动请求，需根据checkpoint和原调用日志核对锁与远端命令完成状态；不要直接删锁、删output或重新发起。若板端子进程挂住，本版本保留PID并等待，不自动kill或重跑。当前入口仍按次加载模型，板端保留完整中间张量；本次I/O优化没有实现模型常驻，也没有实机加速比。

用户已经运行的旧host任务可继续产生联调证据。旧 `run_serial_dflash_smoke.py` 即使完成，也不计为今天端侧自主目标通过。

用户要求并行减少运行耗时并考虑模型常驻。已取得所选DDK的实际V2 `InitWeights(weightDir)`／`DeinitWeights()`声明，正在补独立target常驻适配。它和现有入口分开集成，未经实机加载验证不替换已成功的工具路径。当前P8流程没有独立prefill模型，8个prompt token仍由decode1输入；未来prefill桥接完成后，采用prefill一次运行并卸载、decode与draft常驻的生命周期。并发卸载／加载必须在状态交接和峰值内存可行后单独验证。见[常驻执行方案](resident-runtime-plan.md)。

当前仍未收到端侧自主完整流程的实机成功结果。P0成功后再处理普通基线／精度、跨导出prefill桥接、长上下文、模型常驻、批量verify／回滚与实测加速。

完整后续阶段、预计天数和换助手交接见 [执行总计划](dflash-delivery-plan.md)，机器可读进度见 [执行状态](../specs/dflash-execution-state.json)。
