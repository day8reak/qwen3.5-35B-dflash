# 当前执行计划

<a id="用户现在执行什么"></a>

**2026-10-30：P0 原 prefill 端侧完整执行已完成。** 用户回传 prefill 1 次、decode 8 次、draft 7 次，生成 `[4,12,10,8,0,4,32,2]`，接受 0／7，总墙钟 305.788 秒，回收及锁释放完成。见 [实机成功基线](board-prefill-first-completion.md)。这属于执行通过，精度和加速尚未验收。

**当前阻点：首次常驻请求在 target 加载失败，暂不重复同一调用顺序。** 实际 prefill 已完成，`InitWeights` 在 `Init` 之前返回 `3=INVALID_PARAM`；decode／draft 均未 Run，draft 尚未加载。失败请求已回收并释放所持锁，见 [首个常驻加载失败](../specs/resident-v2-first-load-failure.json)。当前先核对原成功工具的 `weight_dir`／V2 加载实现，并修复错误摘要及构造失败快照；未取得证据前不交换调用顺序、不重编模型。

**已实现的常驻入口（下面命令保留作参数参考，待加载修复后换新目录运行）：** 新增显式 `--resident`，原 prefill 完成并退出后，同一 native 进程保留 decode 和内嵌 draft 各一个 V2 manager，加载一次、多次 `Run`。旧成功模式 draft 为 `legacy-buffer`；实际 DDK 编译及 prefill 调用已由本次执行观察到；target V2 初始化失败，同时驻留仍未验证，不能把旧成功结果算到新路径。见 [常驻方案](resident-runtime-plan.md) 和 [主机验证记录](board-resident-validation.md)。

服务器仍只做一次部署、启动和最终回收。循环内的 embedding 查表、draft、decode1 verify、接受／纠正、80 状态及特征反馈都在板端。原三个 OMC 和权重不变，只编译 runner；不要求 S128 AIR 或重编 draft。

## 在 DDK 主机启动新请求

沿用配置好的 `REPO_DIR`、`AI_RUN_DIR`、`BIP`、`SN` 与 Native SDK 环境。保留已有成功目录；下面 `--output` 必须是尚不存在的新目录。共享板端锁有其他任务时等待，不通过删除 output 或锁抢占。

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
  --output "$AI_RUN_DIR/dflash-board-resident8-run" \
  --resident --apply --detach
```

`BOARD_SERIAL_DFLASH_STARTED` 只表示已启动，不代表完成。启动响应返回后无需主机逐步驱动；恢复只等待和取回同一个 attempt，不重新推理：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_board_serial_dflash.py" \
  --recover "$AI_RUN_DIR/dflash-board-resident8-run" \
  --output "$AI_RUN_DIR/dflash-board-resident8-recovery"
```

恢复命令不再添加 `--resident`、`--dump` 或模型参数，沿用原准备记录。若无需分开启动／回收，启动命令可省略 `--detach`；主机仅粗粒度查询完成状态。HDC 中断或启动响应丢失时保留原目录，先恢复同一请求；不要删目录重跑，避免重复推进。

## 这次检查什么

- `execution_mode=resident-v2`，一次 prefill、8 次 decode、至少两次完整六层 draft，最终 8 个 token。每轮 draft 实际产出 15 个候选，默认只考虑前 K=2，遇到首个不匹配即纠正。
- `resident_lifecycle` 中 `target_loads=1`、`draft_loads=1`，多次 Run 共用同一个进程／session；成功结束时两者关闭。prefill 是唯一模型子进程，不能把 resident 每次 Run 当作子进程退出。
- 对照同模型、同输入、同控制的旧成功 token `[4,12,10,8,0,4,32,2]`。若不同先定位新运行接口／状态差异，不因 P0 容许数值误差就忽略生命周期回归。
- 接受率、`phase_latency.prefill/decode/draft`、一次性加载／关闭耗时及内存摘要。prefill 是含加载／卸载的子进程墙钟；decode／draft 是排除模型加载的 session Run 墙钟，包含输入复制、读回校验和输出复制，不是纯 NPU 时间。
- `resident_memory` 记录进程 VmRSS／VmHWM 与系统 MemAvailable；不是专用 NPU 内存。读取受限时保留缺失状态，不把 0 当作实际占用。

默认 `dump=false`，只回收最终 token、小型完成记录、时延和内存摘要；不在主机逐状态重放。需要详细状态／精度诊断时，在**新启动命令**加 `--dump` 并改用新 output。完整模式保留逐状态、实际输入读回、候选和提交证据。旧请求恢复模式不会因更新代码改变。

本次消除 decode／draft 逐调用加载和子进程启动；仍保留板端私有状态文件和输出副本。compact 清理已消费的成功 scratch，失败或未知状态保留。它尚不是零拷贝执行，也没有经过实机配对加速验收。若 V2 初始化／Run 失败，记录确切阶段并停止，不自动降级旧接口，不自动换 Init 顺序。

## 保留的模型及阶段边界

原 prefill 的物理长度为 64、逻辑 prompt 长度为 8，使用 `PREFILL64_PADDED_SHORT_PROMPT_EXPERIMENTAL`。8 行原表 embedding／scale 后补零，logits 选择第 7 行；128 项 conv 控制、有效长度及 mask 在准备清单中明确记录。prefill 退出后交接 80 状态及前 8 行特征；首个生成 token 在绝对位置 8 推进 decode 后，将完整 9 行历史交给原 S64 draft。

完整历史不裁剪，整个请求 P＋G 不超过 64。原私有算子的短有效长度语义、数值精度和接受分支仍需验证。输入是 token ID 文件，板端文本分词／token 转文字尚未包含。物理断连实测尚未提供；detached 设计不等于该测试已通过。

已成功的按次加载入口保留：省略 `--resident` 即为旧模式，但复现也需新 output。旧 `dflash-board-prefill8-run/recovery` 不要重跑或删除；更早 `run_serial_dflash_smoke.py` 的逐步主机交互路线不再作为当前执行主线。

P0 已通过，P5 工程按用户要求提前；P1／P2 精度、P3 历史一致性、P4 增量 draft、P6 批量 verify、P7 性能、P8 交付继续按 [总计划](dflash-delivery-plan.md) 推进。状态及预计天数见 [机器可读执行状态](../specs/dflash-execution-state.json)。
