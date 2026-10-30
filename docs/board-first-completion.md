# 已完成子里程碑：decode1串行prompt的端侧自主循环

2026-10-30，用户先回传以下serial流程的完整dump及compact成功结果。这两项本身只是子里程碑；随后原prefill驱动的完整请求也已成功，P0执行通过见[独立原prefill记录](board-prefill-first-completion.md)。助手未连接设备，以下依据用户终端回传。

## 原完整dump回传

- 原请求：`$AI_RUN_DIR/dflash-board8-run`；回收：`$AI_RUN_DIR/dflash-board8-recovery`。
- attempt：`/data/local/tmp/qwen35_board_serial_1056ef4d658548d5ba16ee7c505a34d7`。
- 状态：`BOARD_SERIAL_DFLASH_EXECUTED_UNVALIDATED`，`EXECUTED_OUTPUTS_RETRIEVED`。
- target16次、draft7次；considered7、accepted0、corrections7、bonuses0。
- 锁：`NO_LOCK_HELD_OR_RELEASED_AFTER_KNOWN_EXIT`。模型编译未运行，精度未评估。

该请求用decode1逐token输入8个prompt token，再生成8个token；没有调用原prefill模型。恢复只回收同一次attempt，不重新推理。原完整dump回收路径检查每次receipt、输出槽位及80状态字节链。这是已通过的serial子里程碑，不能将P0整体改为passed。

## 最新compact回传

用户随后回传compact模式的同类serial请求成功结果：

| 项目 | 用户回传 |
|---|---:|
| 生成token IDs | `[26,18,2,4,6,2,0,2]` |
| target／draft调用 | 16／7 |
| accepted／considered | 0／7，接受率0 |
| corrections／bonuses | 7／0 |
| 总墙钟 | 367791 ms（367.791秒） |
| 串行prompt输入 | 8次，74389 ms（74.389秒） |
| 生成阶段decode | 8次，108209 ms（108.209秒） |
| draft | 7次，137646 ms（137.646秒） |

`prompt_ingest`的机制是`DECODE1_SERIAL_PROMPT_INGESTION`，不是原prefill模型的耗时。各子进程时延包含加载、执行和卸载；总墙钟还包含源检查及调度I/O，不含主机部署／回收。这些数值不是纯NPU推理时间，也没有普通baseline配对加速比。最新compact请求的精确目录与模型／配置身份以用户保留的原始result和checkpoint为准，不由此摘要补造路径。

compact只回收tokens、小型时延／计数报告和进程完成证据；**没有在主机重放逐候选或80状态字节链**，不能继承完整dump的证据范围。它仍是同一serial子流程，不是原prefill集成通过。

## 当时尚未完成及后续结果

- 原prefill输出的80状态、logits及8层features接入draft／serialverify，并完成后续多轮生成。
- 同一请求的prefill／decode variant身份、anchor和draft容量处理；路线仍在确定，S128不是自动前置条件。
- 真实接受或bonus分支覆盖：本次自然出现的是全拒绝／target纠正。
- 物理断连后的专门实测、数值精度、普通baseline一致性、模型常驻及配对性能验收。

上述前两项随后由原prefill请求完成执行闭环，数值语义仍未验收；其余项继续独立验收。保留全部成功目录，当前沿统一V2常驻推进。
