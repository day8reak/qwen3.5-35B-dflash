# draft加载阶段计时与顺序预读对照

## 最新实机结果：包含预读的draft加载降到3.883秒

用户回传 `dflash-board-resident-preread8-recovery`（PID21363）：prefill1／decode8／draft7，两个常驻模型各加载一次、正常关闭，接受0/7。完整结果来自用户摘要，助手未连接设备。[结构化记录](../../specs/board-resident-preread-observation.json)保留计数与边界。

| 指标 | 无预读PID17495 | 开启预读PID21363 |
|---|---:|---:|
| 顺序预读 | 未执行 | 1.621秒，完整2,852,119,022字节 |
| SDK Load | 206.630秒 | 2.251秒 |
| draft总加载（含预读） | 206.640秒 | 3.883秒 |
| 完整请求墙钟 | 234.506秒 | 31.455秒 |
| decode同步Run均值 | 112.819ms | 110.890ms |
| draft同步Process均值 | 739.192ms | 733.126ms |

本轮draft总加载约缩短53.2倍、完整请求墙钟减少86.6%，收益主要来自启动。预读耗时已计入，不是仅把等待移出Load。此结果支持提前顺序读取有益于加载的解释；仍有44,297次Load major fault，无法断言缺页已清零、具体存储／回收根因或重复稳定性。不同请求生成的token不同，以上不是数值等价后的推测解码加速比。

下一步按用户要求与[文字版完整请求](../reference/board-text-input-output.md#2-运行文字版完整闭环)合并；文档命令已带`--draft-preread`。CLI仍显式开启，不改变历史请求策略。

## 上轮加载定位：慢在SDK Load

用户回传 `dflash-board-resident-loadtiming8-recovery`（PID17495）已完成prefill1／decode8／draft7、接受0/7，正常关闭。总耗时234.506秒，draft一次加载206.640秒，其中 **MODEL_LOAD占206.630436秒**。mmap仅59µs，SetModelBuffer仅1µs。Load阶段记录128,122次major fault，其他阶段仅TENSOR_SETUP另有10次。完整关键字段见[用户观察](../../specs/board-resident-load-observation.json)。助手没有连接设备，以上来自用户粘贴的摘要。

这次测量确定了应用侧长耗时位于SDK Load。它支持“加载映射文件时缺页读取很慢”的假设，但无法确定fault地址、底层存储耗时、回收或换页，也不能把fault数乘页大小当成模型实际读取量。read_bytes=0不能在本设备上证明没有I/O；[Linux proc文档](https://docs.kernel.org/filesystems/proc.html)说明这些是特定进程计数，不能代替系统或SDK其他进程观测。

此前有全量启动哈希的resident请求draft加载8.380秒；移除哈希后的两次分别202.958、206.640秒。哈希本身也会访问整个映射文件，这个关联值得做顺序预读对照，但旧请求不构成受控冷／热缓存实验。

## 可选对照：`--draft-preread`

默认不开启。开启时沿用同一个只读文件描述符，mmap之后、SetModelBuffer／Load之前，用最多1MiB暂存缓冲区顺序pread完整draft OMC。保留原映射和模型字节、API、prefill→decode→draft加载顺序；不分配第二份整模型副本、不锁页、不清缓存、不加逐token磁盘操作。文件元数据前后变化、读失败或提前EOF会报错；不自动重试SDK Load或换接口。

新增 `MODEL_PREREAD` 独立计时及缺页计数，`load_profile.preread_bytes`须等于该draft OMC大小（当前2,852,119,022字节）。原 `resident_draft_load_ms` 和 `total_wall_ms` **包含预读耗时**。本轮实机已观察到改善；不同缓存和设备负载下仍需复测，CLI维持显式开启。

P8/G8单变量首轮已完成。下一次合并文字入口与预读；若专门复测加载，仍保持P8/G8原token文件，使用下面独立repeat目录。

## 历史定位

用户提供的PID12858筛选日志中，decode外置权重读取于00:29:38.476开始，下一批客户端加载记录于00:33:10.682出现，相隔212.206秒。结合target加载9.793秒、draft构造202.958秒，长耗时在下一批图加载日志出现前；筛选日志缺少SetModelBuffer／Load入口与返回时间，不能进一步确定阶段。

开头00:29:27.866的动态库加载错误后出现LoadModelsByHcl success，原prefill和后续循环均成功。没有证据把该行认定为203秒的原因。00:29:33的worker占用提示在prefill阶段，也不是draft超时的证明。多个client ID是SDK内部记录，不能据此把应用层一次draft Load说成重复加载多次完整draft。

## 新版增加什么

仅resident V1 draft的一次构造采样，结果位于：

`resident_lifecycle.draft_v1_lifecycle.load_profile.phases`

| phase | 覆盖范围 |
|---|---|
| CLIENT_INIT | Init及状态返回检查 |
| RUNTIME_QUERY_AND_DESCRIPTION | 版本／NPU查询、创建描述对象 |
| MODEL_MMAP | 打开、stat、只读mmap OMC |
| MODEL_PREREAD（仅开启时） | 同一文件描述符顺序读取、元数据检查；最多1MiB暂存 |
| SET_MODEL_BUFFER | SetModelBuffer及返回buffer大小检查 |
| MODEL_LOAD | 同步Load及状态检查 |
| IO_DESCRIPTORS | IO描述符查询、一次性打印及数量检查 |
| TENSOR_SETUP | 五个tensor分配、加载时重叠检查、初始化与ready记录 |

每段含wall_us和completed；失败段completed=false，保留原SDK错误，不重试、不降级。阶段开始／结束各写一行日志，便于定位异常中断。无逐token采样，不恢复全量哈希或中间tensor文件。默认七段；开启预读时八段。原模型、加载顺序、缓冲区持有期、推理与关闭路径不变。

可读时同时记录本进程minor_faults、major_faults、read_bytes、read_chars增量。缺失／计数回退记null，读取失败不影响推理。它们不覆盖SDK其他进程或NPU，不是系统总I/O；read_chars还包含探针自身的小量proc读取，不能用作精确模型读取字节数。阶段墙钟排除计数读取和两行日志，原draft_load_ms仍含整个构造，阶段和未必等于总数。mmap阶段快也不意味着后续没有按需读取成本。

## DDK主机下一次完整请求

当前优先执行[文字＋预读命令](../reference/board-text-input-output.md#2-运行文字版完整闭环)。下面保留token输入的重复测量命令：沿用原prompt与生成配置，默认dump=false，使用独立repeat目录。

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
  --prefill-feedback "$AI_RUN_DIR/target-prefill64-decode1-feedback-plan-v2" \
  --prompt-length 8 --max-new-tokens 8 --max-proposals-per-round 2 \
  --resident --draft-preread --apply --detach \
  --output "$AI_RUN_DIR/dflash-board-resident-preread8-repeat-run"
```

回收同一请求：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_board_serial_dflash.py" \
  --recover "$AI_RUN_DIR/dflash-board-resident-preread8-repeat-run" \
  --output "$AI_RUN_DIR/dflash-board-resident-preread8-repeat-recovery"
```

比较以下四项：

1. `MODEL_PREREAD.wall_us`、`preread_bytes`：预读花多久、是否读完。
2. `MODEL_LOAD.wall_us`、`major_faults`：SDK Load是否缩短、缺页是否减少。
3. `resident_draft_load_ms`：包括预读的完整加载时间，与206,640ms对照。
4. `total_wall_ms`：与234,506ms对照；同时保留prefill/decode/draft API时间和输出。

若Load变快但总加载仍约207秒，只是转移了等待。若总加载显著下降，再用同一模型与prompt做无预读／预读交替复测，控制设备负载；不使用全局drop_caches，也不把本轮数字直接当作稳定性能验收。若无改善，保留分段结果，再选择其他加载方案。

恢复仅读取原请求，不能给`--recover`追加`--draft-preread`来改变已执行策略。数值正确性、0接受率和不同请求token差异仍未解决；本实验不重编模型。

主机验证：190项准备／回收测试、14项真实C++＋fake SDK循环场景及1组native生命周期测试通过。[验证记录](../../specs/board-resident-preread-validation.json)包含源码身份与测试范围；未连接用户DDK或板端，实机预读收益待回传。
