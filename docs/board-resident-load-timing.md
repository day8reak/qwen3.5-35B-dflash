# draft加载阶段计时

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
| SET_MODEL_BUFFER | SetModelBuffer及返回buffer大小检查 |
| MODEL_LOAD | 同步Load及状态检查 |
| IO_DESCRIPTORS | IO描述符查询、一次性打印及数量检查 |
| TENSOR_SETUP | 五个tensor分配、加载时重叠检查、初始化与ready记录 |

每段含wall_us和completed；失败段completed=false，保留原SDK错误，不重试、不降级。阶段开始／结束各写一行日志，便于定位异常中断。无逐token采样，不恢复全量哈希或中间tensor文件。原模型、加载顺序、缓冲区持有期、推理与关闭路径不变。

可读时同时记录本进程minor_faults、major_faults、read_bytes、read_chars增量。缺失／计数回退记null，读取失败不影响推理。它们不覆盖SDK其他进程或NPU，不是系统总I/O；read_chars还包含探针自身的小量proc读取，不能用作精确模型读取字节数。阶段墙钟排除计数读取和两行日志，原draft_load_ms仍含整个构造，阶段和未必等于总数。mmap阶段快也不意味着后续没有按需读取成本。

## DDK主机下一次完整请求

沿用原prompt与生成配置，只编新版runner，默认dump=false。使用独立目录：

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
  --resident --apply --detach \
  --output "$AI_RUN_DIR/dflash-board-resident-loadtiming8-run"
```

回收同一请求：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_board_serial_dflash.py" \
  --recover "$AI_RUN_DIR/dflash-board-resident-loadtiming8-run" \
  --output "$AI_RUN_DIR/dflash-board-resident-loadtiming8-recovery"
```

贴回完整摘要，重点看load_profile里哪段占用时间、对应计数是否可用，以及总加载时间是否再次出现203秒。一次新运行可能不复现同样的慢加载；即使恢复到8秒，也不能直接断言问题已修复。随后按阶段结果决定是否需要冷／热缓存或资源压力对照，不提前增加整表预读、修改模型或改变加载API。
