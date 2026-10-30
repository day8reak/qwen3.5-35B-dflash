# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

原 target prefill64、独立 decode1、真实 feature→完整六层 draft 单步，以及同模型 decode 的全部80状态字节链已执行。prefill-v2 名字探针和 path-v2 的89输入／82输出文件绑定均通过。完整 DFlash 多轮仍未执行；当前采用 decode1 串行 verify，主机调度器已补齐 target-next anchor、逐 token 接受／纠正和全接受 bonus。

实际 AIR 已找到唯一控制路径：`arg1238_1` 直接进入 `IndexByTensor_40.indices0`，x 的导出 `_meta` 为 `[1,64,2048]`，`indices_mask=[0,1]`。当前原文件值64；按已核原 Python 源码的当前序列最后行规则，独立试验改为63。该试验不声称私有 IndexByTensor 的越界行为或数值根因。

以下 A、B、C 的准备可以并行。A 通过后再运行 C 的板端调用。模型调用仍按现有共享锁排队；不需要重跑 metadata 探针或旧 decode 恢复。

先更新一次：

```bash
git -C "$REPO_DIR" pull --ff-only
```

## A：原始 embedding 行对应检查

先把原权重机器 `/home/l00886453/models/Qwen3.5-35B-A3B_quant/` 中的 `qwen3_5_moe_embdedding_weight.bin`、`qwen3_5_moe_embdedding_scale.bin` 复制到 DDK 的 `$AI_RUN_DIR/original-embedding-tables/`，保留原始字节和文件名。两表合计509,552,640字节，约486MiB。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/inspect_original_embedding_inputs.py" \
  --weight-table "$AI_RUN_DIR/original-embedding-tables/qwen3_5_moe_embdedding_weight.bin" \
  --scale-table "$AI_RUN_DIR/original-embedding-tables/qwen3_5_moe_embdedding_scale.bin" \
  --input-dir /home/w00949577/run_35B/chunksize_64_data \
  --sequence-length 64 \
  --output "$AI_RUN_DIR/prefill64-embedding-row-check"
```

这一步只读必要表行及三个原输入文件，不生成替换输入。预期 `ORIGINAL_EMBEDDING_INPUT_ROWS_MATCHED`、64个位置全部匹配；若 `MISMATCH`，贴回位置摘要，先解决输入对应再做下一次板端 prefill。默认不对整表反复哈希；API 支持任意后续 token 的原 INT8 行与独立 FP32 scale 查表。

## B：真正导出 S128 完整六层 draft AIR

```bash
python3 "$REPO_DIR/targets/cpro/scripts/export_draft_capacity.py" \
  --job "$AI_RUN_DIR/full-draft" \
  --sequence-length 128 \
  --prompt-length 64 \
  --max-new-tokens 32 \
  --output "$AI_RUN_DIR/full-draft-capacity128-air"
```

这会构建 GE serializer、重建真实S128 graph/recipe并导出新AIR；直接只读引用原 prepared 权重，零复制、零链接，不改旧manifest或S64模型。只新增小源码、exporter、AIR和日志，预留384MiB。成功应为 `DRAFT_CAPACITY_AIR_EXPORTED`，并明确 `compilation_ready:false`：它尚不是新的可部署OMC。后续OMG需要显式只读输入视图和空间预算；目前不启动可能额外占约8.46GB加临时文件的普通编译路径。

## C：仅修改 output_pos 的独立 prefill 实验

先生成独立输入，可与 A、B 并行：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/prepare_target_prefill_last_row_trial.py" \
  --execution "$AI_RUN_DIR/target-prefill64-smoke-v3" \
  --named-runtime-probe "$AI_RUN_DIR/target-prefill64-runtime-io-v2" \
  --output "$AI_RUN_DIR/target-prefill64-last-row63-trial"
```

成功应为 `TARGET_PREFILL_LAST_ROW_TRIAL_INPUTS_PREPARED`。它复制13份实际 target 源文件，仅把 output_pos 的一个INT32字改为63；另外复制 input_ids 作为辅助第14文件，不算target第90输入。原输入、旧执行、模型和权重均不改。

A 的64行全部匹配且 C 准备成功后，运行一次原 prefill：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_target_prefill_smoke.py" \
  --job "$AI_RUN_DIR/target-pytorch-prefill64-tmpdirfix" \
  --runtime-probe "$AI_RUN_DIR/target-prefill64-runtime-io-v2" \
  --input-dir "$AI_RUN_DIR/target-prefill64-last-row63-trial/inputs" \
  --output "$AI_RUN_DIR/target-prefill64-last-row63-smoke" \
  --apply
```

沿用原40层S64模型和已验证 V2 工具路径。已绑定 runtime proof 的 prefill 现在先核 owned board cache 的实际SHA：匹配的 sidecar 不访问本地旧stat；需要 SEND 时仍严格验证本地文件。小 primary OMC 的历史ctime漂移只在完整SHA相同后接纳，并维持本次当前身份检查。缓存或连接不确定时保留证据和锁，不自动重跑模型。

A、B、C 各自完成就贴回摘要，不需要等全部完成。所有输出目录都需是新目录。

## 当前仍在处理的接续项

path-v2 观察到1126种共同参数内容、1123种唯一内容；两图各80个源状态owner完整，但通用上下文追踪仅30个候选，另50个触及深度限制。正在按卷积x1、recurrent beta、K/V updates分别匹配显式层参数名、内容和真实端口；尚未准备跨模型反馈。源状态对应还需接到各自 compiled/runtime/file 绑定，不能用RefData编号或相同形状替代。

标准 DFlash 首轮需保留全部64个prompt feature，并以target下一token作位置64的anchor；旧S64 draft机械单步用了prompt末token220@63，不能当作标准首轮。S128导出保持完整六层和原权重，为后续32-token预算留容量；仍需真正编译、设备执行和89输入target循环adapter联调。最终需要多轮接受／拒绝后继续生成与普通基线对照，再验证精度与性能。
