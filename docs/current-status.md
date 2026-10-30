# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

原 target prefill64、独立 decode1、真实 feature→完整六层 draft 单步，以及同模型 decode 的全部80状态字节链已执行。prefill-v2 名字探针和 path-v2 的89输入／82输出文件绑定均通过。完整 DFlash 多轮仍未执行；当前采用 decode1 串行 verify，主机调度器已补齐 target-next anchor、逐 token 接受／纠正和全接受 bonus。

实际 AIR 已找到唯一控制路径：`arg1238_1` 直接进入 `IndexByTensor_40.indices0`，x 的导出 `_meta` 为 `[1,64,2048]`，`indices_mask=[0,1]`。当前原文件值64；按已核原 Python 源码的当前序列最后行规则，独立试验改为63。该试验不声称私有 IndexByTensor 的越界行为或数值根因。

A 的表传输／行检查与 C 的单变量板端实验可以并行；B 的全前缀 S128 导出暂缓，优先参考4B分支的增量 draft 缓存。C 的结果在 A 通过前仅作实验观察，不作为正确生成基线。模型调用仍按现有共享锁排队；不需要重跑 metadata 探针或旧 decode 恢复。

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
  --output "$AI_RUN_DIR/prefill64-embedding-row-check-v2"
```

用户首次检查在读表行前停止：DDK上的weight文件只有786,432字节，完整应为508,559,360字节；scale完整应为993,280字节。等待复制完成后，在原权重机器和DDK分别核对这两个文件大小／SHA，再用新输出目录 `prefill64-embedding-row-check-v2` 重跑，保留首次失败目录。row63 trial已准备成功，无需重复准备。

这一步只读必要表行及三个原输入文件，不生成替换输入。预期 `ORIGINAL_EMBEDDING_INPUT_ROWS_MATCHED`、64个位置全部匹配；若 `MISMATCH`，贴回位置摘要；单变量 prefill 实验可以继续观察，但输入修复前不接正确生成基线。默认不对整表反复哈希；API 支持任意后续 token 的原 INT8 行与独立 FP32 scale 查表。

## B：改为增量 draft，暂缓全前缀 S128 导出

用户指定的 [4B feature/gdr-chunk-verify](https://github.com/day8reak/qwen3.5-4B-dflash/tree/feature/gdr-chunk-verify) 已按当前commit `90186732d915e8b058b2a053d205991d805eae76` 只读核查。它将新增context行数、提案block和总KV容量分开：每次输入C16/C64新特征、绝对start_position、valid_rows、anchor、proposal_count及每层K/V；query block固定16行。历史留在KV，不必每轮输入整个前缀。[实际图实现](https://github.com/day8reak/qwen3.5-4B-dflash/blob/90186732d915e8b058b2a053d205991d805eae76/framework/python/qwen35_dflash/ascend310p/incremental.py#L643)。

因此不把S128全前缀AIR作为DFlash必做步骤。之前扩容针对本仓库现有三输入、prefix-recompute的35B draft OMC；它没有缓存／位置ABI，不能直接靠运行参数变成增量模型。下一步按参考接口适配实际35B完整六层、8taps/F16384、原权重与舍入边界，重新导出增量draft AIR/OMC。4B是Ascend310P/TorchAir/ATC/AscendCL，不能直接复用其二进制到Kirin HIAI。

总历史KV容量仍要覆盖prompt和生成预算；它与单次C16/C64输入档位不同。4B参考将logical capacity向64对齐并增加64行物理guard。35B的新buffer容量需明确锁定。采用当前eager serial anchor时，每次只向draft缓存追加当前anchor之前尚未同步的已提交特征，排除最新correction/bonus的feature。

`export_draft_capacity.py`已发布并通过主机测试，保留为可选全前缀对照工具；当前先不运行它。旧成功S64 artifact保持保留。4B的FP16六层checkpoint与W4/W8五层checkpoint也不混用，35B继续使用实际六层模型。

## C：仅修改 output_pos 的独立 prefill 实验

该目录已准备成功，不用重做。下面准备命令仅保留作记录：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/prepare_target_prefill_last_row_trial.py" \
  --execution "$AI_RUN_DIR/target-prefill64-smoke-v3" \
  --named-runtime-probe "$AI_RUN_DIR/target-prefill64-runtime-io-v2" \
  --output "$AI_RUN_DIR/target-prefill64-last-row63-trial"
```

成功应为 `TARGET_PREFILL_LAST_ROW_TRIAL_INPUTS_PREPARED`。它复制13份实际 target 源文件，仅把 output_pos 的一个INT32字改为63；另外复制 input_ids 作为辅助第14文件，不算target第90输入。原输入、旧执行、模型和权重均不改。

C 已准备成功，现在运行一次原 prefill 单变量实验，可与 A 的传输／检查并行：

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

path-v2 观察到1126种共同参数内容、1123种唯一内容；两图各80个源状态owner完整，但通用上下文追踪仅30个候选，另50个触及深度限制。新模型专用参数命名合同已按卷积x1、recurrent beta、K/V updates分别追踪完整W_q/scale对，主机测试覆盖80个重排source owner；实际两图的新合同结果待收集。它只按已保留的层参数文件名、原始属性、内容pin和实际端口配对，不声称Python FQN命名attestation、数值dtype/extent或完整runtime反馈已证明。源状态对应还需接到各自 compiled/runtime/file 绑定，不能用RefData编号或相同形状替代。

标准 DFlash 首轮需保留全部64个prompt feature，并以target下一token作位置64的anchor；旧S64 draft机械单步用了prompt末token220@63，不能当作标准首轮。当前优先适配增量缓存ABI，保持完整六层和原权重；仍需新artifact导出、编译、设备执行和89输入target循环adapter联调。最终需要多轮接受／拒绝后继续生成与普通基线对照，再验证精度与性能。
