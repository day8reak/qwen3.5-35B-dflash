# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A：用户已完成 `target-decode1-continuation-recovery`。同一次原工具执行 PID 10457 的 82 个输出已恢复，全部 80 份父输出与子输入的字节链通过，两把锁均已释放或不存在。恢复没有重跑模型、上传输入或权重。原 prefill64、独立 decode1、真实 feature → 完整六层 draft 单步也已执行。

完整 DFlash 尚未跑通。当前路线采用 `decode1` 逐 token 串行 verify，先验证多轮接受／拒绝／状态提交；批量 verify 和性能优化后续处理。仍有三个联调关口：真实 prefill→decode 与普通生成基线、完整 draft→串行 verify 单轮、连续多轮与拒绝后继续生成对照。它们不是保证三次命令即可完成。

主机调度器已补上显式 `target_next_anchor=True`：先提交 target 首 token；每个候选先与当前 target 预测比较，再执行接受 token 或纠正 token；全接受且预算允许时提交 bonus。被拒候选不进入 target 状态。65 项调度测试及独立审查通过，包含多轮与普通 oracle 对照、拒绝位置、EOS、预算、状态链和旧行为兼容。实际 89 输入 target／完整六层 draft 的设备 adapter 尚未接通，不能将这些主机测试称为板端完整流程。

下一步先补齐原 prefill 的名字和状态／logits 证据，复用已有编译产物。在 DDK 主机执行：

```bash
git -C "$REPO_DIR" pull --ff-only

python3 "$REPO_DIR/targets/cpro/scripts/inspect_target_runtime_io.py" \
  --job "$AI_RUN_DIR/target-pytorch-prefill64-tmpdirfix" \
  --tool-inspection "$AI_RUN_DIR/target-board-tool-inspection-writable" \
  --lock-wait-seconds 1800 \
  --summary-only \
  --output "$AI_RUN_DIR/target-prefill64-runtime-io-v2"

python3 "$REPO_DIR/targets/cpro/scripts/inspect_target_prefill_decode_path.py" \
  --prefill-job "$AI_RUN_DIR/target-pytorch-prefill64-tmpdirfix" \
  --decode-job "$AI_RUN_DIR/target-pytorch-decode1-compile" \
  --prefill-execution "$AI_RUN_DIR/target-prefill64-smoke-v3" \
  --prefill-runtime-probe "$AI_RUN_DIR/target-prefill64-runtime-io-v2" \
  --source-torchair-pb2 "$AI_RUN_DIR/target-decode1-row-response/ge_ir_by_protoc_3_19_pb2.py" \
  --output "$AI_RUN_DIR/target-prefill-decode-path"
```

第一项只恢复小 OMC 的描述信息，记录当前 SDK 已支持的 tensorName；不编译模型、不执行推理、不初始化 manager 或权重。完整数组保留在新目录，终端只打印摘要。已有目录需换新输出名，不能覆盖旧证据。

第二项只在主机检查三类现有证据：同一 prefill 模型的新名字与旧执行全部 89 输入／82 输出的对应、实际 logits Gather 的索引／轴／形状路径，以及两次导出状态更新边的参数上下文。请贴回第二项的摘要；若第一项失败，也贴回它的错误。

成功收集后仍返回 `PREFILL_DECODE_PATH_EVIDENCE_OBSERVED_WITH_GAPS`，不会生成反馈输入。跨导出的参数上下文匹配只作候选；同名 RefData、相同形状或 copy 编号均不直接证明同一层状态。旧 prefill 的 `output_pos=64` 必须结合实际索引路径判断，尚不能把其 argmax 0 当作正确的下一 token。还需证明 64 个原始 token 与实际 embedding／scale 的对应，并准备正确下一 token 的原始表行。

之前 draft 单步的 anchor 是 prompt 最末 token 220、位置 63，因此它证明真实 feature 能进入完整六层 draft。标准首轮需要保留全部 64 个 prompt feature，并以 target 下一 token 作位置 64 的 anchor；现有 S64 前缀 ABI 容量不足。不能简单替换 `ids[63]`、截断上下文或重定位 RoPE。需为生成预算扩容或补增量 ABI，保持完整六层拓扑与权重。当前只有 target 的 S64／S1 及 draft S64 已有实际执行证据。

同模型 decode 续跑的控制值仍是明确实验：绝对位置 256→257、mask 多开放一项、本地 query 控制沿用已执行试验值。字节链通过不等于这些控制语义、数值精度或 prefill64 历史连续性通过。后续串行 verify 必须处理正确 anchor、原始 embedding 查表、80 状态提交、被拒绝候选不污染状态，以及普通基线对照。

多轮动态查表还需要两张原始 embedding 表，共 509,552,640 字节（约 486 MiB）。源在原权重机器 `/home/l00886453/models/Qwen3.5-35B-A3B_quant/` 下的 `qwen3_5_moe_embdedding_weight.bin` 和 `qwen3_5_moe_embdedding_scale.bin`；执行机可放到 `$AI_RUN_DIR/original-embedding-tables/`，保留原文件名和原始字节。当前 query269／next0 行包仅覆盖那一次已执行 decode，不能提供任意后续 token 的查表。

B：55 档特征 AIR 保留备份，本轮无待执行命令。
