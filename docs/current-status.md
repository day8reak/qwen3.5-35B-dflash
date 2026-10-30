# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A：完整 target 的 prefill64、独立 decode1，以及真实 target feature → 完整六层 draft 单步都已执行。decode 的 82 个输出已按日志、SHA 和字节大小绑定；logits、feature、80 个状态均无 NaN/Inf。feature 范围达到 ±65504，logits argmax 为 0，尚未通过精度或连续状态验证。

状态检查的失败已定位：实际编译报告为 10,306,827 字节，超过检查器原先的 8 MiB JSON 限制。已统一为既有 decode 规划器的 32 MiB，并为读取失败补充文件角色、路径、实际大小和上限。AIR/小 OMC 的 64 MiB 上限不变。

拉取更新后，只重跑主机状态检查，使用新输出目录；复用已通过的输出检查，不重跑模型或上传权重。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/inspect_target_decode_state_lineage.py" \
  --job "$AI_RUN_DIR/target-pytorch-decode1-compile" \
  --runtime-probe "$AI_RUN_DIR/target-decode1-runtime-io-v2" \
  --output-inspection "$AI_RUN_DIR/target-decode1-output-inspection" \
  --output "$AI_RUN_DIR/target-decode1-state-lineage-v2"
```

完整状态绑定时为 `ALL80_SAME_MODEL_STATE_BINDINGS_OBSERVED`、`counts.bound_state_files: 80`；缺少图或运行时字段时返回 `DECODE_STATE_LINEAGE_OBSERVED_WITH_GAPS` 和具体原因。它检查同一个 decode 模型的状态归属，不证明 prefill64→decode1 接续。

embedding 的第二处失败也已定位：原 `input_ids.bin` 为 1024 字节，包含 256 个 INT32 ID，旧检查只接受一个 ID。现已提供显式 `--match-source-id-candidates`：逐个核对候选 ID 的原始 INT8 行及独立 FP32 scale，只有唯一 token ID 匹配实际执行输入才建立绑定；同一 ID 的重复位置全部保留，无匹配或多个不同 ID 匹配均不生成下一行。

当前 YAML 存在，但其两张原表路径在当前主机上不存在。已确认原表仍在原权重机器的 `/home/l00886453/models/Qwen3.5-35B-A3B_quant`，正在准备小请求包和独立行提取脚本。暂不重复当前已知缺表的 embedding 命令；候选匹配功能已准备好，最多接收 1024 个 ID，不扫描整个词表或重建 embedding。

状态归属通过后，下一步是 80 个状态分别回填的连续 decode，再接已有逐 token verify 和 DFlash 循环。当前 draft 长度 64 已被 64-token 前缀占满，还需明确较短真实 prompt 或更大档位。完整流程、精度和加速比仍未通过。

B：55 档特征 AIR 保留备份，本轮无待执行命令。
