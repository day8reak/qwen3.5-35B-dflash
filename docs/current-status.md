# 当前阶段：Draft 性能优化

2026-11-05：用户报告服务器剩余空间6.4G，恢复Draft性能工作；精度排查继续暂缓。当前RoPE候选为C16／cap192／K15，OMC SHA `b43ae9cfcd50999277829a21adc24099170d4a7c4856b70729385b59d362a828`。最近同步API四组均值约65.94～66.00ms，目标仍为常驻预热后proposal均值≤50ms。原数值结果FAILED，hidden相对L2约1.16%～1.77%，新增KV最差约0.61%，两模式各83/86个Top1匹配。[完整结果](results.md)。

## 优先检查的优化方向

| 方向 | 已有依据 | 下一步验证 |
|---|---|---|
| 零新增行的proposal专用路径 | `valid_rows=0` 时写入区为空，present KV等于past KV；源码仍计算上下文空间打包、特征投影和新KV | 当前模型case18 profiling；再评估移除这条无效计算分支 |
| 仅更新KV的catchup路径 | `proposal_count=0` 时调用方不需要候选hidden／Top1，统一图仍计算六层候选分支和head | 保留同时追加与proposal的一般路径，另行做运行时路由和ABI适配 |
| 全词表head与布局 | 旧同系列模型MatMul_30约19.965ms；入口Conv2D约8.753ms、Transpose_1约5.284ms | 核实当前编译热点与形状后选择改图；保留完整词表和原舍入边界 |

旧profiling中的StableArgMax仅0.415ms，暂不优先优化。旧表来自RoPE修复前的一次新进程调用，不能直接当作当前模型的耗时分解。算子duration之和也不等于实际关键路径；尚不能承诺某个改动能省多少ms。

当前head逻辑权重约1.017GB，logits约7.45MB。仅融合argmax以免写出logits，仍需访问完整head权重；需要实测，不能从减少一个中间张量推断大幅加速。暂无匹配的X90实测能力画像，不作硬件带宽极限或理论剩余空间结论。

## 服务器执行一次当前proposal profiling

复用现有OMC和板端缓存，不编译新模型、不复制大权重、不重做精度探针。选择最新完整回放中的case18：start192、valid0、15个proposal、参考KV输入。该调用明确用于定位热点，不是3次预热＋10次常驻性能验收。

```bash
git -C "$REPO_DIR" pull --ff-only
export ROPE_DIR="$AI_RUN_DIR/draft/tests/top1-integration/rope-fp32"
export PROFILE_DIR="$AI_RUN_DIR/draft/tests/rope-zero-append-profile"

python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --accuracy "$ROPE_DIR/kv-reproduction/result" \
  --case-index 18 --allow-numerical-failure \
  --modelmanager-itf V1 --output "$PROFILE_DIR-run" --apply
```

启动后回收同一个请求：

```bash
python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --recover "$PROFILE_DIR-run" --output "$PROFILE_DIR-result"
```

若返回 `DRAFT_OPERATOR_PROFILE_PENDING`，稍后只重复回收命令，不重复启动。完成后回传摘要与对应的原始 `*_op.csv`、`*_model.csv` 内容；文件位置由 `csv_files` 给出。原始CSV可能包含服务历史，必须结合本次PID／模型记录确认归属。

本次新增 `--case-index` 只选择已保留的完整输入，支持纯catchup和零新增行proposal；默认行为保持case2。9项主机检查通过，未本地执行设备。没有修改模型图或权重，也没有新增性能改善结论。整行重复仍记录为上轮未复现、原因未查明。
