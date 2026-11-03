# 当前阶段测试命令

当前目标是 C16／cap192／K15 常驻同步 Process 的预热后平均耗时 ≤50 ms，包含内部调度、同步和搬运，排除加载与初始 catchup。用户已接受上一版最差 hidden 相对 L2 2.61%、Top1 7/172 不一致，先优化性能；原 NUMERICAL_FAILED 和 golden 保留。

2026-11-03，第一轮优化代码已接入 PyTorch Draft，选项如下。原有命令默认 baseline；本轮推荐先测 packed-spatial。

| --optimization | 内容 |
|---|---|
| baseline | 原独立投影、KV 复制和 batch Conv |
| packed | context/block 合并 K/V 投影、gate/up 合并、GQA 折叠 query、共享 RoPE |
| packed-spatial | packed 加上 Conv 空间排布；C16 从 [16,1024,4,4] 改为 [1,1024,16,16]、stride4 |

三档均保留完整点积、全词表、17输入／14输出与原 KV 写入区间。合并权重按原 K/V、gate/up 顺序核对每段字节；原始 prepared 不改。真实尺寸主机图捕获：线性投影55→31、sin/cos各18→2、消除12次KV头复制，仍为12个attention MatMul、6个FP32 Softmax和32个RMS。440次缩小宽度六层反馈对照通过原0.01／0.01门槛，427次逐位一致，最大绝对差0.005859375，Top1全部一致，未写KV逐位保持。完整16384→2048投影的C16/C64主机对照逐位一致。主机结果不代表所选DDK或设备结果。

本地缺少所选 CPro DDK，未编译 OMC、未访问设备、未测得提速。上一版仍是5个子图，单次厂商模型推理627.647ms、算子加和134.940ms，差额492.707ms尚未归因；新分区数量以本次编译为准。Conv69.038ms与CPU ArgMax16.199ms仍是需要实测的热点。NPU Top1与运行时边界优化尚未实现。

## 导出与编译

在现有 custom_pypto 环境运行。REPO_DIR 指向本仓库，AI_RUN_DIR 指向现有运行根目录；CPRO_DDK_ROOT／CPRO_DDK_ID 沿用已选 X90 DDK（100.520.020.010）。需要已有 draft/source/prepared 和 runtime/runtime.json。本候选使用固定测试槽位，保留当前 draft/air、pytorch-omc 及 pytorch-result 基线。

```bash
git -C "$REPO_DIR" pull --ff-only

python3 "$REPO_DIR/targets/cpro/scripts/export_incremental_draft.py" \
  --frontend pytorch --optimization packed-spatial \
  --prepared "$AI_RUN_DIR/draft/source/prepared" \
  --context-rows 16 --cache-capacity 192 --max-draft-tokens 15 \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-air" --replace

python3 "$REPO_DIR/targets/cpro/scripts/compile_incremental_draft.py" \
  --air-export "$AI_RUN_DIR/draft/tests/pytorch-optimized-air" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-omc" --replace
```

导出成功应显示 PYTORCH_DRAFT_AIR_EXPORTED、optimization.policy=packed-spatial、17输入／14输出；编译成功应显示 PYTORCH_DRAFT_OMC_COMPILED，并报告实际 graphop_counts。若编译失败，保留该槽位的 manifest 和 compiler 日志；不要据主机捕获的节点数推算板端子图或速度。

## 执行并回收

复用现有 native runner。先启动，收到启动确认后执行回收；回收不重新运行模型。

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --config "$AI_RUN_DIR/runtime/runtime.json" \
  --compiled "$AI_RUN_DIR/draft/tests/pytorch-optimized-omc" \
  --source "$AI_RUN_DIR/draft/source/prepared" \
  --air-export "$AI_RUN_DIR/draft/tests/pytorch-optimized-air" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-run" --apply

python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --recover "$AI_RUN_DIR/draft/tests/pytorch-optimized-run" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-result"
```

检查所有调用完成、hidden／新增KV相对L2、Top1差异、未写区与重复稳定性，并与保留的 pytorch-result 对照。新候选不会因旧版本已被接受而自动通过数值验收。native_api_latency 是混合精度用例的同步API统计，不能当作固定输入3次预热＋10次测量的50ms达标成绩。

## 采集同一候选的厂商 CSV

精度回收完成后运行。allow-numerical-failure 仅允许对原门槛失败结果采集性能，不会修改精度状态。

```bash
python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --accuracy "$AI_RUN_DIR/draft/tests/pytorch-optimized-result" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-profile-run" \
  --allow-numerical-failure --apply

python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --recover "$AI_RUN_DIR/draft/tests/pytorch-optimized-profile-run" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-optimized-profile-result"
```

回传编译 graphop_counts、精度摘要和 *_model.csv／*_op.csv。重点对照完整 inference、Conv、末尾 MatMul、ArgMax，以及新分区边界。packed-spatial 若收益不明确，下一轮在相同固定候选槽位改用 --optimization packed，单独辨别 Conv 排布影响；先回收已启动任务再替换候选。

工程估计先按5～7个工作日安排验证窗口，假设每天有1～2轮编译／板端反馈；这不是50ms达标承诺。未解释的运行时耗时差额是主要工期风险。首轮实测后重新评估。完整基线与历史分析见[当前结果](results.md)。
