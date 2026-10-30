# 当前执行计划

<a id="用户现在执行什么"></a>

按用户最新要求，当前只验收完整模型的真实多轮 DFlash 执行，精度随后收敛。主线复用已有40层 target decode1 OMC 和完整六层 `full-draft/residual-fix/bundle`，不重新导出／编译模型。

## 本次固定范围

- 新请求取原 input_ids 文件的前8个 token 作短提示词，生成8个新 token；完整历史最多16行，始终装得下旧 draft S64。
- 同一个 target decode1 从80个零状态开始，逐 token 读入提示词。每次通过原 INT8 embedding 表和独立 FP32 scale 查当前 token；不使用旧64-token prefill 的状态或裁剪其输出。
- target 首个预测先提交为 anchor；draft 接收完整真实 feature 历史，右侧补零到64，原图排除当前 anchor 的 feature，每轮仍生成完整15个候选。
- host 每轮先验证最多2个候选：匹配则提交，首个不匹配处提交 target 纠正；考虑的候选全部匹配时可提交 bonus。只有提交的 token 进入 target，因此串行路径不用回滚投机状态。
- 每次记录全部80状态接续、原始输入／输出、native PID／日志和哈希；至少观察两次真实完整 draft 调用。EOS 在这次固定预算实验中关闭。

原模型权重、层数和词表均保持。短提示词是明确的新请求，不冒称原64-token请求的等价接续。控制值 `output_pos=0 / allQLen=1 / effective_length=1 / conv_state_pos=[1,0×127]`、绝对位置与 mask 仍是声明过的实验配方，执行成功不代表数值语义通过。

## 现在运行

embedding 两表已由用户确认复制完成；入口会核对完整文件大小，并直接读取每个实际 token 的原始行，不再往返抽行。一次命令先做 host 预检并保存计划，再执行循环：

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/run_serial_dflash_smoke.py" \
  --decode-job "$AI_RUN_DIR/target-pytorch-decode1-compile" \
  --runtime-probe "$AI_RUN_DIR/target-decode1-runtime-io-v2" \
  --state-lineage "$AI_RUN_DIR/target-decode1-state-lineage-v7" \
  --source-torchair-pb2 "$AI_RUN_DIR/target-decode1-row-response/ge_ir_by_protoc_3_19_pb2.py" \
  --draft-bundle "$AI_RUN_DIR/full-draft/residual-fix/bundle" \
  --weight-table "$AI_RUN_DIR/original-embedding-tables/qwen3_5_moe_embdedding_weight.bin" \
  --scale-table "$AI_RUN_DIR/original-embedding-tables/qwen3_5_moe_embdedding_scale.bin" \
  --prompt-input-ids /home/w00949577/run_35B/chunksize_64_data/input_ids.bin \
  --prompt-length 8 --max-new-tokens 8 --max-proposals-per-round 2 \
  --output "$AI_RUN_DIR/dflash-serial8-run" \
  --apply
```

输出目录必须是新目录。去掉 `--apply` 可只准备，实际执行另用新目录；没有额外必做的 prepare／inspect 命令。runtime probe 和旧 lineage 的当前字节仍会在入口核对，读取原权重表行时保留本次身份。模型命中 SHA 所属板端缓存时复用；target 的19GB sidecar 不反复本地哈希／复制。draft 模型只在 session 初始化绑定一次，runner 可在本次 session 复用；必要的 runner 构建不是模型重编译。

成功目标是 `DFLASH_SERIAL_LOOP_EXECUTED_UNVALIDATED`：至少两轮 draft、8个新 token、16次真实 target decode1 调用，全部步骤实际输出取回。`counts.target_calls` 为协议观察次数（串行 prompt 汇总算1次）；`native_executions.target.native_calls` 才是包含8次提示词输入的真实调用数。接受率可以为0，乱码不阻塞本里程碑。接受／拒绝按真实输出记录，不能为了凑分支而伪造候选或报告。

如果某步失败，入口停止，保留同一 attempt、日志及锁状态，不自动重复模型调用。先看失败步骤的具体错误；不要清锁、重复整个命令或覆盖旧结果。

## 已完成和后续

用户实机已执行原 prefill64、独立 decode1、80状态同模型接续及真实 feature→完整 draft 单步。最新 row63 prefill PID28669 成功取回82输出：feature SHA 与旧 row64 完全相同；logits SHA 改变，argmax 从0变为6。这是单变量输出观察，尚未判为精度修复。

今天闭环执行之后再处理：原 prefill64→decode1 的跨模型映射／普通基线、控制和 embedding 对齐精度、增量 draft KV、block verify 与状态回滚、常驻模型和实测加速。S128 全前缀导出和新增缓存 draft 编译均不在本次前置条件。已完成的缓存图源码／主机等价测试保留为后续优化，不启动新模型编译。
