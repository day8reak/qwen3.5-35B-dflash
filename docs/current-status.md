# 当前阶段测试命令

<a id="用户现在执行什么"></a>

本轮验证“离线原词表＋文字输入输出＋Draft预读”在同一次板端请求中完成。原token输入、常驻与预读已实机跑通；本轮保持8个生成token、每轮最多验证2个候选，不重编OMC。

## 1. 启动本轮测试

沿用已加载的项目环境和准备好的本地词表。若此请求已经启动，直接执行第2段；新一轮测试改用新的运行和回收目录。

```bash
export TOKENIZER_DIR="${TOKENIZER_DIR:-$AI_RUN_DIR/tokenizer-offline/source}"
export RUN_DIR="$AI_RUN_DIR/dflash-board-text-preread8-run"
export RECOVERY_DIR="$AI_RUN_DIR/dflash-board-text-preread8-recovery"
export PROMPT='你好，请介绍一下自己。'
python3 -c 'import tokenizers; print(tokenizers.__version__)'

python3 "$REPO_DIR/targets/cpro/scripts/run_board_serial_dflash.py" \
  --decode-job "$AI_RUN_DIR/target-pytorch-decode1-compile" \
  --runtime-probe "$AI_RUN_DIR/target-decode1-runtime-io-v2" \
  --state-lineage "$AI_RUN_DIR/target-decode1-state-lineage-v7" \
  --source-torchair-pb2 "$AI_RUN_DIR/target-decode1-row-response/ge_ir_by_protoc_3_19_pb2.py" \
  --draft-bundle "$AI_RUN_DIR/full-draft/residual-fix/bundle" \
  --weight-table "$AI_RUN_DIR/original-embedding-tables/qwen3_5_moe_embdedding_weight.bin" \
  --scale-table "$AI_RUN_DIR/original-embedding-tables/qwen3_5_moe_embdedding_scale.bin" \
  --prefill-feedback "$AI_RUN_DIR/target-prefill64-decode1-feedback-plan-v2" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --text "$PROMPT" \
  --max-new-tokens 8 --max-proposals-per-round 2 \
  --resident --draft-preread --apply --detach \
  --output "$RUN_DIR"
```

## 2. 回收并贴出摘要

本轮关注板端文字编码／解码、一次Prefill、后续Draft/Decode调用，以及预读和三阶段时延。接受率与语言质量继续如实记录，本轮完成不代表精度通过。

```bash
export RUN_DIR="$AI_RUN_DIR/dflash-board-text-preread8-run"
export RECOVERY_DIR="$AI_RUN_DIR/dflash-board-text-preread8-recovery"

python3 "$REPO_DIR/targets/cpro/scripts/run_board_serial_dflash.py" \
  --recover "$RUN_DIR" \
  --output "$RECOVERY_DIR"
```
