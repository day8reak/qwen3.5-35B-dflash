# 完整流程执行

复用当前已编译的原Prefill64、Decode1、完整Draft及80状态映射。环境与离线词表就绪后，执行下面两段：板端完成文字编码→Prefill一次→常驻Draft/Decode循环→文字解码，服务器只下发和最终回收。

## 1. 输入文字并启动

新请求使用新目录；已启动的请求直接回收。默认关闭中间dump，开启Draft预读。模板后的prompt＋生成预算须≤64，当前固定生成8个token，不按EOS提前停止。

```bash
export TOKENIZER_DIR="${TOKENIZER_DIR:-$AI_RUN_DIR/tokenizer-offline/source}"
export RUN_DIR="$AI_RUN_DIR/dflash-text-run"
export RECOVERY_DIR="$AI_RUN_DIR/dflash-text-recovery"
export PROMPT='你好，请介绍一下自己。'

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

## 2. 回收结果

等待同一请求完成后执行。结果包含生成文字、token ID、接受率、Prefill/Decode/Draft时延和加载耗时。另开终端时先恢复相同环境及下列路径。

```bash
export RUN_DIR="$AI_RUN_DIR/dflash-text-run"
export RECOVERY_DIR="$AI_RUN_DIR/dflash-text-recovery"

python3 "$REPO_DIR/targets/cpro/scripts/run_board_serial_dflash.py" \
  --recover "$RUN_DIR" \
  --output "$RECOVERY_DIR"
```
