# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A：将原机 `/home/w00949577/qwen35-target-pytorch-run/target-pytorch/canonical-export/` 下的 `export.air`、`inventory/inventory.json`、`result.json` 手动上传到 DDK 机 `$AI_RUN_DIR/target-pytorch-transfer/`，分别保存为 `export.air`、`inventory.json`、`result.json`，再在 DDK 窗口复用权重。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/stage_target_export_reuse.py" \
  --export-receipt "$AI_RUN_DIR/target-pytorch-transfer/result.json" \
  --inventory "$AI_RUN_DIR/target-pytorch-transfer/inventory.json" \
  --air "$AI_RUN_DIR/target-pytorch-transfer/export.air" \
  --existing-export /home/l00886453/qwen3_5_moe/glm_export_air_cpro_40block_55_quant \
  --output "$AI_RUN_DIR/target-pytorch-import" --apply
```

B：在当前 DDK 服务器核验已确认的源码文档修正版本后，重试追加原 55 档 target 的真实八层特征。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/append_original_ge_features.py" \
  --air /home/l00886453/qwen3_5_moe/glm_export_air_cpro_40block_55_quant/export.air \
  --torchair-pb2 /root/anaconda3/envs/custom_pypto/lib/python3.11/site-packages/torch_npu/dynamo/torchair/_ge_concrete_graph/ge_ir_by_protoc_3_19_pb2.py \
  --source-root /home/w00949577/qwen3.5-35B-moe \
  --gamma-template './glm_export_air_cpro_40block_55_quant/model_model_layers_{layer}_input_layernorm_weight' \
  --sequence-length 55 --ge-root "$CANN_ROOT" \
  --output "$AI_RUN_DIR/target-ge/features" --retry
```
