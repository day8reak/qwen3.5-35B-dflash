# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A（DDK 服务器）：读取实际 `dump_graph` 文件后，仅重编静态 64 档，核验并共享旧只读输入快照，不再复制大权重。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/inspect_target_compile_failure.py" \
  --job "$AI_RUN_DIR/target-pytorch-compile" \
  --node Reshape \
  --output "$AI_RUN_DIR/target-pytorch-compile-diagnosis-v2.json" &&
python3 "$REPO_DIR/targets/cpro/scripts/compile_full_target.py" \
  --retry-static-prefill-from "$AI_RUN_DIR/target-pytorch-compile" \
  --profile cpro-40block-prefill-64 \
  --input-layout air-directory --target-features \
  --tbe-diagnosis "$AI_RUN_DIR/fused-all-f6b81ead7ae8/diagnosis" \
  --output "$AI_RUN_DIR/target-pytorch-prefill64-compile"
```

A（原权重机器）：1-token AIR 已导出，现在只归一其输出顺序并原地生成权重哈希清单，不重复导出或搬运大权重。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/canonicalize_target_export.py" \
  --air "$AI_RUN_DIR/target-pytorch-decode1/target-pytorch-source/source/glm_export_air_cpro_40block_1_quant/export.air" \
  --discover-torchair-schema --sequence-length 1 \
  --hash-weights --resume \
  --output "$AI_RUN_DIR/target-pytorch-decode1/canonical-export"
```

B：55 档特征 AIR 和文件清单已完成，本轮保留备份，无需重复生成或编译。
