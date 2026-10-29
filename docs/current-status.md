# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A（DDK 服务器，仅在尚未启动 prefill64 时执行）：读取实际 `dump_graph` 后重编静态 64 档，核验并共享旧只读输入快照。

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

A（原权重机器）：将下面三个小文件手动上传到 DDK 的 `$AI_RUN_DIR/target-pytorch-decode1-transfer/`，平铺保存为 `export.air`、`inventory.json`、`result.json`。

```bash
ls -lh \
  "$AI_RUN_DIR/target-pytorch-decode1/canonical-export/export.air" \
  "$AI_RUN_DIR/target-pytorch-decode1/canonical-export/inventory/inventory.json" \
  "$AI_RUN_DIR/target-pytorch-decode1/canonical-export/result.json"
```

A（DDK 服务器，三个小文件上传完成后）：只核对 decode1 的文件哈希、特征声明及实际输入名，不读取大权重或创建硬链接。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/stage_target_export_reuse.py" \
  --inspect-transfer --sequence-length 1 \
  --export-receipt "$AI_RUN_DIR/target-pytorch-decode1-transfer/result.json" \
  --inventory "$AI_RUN_DIR/target-pytorch-decode1-transfer/inventory.json" \
  --air "$AI_RUN_DIR/target-pytorch-decode1-transfer/export.air"
```

B：55 档特征 AIR 和文件清单已完成，本轮保留备份，无需重复生成或编译。
