# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A（DDK 服务器）：只读检查现有失败编译的快照、Reshape 小常量和 IR，不复制权重或重新编译。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/inspect_target_compile_failure.py" \
  --job "$AI_RUN_DIR/target-pytorch-compile" \
  --node Reshape \
  --output "$AI_RUN_DIR/target-pytorch-compile-diagnosis.json"
```

A（原权重机器）：沿用原 64 档导出环境和 `ORIGINAL_TARGET_SOURCE`，单独导出完整 1-token decode AIR，新增 sidecar 写在原机。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/prepare_target_pytorch.py" \
  --source-root "$ORIGINAL_TARGET_SOURCE" \
  --config "$ORIGINAL_TARGET_SOURCE/config/qwen3_5_moe.yaml" \
  --output "$AI_RUN_DIR/target-pytorch-decode1" \
  --sequence-length 1 &&
bash "$AI_RUN_DIR/target-pytorch-decode1/export.sh"
```

B：55 档特征 AIR 和文件清单已完成，本轮保留备份，无需重复生成或编译。
