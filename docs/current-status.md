# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A（DDK 服务器）：prefill64 已编译成功，复用其只读权重快照单独编译 decode1。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/compile_full_target.py" \
  --borrow-decode-inputs-from "$AI_RUN_DIR/target-pytorch-prefill64-tmpdirfix" \
  --air "$AI_RUN_DIR/target-pytorch-decode1-transfer/export.air" \
  --decode-inventory "$AI_RUN_DIR/target-pytorch-decode1-transfer/inventory.json" \
  --decode-export-receipt "$AI_RUN_DIR/target-pytorch-decode1-transfer/result.json" \
  --profile cpro-40block-decode-1 \
  --input-layout air-directory --target-features \
  --tbe-diagnosis "$AI_RUN_DIR/fused-all-f6b81ead7ae8/diagnosis" \
  --output "$AI_RUN_DIR/target-pytorch-decode1-compile"
```

A（已配置 HDC 的窗口，可与编译并行）：只读采集现有上板工具的哈希、固件和帮助信息。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/inspect_target_board_tool.py" \
  --output "$AI_RUN_DIR/target-board-tool-inspection"
```

B：55 档特征 AIR 和文件清单已完成，本轮保留备份，无需重复生成或编译。
