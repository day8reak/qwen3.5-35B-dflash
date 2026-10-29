# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A（DDK 服务器，decode1 尚未启动时执行一次）：复用成功 prefill64 的只读权重快照单独编译 decode1。

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

A（已配置 HDC 的窗口，可与编译并行）：绑定上次结果，在独占可写目录重新采集工具帮助。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/inspect_target_board_tool.py" \
  --writable-help-from "$AI_RUN_DIR/target-board-tool-inspection" \
  --output "$AI_RUN_DIR/target-board-tool-inspection-writable"
```

B：55 档特征 AIR 和文件清单已完成，本轮保留备份，无需重复生成或编译。
