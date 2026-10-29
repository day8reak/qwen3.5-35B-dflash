# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A（已配置 HDC 的窗口，prefill64 和 decode1 均已编译完成）：绑定上次结果，在独占可写目录重新采集工具帮助。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/inspect_target_board_tool.py" \
  --writable-help-from "$AI_RUN_DIR/target-board-tool-inspection" \
  --output "$AI_RUN_DIR/target-board-tool-inspection-writable"
```

A（DDK 服务器）：读取现有 64 档输入的小控制文件和 mask 统计，为首次上板核对输入。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/inspect_original_target_inputs.py" \
  --input-dir /home/w00949577/run_35B/chunksize_64_data \
  --sequence-length 64
```

B：55 档特征 AIR 和文件清单已完成，本轮保留备份，无需重复生成或编译。
