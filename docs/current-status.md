# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A（DDK 服务器，原窗口已打印最终失败报告后）：核验停止、目录探针和输入审计收据，复用现有只读快照重试静态 prefill64。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/compile_full_target.py" \
  --retry-static-prefill-from "$AI_RUN_DIR/target-pytorch-prefill64-compile" \
  --profile cpro-40block-prefill-64 \
  --input-layout air-directory --target-features \
  --tmpdir-probe "$AI_RUN_DIR/target-pytorch-tmpdir-probe" \
  --stop-receipt "$AI_RUN_DIR/target-pytorch-prefill64-stop-legacy" \
  --tbe-diagnosis "$AI_RUN_DIR/fused-all-f6b81ead7ae8/diagnosis" \
  --output "$AI_RUN_DIR/target-pytorch-prefill64-tmpdirfix"
```

B：55 档特征 AIR 和文件清单已完成，本轮保留备份，无需重复生成或编译。
