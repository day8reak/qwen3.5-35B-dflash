# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A（DDK 服务器，另开窗口）：仅对已暂停且身份匹配的本次编译树排队终止并恢复执行，让外层驱动继续完成审计。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/stop_target_compile.py" \
  --job "$AI_RUN_DIR/target-pytorch-prefill64-compile" \
  --apply --legacy-stopped-tree \
  --output "$AI_RUN_DIR/target-pytorch-prefill64-stop-legacy"
```

A（DDK 服务器）：用同版源码校验目录修复，并验证 forkserver 短路径通信。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/probe_tbe_tmpdir.py" \
  --job "$AI_RUN_DIR/target-pytorch-prefill64-compile" \
  --expected-ccec-sha256 5bef4eba82325b701a544e1644a38e566788de7f1cff5888e36d3ac609fddb6a \
  --output "$AI_RUN_DIR/target-pytorch-tmpdir-probe"
```

A：decode1 三小文件已核对成功；等待停止后的最终审计和目录探针结果，再安排复用权重编译。

B：55 档特征 AIR 和文件清单已完成，本轮保留备份，无需重复生成或编译。
