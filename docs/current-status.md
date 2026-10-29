# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A：在原权重机器调整新 AIR 的输出顺序并原地生成权重哈希清单，无需重新导出或复制权重。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/canonicalize_target_export.py" \
  --air /home/w00949577/qwen35-target-pytorch-run/target-pytorch/target-pytorch-source/source/glm_export_air_cpro_40block_64_quant/export.air \
  --discover-torchair-schema --sequence-length 64 \
  --hash-weights --resume \
  --output "$AI_RUN_DIR/target-pytorch/canonical-export"
```

B：在当前 DDK 服务器只读比较两个变更源码文件的 AST，定位具体代码差异。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/append_original_ge_features.py" \
  --inspect-source --source-ast \
  --source-root /home/w00949577/qwen3.5-35B-moe
```
