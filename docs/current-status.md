# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A：在原权重机器只读检查新 AIR 的实际输出接口，不读取外置权重。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/inventory_target_export.py" \
  --inspect-only \
  --air /home/w00949577/qwen35-target-pytorch-run/target-pytorch/target-pytorch-source/source/glm_export_air_cpro_40block_64_quant/export.air \
  --discover-torchair-schema --sequence-length 64
```

B：在当前 DDK 服务器只读核对原源码，列出具体不匹配的文件。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/append_original_ge_features.py" \
  --inspect-source \
  --source-root /home/w00949577/qwen3.5-35B-moe
```
