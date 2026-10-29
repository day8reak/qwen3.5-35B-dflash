# 当前更新与待执行命令

<a id="用户现在执行什么"></a>

A：在 DDK 窗口编译已导入的完整 40 层 target（64/1 档、82 路输出），本轮只启动这一份大编译。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/compile_full_target.py" \
  --air "$AI_RUN_DIR/target-pytorch-import/export.air" \
  --input-layout air-directory --target-features \
  --tbe-diagnosis "$AI_RUN_DIR/fused-all-f6b81ead7ae8/diagnosis" \
  --output "$AI_RUN_DIR/target-pytorch-compile"
```

B：在 DDK 窗口为已生成的 55 档特征 AIR 检查文件引用并生成清单，不复制权重或启动第二次大编译。

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/inventory_target_export.py" \
  --air "$AI_RUN_DIR/target-ge/features/air/export.air" \
  --torchair-pb2 /root/anaconda3/envs/custom_pypto/lib/python3.11/site-packages/torch_npu/dynamo/torchair/_ge_concrete_graph/ge_ir_by_protoc_3_19_pb2.py \
  --reference-root /home/l00886453/qwen3_5_moe \
  --sequence-length 55 \
  --output "$AI_RUN_DIR/target-ge/export-inventory"
```
