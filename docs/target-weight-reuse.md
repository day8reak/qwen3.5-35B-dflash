# A 导出后的权重核对与空间复用

用户已在原权重机导出完整A：AIR为6,375,344字节，SHA256为`3474492f31d3de05ddbc843daa6cbd75e52b641f277f684442dde6d4385f26d9`；导出目录1155个常规文件，总19,478,109,527字节。原机剩余约7.1TB，当前DDK机释放到60GB；`/data`与根目录使用同一已占用卷，不是另一块空闲大盘。

先比较新导出和当前服务器已有55档sidecar的真实字节身份。完全一致才复用；文件名相同或shape相同均不够。

## 1. 原权重机生成清单

保持原`qwen3_5_cpro`环境和原机的`AI_RUN_DIR`。这会流式读取约19.48GB参数算SHA，不复制这些文件、不重新导出：

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/inventory_target_export.py" \
  --air /home/w00949577/qwen35-target-pytorch-run/target-pytorch/target-pytorch-source/source/glm_export_air_cpro_40block_64_quant/export.air \
  --discover-torchair-schema --hash-weights --sequence-length 64 \
  --output "$AI_RUN_DIR/target-pytorch/export-inventory"
```

返回`INVENTORIED`、`weight_identity=SHA256_COMPLETE`后，清单是`$AI_RUN_DIR/target-pytorch/export-inventory/inventory.json`。schema发现只读pb2 AST，所有可用literal descriptor必须同SHA才自动选取，否则要求显式`--torchair-pb2`；不导入NPU包。

清单分别记录89输入/82输出及末路FP16`[1,64,16384]`的声明、实际FileConstant路径/shape属性/dtype属性、每个唯一payload的大小和SHA。producer空shape仍单列未解决，不把边界声明或文件哈希当精度/真实特征语义通过。

## 2. 仅传小文件到DDK机

将以下文件传到当前服务器的`$AI_RUN_DIR/target-pytorch-transfer/`，保持目标文件名：

- 新的`export.air`（约6.4MB）。
- 上一步`inventory.json`。
- 原机`$AI_RUN_DIR/target-pytorch/target-pytorch-source/receipt.json`，保存为`source-receipt.json`，用于两路线的原源码审计。

此时不传19.48GB sidecar，也不传HF/quant检查点。两台机器的`AI_RUN_DIR`分别使用各自实际值。

## 3. DDK机核对并复用原有文件

以下只创建新的导入目录和同文件系统硬链接。每个现有文件必须实际SHA/大小匹配清单；源路径逐级拒绝symlink。不更改原文件字节或权限，不覆盖旧目录，跨文件系统不退化成大文件复制：

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/stage_target_export_reuse.py" \
  --inventory "$AI_RUN_DIR/target-pytorch-transfer/inventory.json" \
  --air "$AI_RUN_DIR/target-pytorch-transfer/export.air" \
  --existing-export /home/l00886453/qwen3_5_moe/glm_export_air_cpro_40block_55_quant \
  --output "$AI_RUN_DIR/target-pytorch-import" --apply
```

省略`--apply`仅生成只读计划。返回`TRANSFER_REQUIRED`时，没有创建导入目录；只按报告的缺失/不同文件安排补传，不能把旧文件直接当成新参数。返回`STAGED_HARDLINKS`才得到`$AI_RUN_DIR/target-pytorch-import/export.air`及原64前缀目录。硬链接共享原inode，因此不对导入文件执行chmod/写入；工具只允许自身link计数/ctime变化并复核最终SHA。工具没有显式复制权重；底层文件系统的物理分配仍以实际空闲变化为准。

## 4. 优先编译A，暂不同时启动B的大编译

导入成功后，仍需给隔离输入快照和OMG输出留空间。外置编译权重的实际大小未知，60GB不能直接视作全流程容量保证。当前先安排A的一次大编译，B继续小图生成/检查。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/compile_full_target.py" \
  --air "$AI_RUN_DIR/target-pytorch-import/export.air" \
  --input-layout air-directory --target-features \
  --tbe-diagnosis "$AI_RUN_DIR/fused-all-f6b81ead7ae8/diagnosis" \
  --output "$AI_RUN_DIR/target-pytorch-compile"
```

这里明确使用`air-directory`：导入后的AIR位于外层，原`glm_export_air_cpro_40block_64_quant/`前缀已在其下保留。不能使用默认`export-directory`再重复加一层前缀。编译入口保留源文件、先检查实际OMG选项，再创建独立输入快照；有reflink则复用物理块，否则快照仍可能占约19.48GB。

本命令只编译和检查，不运行导出dummy输入。原80状态和真实控制值、64→1连续性、真实features与draft的多轮HIAI接续仍要实测。
