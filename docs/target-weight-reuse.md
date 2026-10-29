# A 导出后的权重核对与空间复用

当前64/1双档OMG编译已失败：`branches1`中的`Reshape`输入为`[1,1,2048]`，目标为`[64,2048]`，元素数不一致。原TorchAir调用使用`dynamic=False`，量化投影原代码写的是`reshape(-1, x.shape[-1])`；需要失败IR确定64在哪一阶段固定，不能只按节点名称推断来源。原GDN还有`seq_len == 1`对应的chunk分支，不能只修一个reshape就宣布decode成立。

现在分开推进64-token prefill和1-token decode：64保留已有AIR，先检查失败编译收据和快照；1在原权重机用全新`target-pytorch-decode1`目录导出。实际原机导出会新写约19.46GB sidecar，后续逐文件SHA相同时再复用DDK现有参数，不把重复权重搬到DDK机。失败任务目前不能直接`--resume`重编译，也不能仅把`--sequence-length`改为64来改变OMG档位。下方第1至3步已完成，第4步的旧双档命令暂不再次运行。

新的只读诊断只检查小元数据、编译日志、失败IR、微小整数shape常量；1153权重快照只stat，不重新读payload。`INSPECTED`表示报告读取完成，`RECORDED_HASH_WITH_CURRENT_STAT`不是新一轮完整SHA验证，也不是重试通过；后续要按实际报告决定怎样保留并复用现有编译输入。源码AIR未提供其原机schema时不猜用另一版TorchAir schema解析，失败dump则使用所选DDK自己的HIAI schema。

用户已在原权重机导出完整A：AIR为6,375,344字节，SHA256为`3474492f31d3de05ddbc843daa6cbd75e52b641f277f684442dde6d4385f26d9`；导出目录1155个常规文件，总19,478,109,527字节。原机剩余约7.1TB，当前DDK机释放到60GB；`/data`与根目录使用同一已占用卷，不是另一块空闲大盘。

用户已完成下方第1至3步：归一`EXPORT_READY`、三小文件传输、1153份参数全部`STAGED_HARDLINKS`。归一AIR仍为6,375,344字节，SHA为`b48169f46dcd0964501bc27f0d2de42d8dad7991b8d72532ae40e208d407d734`；清单SHA为`63f19e8f2291a70a38b0775cfc08abeef2517f3dbab10ba328d34178970eb4d4`。1153个唯一参数文件已实际哈希，总19,459,474,432字节；9537个原计算节点编码逐字节保留，原AIR不变。不必重复归一、传输或stage。

实际staging收据SHA为`f5206cbdef5ce496250f52934dca75c7ae5b24c869826547e525ccbfc3cc77f9`，九个Data名称/形状已匹配编译profile，`transfer_required_count=0`；工具显式复制的sidecar字节为0。收据写入前观测到空闲63,754,436,608字节，不能把此共享文件系统观测当作独立物理占用测量。编译输入扫描应选入1154个文件，即小AIR加1153份已绑定sidecar。

先比较新导出和当前服务器已有55档sidecar的真实字节身份。完全一致才复用；文件名相同或shape相同均不够。

## 1. 原权重机调整输出顺序并生成清单

用户的接口检查已确认：特征位于物理端口1，FP16 ND `[1,64,16384]`；其后是原80路状态。保持原`qwen3_5_cpro`环境和原机的`AI_RUN_DIR`，生成新的小AIR，将顺序从`[logits,features,80 states]`改为`[logits,80 states,features]`。独立校验器检查固定端口排列和全部计算节点的原始字节保留，随后流式读取约19.48GB原参数算SHA，不复制这些文件、不重新导出：

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/canonicalize_target_export.py" \
  --air /home/w00949577/qwen35-target-pytorch-run/target-pytorch/target-pytorch-source/source/glm_export_air_cpro_40block_64_quant/export.air \
  --discover-torchair-schema --sequence-length 64 \
  --hash-weights --resume \
  --output "$AI_RUN_DIR/target-pytorch/canonical-export"
```

返回`EXPORT_READY`、`weight_identity=SHA256_COMPLETE`后，新AIR是`$AI_RUN_DIR/target-pytorch/canonical-export/export.air`，清单是`$AI_RUN_DIR/target-pytorch/canonical-export/inventory/inventory.json`。原AIR、全部计算节点及外置参数引用保持原字节；只调整NetOutput端口及其对应元数据。完整清单可在源身份和文件stat均未变时复用，避免重复哈希大参数。已有失败检查目录保留不动。

schema发现只读原机pb2 AST，所有可用literal descriptor必须同SHA才自动选取，否则要求显式`--torchair-pb2`；不导入NPU包。原机schema描述符SHA为`a7cd334beff74bfeb637b1886329a70122842667a257d84ba111c70991695679`，与当前DDK机的schema不同，不能强行替换成DDK机版本。

清单分别记录89输入/82输出及末路FP16`[1,64,16384]`的声明、实际FileConstant路径/shape属性/dtype属性、每个唯一payload的大小和SHA。producer空shape仍单列未解决，不把边界声明或文件哈希当精度/真实特征语义通过。

## 2. 仅传小文件到DDK机

将原机`$AI_RUN_DIR/target-pytorch/canonical-export/`内的以下三项手动上传到当前服务器的`$AI_RUN_DIR/target-pytorch-transfer/`，平铺保存为以下目标文件名：

- `export.air`（约6.4MB）→`export.air`。
- `inventory/inventory.json`→`inventory.json`。
- `result.json`→`result.json`，记录原AIR、归一证明、schema及清单身份；这是本次归一任务收据，并非原PyTorch源码六文件receipt。

原六文件`target-pytorch-source/receipt.json`可另外保留；当前B使用实际`--source-root`重读源码，不依赖这份receipt。此时不传19.46GB sidecar，也不传HF/quant检查点。两台机器的`AI_RUN_DIR`分别使用各自实际值。

## 3. DDK机核对并复用原有文件

以下只创建新的导入目录和同文件系统硬链接。每个现有文件必须实际SHA/大小匹配清单；源路径逐级拒绝symlink。不更改原文件字节或权限，不覆盖旧目录，跨文件系统不退化成大文件复制：

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/stage_target_export_reuse.py" \
  --export-receipt "$AI_RUN_DIR/target-pytorch-transfer/result.json" \
  --inventory "$AI_RUN_DIR/target-pytorch-transfer/inventory.json" \
  --air "$AI_RUN_DIR/target-pytorch-transfer/export.air" \
  --existing-export /home/l00886453/qwen3_5_moe/glm_export_air_cpro_40block_55_quant \
  --output "$AI_RUN_DIR/target-pytorch-import" --apply
```

省略`--apply`仅生成只读计划。返回`TRANSFER_REQUIRED`时，没有创建导入目录；只按报告的缺失/不同文件安排补传，不能把旧文件直接当成新参数。返回`STAGED_HARDLINKS`才得到`$AI_RUN_DIR/target-pytorch-import/export.air`及原64前缀目录。硬链接共享原inode，因此不对导入文件执行chmod/写入；工具只允许自身link计数/ctime变化并复核最终SHA。工具没有显式复制权重；底层文件系统的物理分配仍以实际空闲变化为准。

`--export-receipt`先交叉绑定这三个小文件，并核对AIR实读记录的九个Data名称/形状与64/1编译profile一致；发现参数编号或固定形状不同即在读大权重前停止。该检查不据同形状猜控制值含义，也不证明实际OMG支持或状态连续性。

## 4. 首次双档编译命令记录（已失败，暂不重跑）

导入成功后，仍需给隔离输入快照和OMG输出留空间。外置编译权重的实际大小未知，60GB不能直接视作全流程容量保证。当前先安排A的一次大编译，B继续小图生成/检查。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/compile_full_target.py" \
  --air "$AI_RUN_DIR/target-pytorch-import/export.air" \
  --input-layout air-directory --target-features \
  --tbe-diagnosis "$AI_RUN_DIR/fused-all-f6b81ead7ae8/diagnosis" \
  --output "$AI_RUN_DIR/target-pytorch-compile"
```

这里明确使用`air-directory`：导入后的AIR位于外层，原`glm_export_air_cpro_40block_64_quant/`前缀已在其下保留。不能使用默认`export-directory`再重复加一层前缀。编译入口保留源文件、先检查实际OMG选项，再创建独立输入快照；有reflink则复用物理块，否则快照仍可能占约19.48GB。

编译输入扫描只接受staging收据绑定的前缀和参数文件，不自动扫描其它任意子目录。原机TorchAir schema无需作为OMG输入；编译后使用当前DDK的HIAI schema检查新OMC，不能拿原机schema替换DDK schema。

本命令只编译和检查，不运行导出dummy输入。原80状态和真实控制值、64→1连续性、真实features与draft的多轮HIAI接续仍要实测。
