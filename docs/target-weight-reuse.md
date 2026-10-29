# A 导出后的权重核对与空间复用

当前64/1双档OMG编译已失败：`branches1`中的`Reshape`输入为`[1,1,2048]`，目标为`[64,2048]`，元素数不一致。原TorchAir调用使用`dynamic=False`，量化投影原代码写的是`reshape(-1, x.shape[-1])`；需要失败IR确定64在哪一阶段固定，不能只按节点名称推断来源。原GDN还有`seq_len == 1`对应的chunk分支，不能只修一个reshape就宣布decode成立。

现在分开推进64-token prefill和1-token decode：64保留已有AIR，用显式静态profile重试；1已在原权重机的`target-pytorch-decode1/canonical-export`完成输出归一和全部参数SHA清单，下一步只传三个小文件，不重复导出或搬运大权重。旧失败任务仍不能直接`--resume`，也不能仅把`--sequence-length`改为64来改变OMG档位；下方第4步的旧双档命令不再运行。

用户回传decode1为`EXPORT_READY`：原AIR SHA `18403bb8c5485b7d1c912ccb8cac8e37c9a8882656e6828d143a3e9b9076e525`，归一AIR为5,970,065字节、SHA `75dc9bd236518db9421532b2791d37e041a9c9e810e588fff19b8d26e155d527`；inventory SHA `eb01e38ffe0ea5d40df8d3aac6fc36dea19abe7e5b3046e2aa54a75d3b686556`。8847个原计算节点逐字节保留，89输入、80 RefData、82输出，特征在81；1153份参数共19,459,474,432字节已在原机实际哈希。这只是输出顺序及参数身份验证，尚无decode1编译、设备或状态连续性通过。

decode1只手动传`canonical-export/export.air`、`canonical-export/inventory/inventory.json`、`canonical-export/result.json`到DDK新目录`target-pytorch-decode1-transfer`，平铺为同名三文件。`stage_target_export_reuse.py --inspect-transfer --sequence-length 1`只实读三小文件，交叉核AIR/清单/归一收据SHA、schema、序列长度与末路FP16`[1,1,16384]`声明，并输出九路Data的原记录名称、shape、dtype、layout和serialized_index；随后重核三文件身份。参数的SHA此时仍是原机声明，没有在DDK重新哈希。AIR内容未在DDK用原机schema重新解码，输入声明来自与该AIR哈希绑定的收据；同shape控制角色及运行时输入顺序仍未证明。

新检查的`TRANSFER_INSPECTED`仅表示三文件检查完成；候选decode1 profile单独报告`MATCHED_EXACT_DATA_NAMES_AND_DECLARED_GEOMETRIES`或`PROFILE_BINDING_REQUIRED`，名称有变化时保留实际九路记录，不按几何自动改名。此模式不访问sidecar路径、不创建目录或链接、不调用OMG，可与prefill64并行。普通stage仍只接受64；decode1的`--apply`未开放，避免链接改变正在使用的prefill源inode的nlink/ctime，或使已完成收据的严格身份复核失效。后续共享参数必须明确处理现有任务的所有权与身份记录。

实际诊断已确认1154份输入全部使用COPY、当前stat均匹配，AIR新算SHA匹配原记录，文件系统空闲44,185,739,264字节；编译输入共19,465,849,776字节。`compile-work/generated/dump_graph`实际是6,443,941字节的普通文件，检查器已按该精确路径和原报告的大小/SHA/stat绑定识别，不再只找同名目录或`.irpb`后缀。记录中的SHA配合stat不是新一轮权重内容验证。

新增`--retry-static-prefill-from`只接受这类已绑定的双档OMG失败任务和原始独立COPY输入，配合`cpro-40block-prefill-64`将五处动态轴固定64并完全省略`dynamic_dims`。它逐项实际哈希旧只读payload后创建硬链接，小AIR独立复制，原staged源文件不变；新输出、日志、TBE隔离和审计都写入新任务。旧报告字节不改，旧快照inode的nlink/ctime会因本次链接而变化，并记录到新JSONL和审计收据；共享inode不能保证抵御特权写入，编译后会重新核SHA和完整身份，变化即失败、保留现场，不自动修复。该入口不是通用失败恢复，不能再次用旧快照普通stat记录盲重试。

静态profile已有`cpro-40block-prefill-64`和`cpro-40block-decode-1`，保留全部80状态及固定尺寸，不加精度覆盖。board计划会拒绝静态profile与所选gear不一致；profile存在仍不代表实际AIR/设备通过。1档后续staging和物理参数共享还需使用其实际新清单，当前不直接套64档导入命令。

用户另贴的原OMG命令只是记录，是否生成过OMC不确定；其中`--d11ynamic_dims`拼写也未得到实际成功日志确认，因此不作为双档成功基线，不据此切换现有已绑定编译器。

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
