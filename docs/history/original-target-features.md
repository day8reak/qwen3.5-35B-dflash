# 原 target 的真实 DFlash 特征接口

> 历史记录：保留当时的观察、失败和候选命令。当前操作请从[文档首页](../README.md)或[当前状态](../current-status.md)进入；不要直接重复本页旧请求。


用户最新安排为[原PyTorch与直接GE两条路线并行](target-parallel-tracks.md)。下文原机导出属于A路线；B使用实际原AIR与源码绑定原算子、权重及状态接口，独立记录结果。当前命令以[当前更新](../current-status.md#用户现在执行什么)为准。

最新回传：A已经成功导出64档AIR和约19.48GB sidecar；当前DDK机已从10GB释放到60GB。按[导出清单与SHA复用](../reference/target-weight-reuse.md)安排A编译，B按[GE特征追加](target-ge-feature-append.md)并行小图构建。下方较早的55档准备示例仅供重现，不重跑已成功任务。

当前目标是先让完整40层量化target与完整6层draft在设备上多轮执行。用户接受这一阶段文本乱码，但明确要求真实target特征。最终精度、状态正确性和至少2x实测目标保留，当前不宣称已完成。

## 模型来源与 verify 范围

target采用用户原PyTorch量化工程，经原TorchAir入口导出AIR，再用OMG编译OMC。原工程中的量化权重加载、专家、完整248320词表及原计算语句保留；本项目没有为该主线自建替代GE target，也不生成69GB FP16整模。

已有原AIR输出是完整词表单位置logits及80个状态，可用于**gear1串行verify**。每轮完整draft输出15候选，调度器先与target已知的下一token比较；匹配才提交该候选，否则提交target纠正token，再让target推进一次。被拒候选不进入target状态，因此这条串行执行路线不依赖并行回滚。它不构成多token并行verify或加速证据。

原AIR没有draft所需的8层hidden。不能从logits或80个cache状态反推这些特征；本轮需原源码追加输出，生成一份新的target AIR/OMC。

## 当前低磁盘安排

执行服务器现有约60GB空闲，用户删除了下载的官网检查点，保留draft资源；原导出机仍有原HF/quant并已完成A导出。OMG继续用当前服务器已配置的DDK；先按新清单实际SHA复用已有原AIR sidecar，只将需要补传的文件和小AIR部署过来。编译快照、临时文件、输出仍独立计算预算，不把`/data`和根目录的同一60GB重复计算。

| 资源 | 安排 |
| --- | --- |
| `model_file` HF目录 | 留原导出机，供原loader使用 |
| `quanted_pth`量化目录 | 留原导出机，保持原量化版本 |
| 原INT8 embedding表 | 连续生成需查行；可单独传约485MiB |
| 原FP32逐词scale表 | 可单独传约0.947MiB |
| 新feature AIR | 原机导出，附引用和校验清单；复用现有源sidecars须逐项校验 |
| 新OMC及编译权重 | 重新编译并绑定；当前服务器的OMG写入共享挂载/大盘，部署时避免本机再缓存整套副本 |

两张embedding表的量级来自248320词×2048 INT8及每词一个FP32 scale，实际文件仍需核对。当前draft prepared只保留FP16 embedding，不能无损恢复原INT8+scale。完整HF/quant目录不因为上传6份源码而成为当前服务器导出的必需搬运项；改在原机使用它们。

当前服务器原AIR和外置权重在：
`/home/l00886453/qwen3_5_moe/glm_export_air_cpro_40block_55_quant/`。
保持该目录内容和旧证据不变。先使用[当前只读命令](../current-status.md#用户现在执行什么)核对AIR结构和控制输入，再确定源sidecar复用或新产物部署。原机已确认只有TorchAir，编译器继续使用当前服务器的已验证环境。把共享存储实际挂载在`$AI_RUN_DIR`的子目录后，可将新的`--output`指向该子目录中的任务路径；这只是存储分工，挂载权限、可用容量和吞吐仍需实际核对。没有可用大盘/共享存储时，先解决容量，再编译；10GB不作为整模编译预算。

直接GE追加输出的入口现为`append_original_ge_features.py`。主机已验证GE生成9节点特征分支、严格schema字段移植和整图Load/Save保真机制；原计算节点编码保持，原NetOutput仅追加1路。当前用户9520节点原图尚待运行该入口。不能用公开Load接口存在或主机机制测试替代真实tap绑定、完整模型编译或设备结果。

原AIR源sidecar和新OMC的编译权重是两套绑定：追加features可能改变OMG分区和packing，旧OMC的`SubGraph_#.weight`不能直接替换新产物。新AIR在原机导出时也可能重新写整套sidecar；原机应为该输出留容量，或在验证后采用单独的复用流程。目前没有“重导出但自动零复制全部权重”的实现。

## 上传到执行服务器

用户指定源码审查目录：`/home/w00949577/qwen3.5-35B-moe`。源码可保留用于审查；当前不在该服务器调用完整检查点导出。原机导出需要的最小源文件同样是以下6个文件，保持相对目录：

```text
export_glm_model_qwen3_5_moe.py
utils.py
models/export_model_wrapper_qwen3_5_moe.py
models/modeling_qwen3_5_moe_hiai_cpro.py
models/modeling_qwen3_5_moe_hiai_nz.py
config/qwen3_5_moe.yaml
```

NZ文件仍必需，因为`utils.py`顶层导入其中的量化类；Cpro入口不需要ND分支。源码合计160773字节，本次私有源码压缩包32917字节，另带`SHA256SUMS`。包仅作为私有任务产物提供，不上传GitHub；不包含权重、AIR、OMC、日志或环境。

将压缩包上传到上述目录后，先解压、校验原文件，再调整YAML路径：

```bash
mkdir -p /home/w00949577/qwen3.5-35B-moe
tar --keep-old-files -xzf /home/w00949577/qwen3.5-35B-moe/qwen35-original-target-source.tar.gz \
  -C /home/w00949577/qwen3.5-35B-moe
(cd /home/w00949577/qwen3.5-35B-moe && sha256sum -c SHA256SUMS)
```

config引用四个外部原始资源，需映射为该服务器真实路径，权重内容保持原版：

| YAML项 | 原始资源 |
| --- | --- |
| `model_file` | HF target目录，含`config.json`及所需`model*.safetensors` |
| `quanted_pth` | 原量化目录，含`data*.safetensors` |
| `embedding_weight_path` | 完整词表INT8 embedding bin |
| `embedding_scale_path` | 完整词表FP32 scale bin |

**先只传小源码包，不重复传大权重或AIR/OMC。** 原loader实际先读HF权重，再覆盖量化参数；仅有量化文件不能推断原target已完整加载。缺少HF参数时原代码仅打印Missing，可能保留初始化值，因此在确认现有权重完整性前不要启动导出。需要保持原定制`torch_npu`/TorchAir环境，源码包不包含这些依赖。

准备器应在原权重机器、路径核对完成后运行，以免准备后修改config造成恢复身份失效。当前服务器使用页首只读AIR检查，不因源码已上传而启动完整检查点导出。

## 底层派生源码工具与第82路输出

当前A路线使用双路线页的`prepare_target_pytorch.py --sequence-length 64`。以下底层工具默认55用于原AIR形状对照，后续55目录示例保留作接口说明，不能替代当前A命令或证明64/1支持。

`prepare_original_target_features.py`从用户私有工程只复制5个Python文件和原config。它通过AST检查实际Cpro类、完整decoder循环、post-layer赋值、原状态收集及return，在派生副本中追加：

- decoder层ID `(1,6,11,16,22,27,32,37)` 的post-decoder hidden；对应参考 `hidden_states[layer_id+1]`，不包括embedding位置。
- 每层转为FP16并clone，按该顺序拼接为 `[1,T,16384]`。
- 原输出0–80保留，特征追加为输出81（第82路）。不足40层、40组cache输入或80个state输出即拒绝。

原私有快照不修改，源码不进入GitHub；准备阶段不读或复制权重、AIR、OMC。派生源码、config、导出脚本和原文件SHA记录在任务的`receipt.json`。未知源码结构、已修改的派生源码或部分任务不会被覆盖。

在原权重机器已有的原TorchAir环境中使用准备器；设置原机的独立AI_RUN_DIR，源码根目录使用原机实际路径，config继续指向原始资源。当前服务器不承担这一完整检查点加载步骤。

准备产物的`export.sh`会从派生源码目录调用原入口，保持 `mode=export quant_mode=enable num_hidden_layers=40 seq_len=55 topK=16`。这与原AIR的导出形状一致，编译仍选64/1动态档。脚本拒绝覆盖已存在的导出目录；它尚未在本轮实际执行，TorchAir及设备仍待验证。

以下导出命令仅在原权重机器、路径和磁盘预算确认后执行一次：

```bash
bash "$AI_RUN_DIR/original-target-features/export.sh"
```

此步骤会新建整模AIR权重，**不会因为准备器只占约162KB而免去导出空间**。服务器只余约10GB时，先保留现有任务并核对预算，不能同时重复导出或编译。生成后可用当前整合入口选择显式82路接口：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/compile_full_target.py" \
  --air "$AI_RUN_DIR/original-target-features/source/glm_export_air_cpro_40block_55_quant/export.air" \
  --source-root "$AI_RUN_DIR/original-target-features/source" \
  --tbe-diagnosis "$AI_RUN_DIR/fused-all-f6b81ead7ae8/diagnosis" \
  --output "$AI_RUN_DIR/full-target-features" \
  --target-features --run-board --sequence-length 64 \
  --input-dir /home/w00949577/run_35B/chunksize_64_data
```

以上是新target的一次构建命令示例；当前机器保留原DDK，`--output`应改为已核实有容量的共享挂载/大盘内任务目录，输入快照和编译scratch随该目录存放；`--run-board`只在该机已有可用HDC和实际输入时使用，否则先只编译后另行部署。当前10GB服务器不立即启动该任务。已有完整编译任务时用`--resume "$AI_RUN_DIR/full-target-features"`继续，配合`--target-features --inspect-only`或`--run-board`；不重新编译。默认无`--target-features`的旧81路接口保持严格校验。新接口只接受末路FP16 `[1,T,16384]`；仅该轴为`-1`时按请求档位解析，固定64不会被推断成支持1。

## 连续状态与运行时输入

原target共有89个输入，包括预量化INT8 embedding、FP32 embedding scale、mask、position、每层两个state及6个控制输入。原源码确认embedding按token查行、只乘一次scale，无需再次量化。状态必须按80个独立槽逐层回灌，不能复用同类型初始文件来代表连续状态。

快照没有实际输入生成器。原导出mask为全`-inf`、`output_pos=[seq_len]`、`conv_state_pos`和`effective_length`为零，均是导出占位，不能当作连续推理配方。`inspect_original_target_inputs.py`只读实际七组小控制文件，可选流式统计mask，明确标记dtype来自原exporter；不读embedding、state或权重，不生成默认值。

`conv_state_pos`128个INT32的真实编码和`allQLen`在decode时的含义仍需实际gear1样本、原输入生成器或所选DDK实现确认。新82路接口检查通过也只证明可见shape/dtype，不证明hidden语义、80状态接续或设备数值正确。

## draft 复用与当前证据

端到端采用现有`full-draft/residual-fix/bundle`。它是完整6层、block16、248320词表，只有原Top1和hidden两路生产输出。用户报告165/165编译精度检查通过、固定合成特征fixture下15/15 Top1一致，hidden相对L2=0.0126796675；尚无真实特征多轮精度或语言质量结论。

本轮主机检查验证了原Cpro类的81输出保持、8层真实特征追加和`torch.compile(fullgraph=True, backend="eager")`控制流；没有执行TorchAir导出或新target设备推理。主机串行调度器具备模型、prefix、状态和features绑定检查，真实HIAI适配与多轮设备执行尚未闭环。
