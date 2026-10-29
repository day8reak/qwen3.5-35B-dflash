# 原 target 的真实 DFlash 特征接口

当前目标是先让完整40层量化target与完整6层draft在设备上多轮执行。用户接受这一阶段文本乱码，但明确要求真实target特征。最终精度、状态正确性和至少2x实测目标保留，当前不宣称已完成。

## 模型来源与 verify 范围

target采用用户原PyTorch量化工程，经原TorchAir入口导出AIR，再用OMG编译OMC。原工程中的量化权重加载、专家、完整248320词表及原计算语句保留；本项目没有为该主线自建替代GE target，也不生成69GB FP16整模。

已有原AIR输出是完整词表单位置logits及80个状态，可用于**gear1串行verify**。每轮完整draft输出15候选，调度器先与target已知的下一token比较；匹配才提交该候选，否则提交target纠正token，再让target推进一次。被拒候选不进入target状态，因此这条串行执行路线不依赖并行回滚。它不构成多token并行verify或加速证据。

原AIR没有draft所需的8层hidden。不能从logits或80个cache状态反推这些特征；本轮需原源码追加输出，生成一份新的target AIR/OMC。

## 派生源码与第82路输出

`prepare_original_target_features.py`从用户私有工程只复制5个Python文件和原config。它通过AST检查实际Cpro类、完整decoder循环、post-layer赋值、原状态收集及return，在派生副本中追加：

- decoder层ID `(1,6,11,16,22,27,32,37)` 的post-decoder hidden；对应参考 `hidden_states[layer_id+1]`，不包括embedding位置。
- 每层转为FP16并clone，按该顺序拼接为 `[1,T,16384]`。
- 原输出0–80保留，特征追加为输出81（第82路）。不足40层、40组cache输入或80个state输出即拒绝。

原私有快照不修改，源码不进入GitHub；准备阶段不读或复制权重、AIR、OMC。派生源码、config、导出脚本和原文件SHA记录在任务的`receipt.json`。未知源码结构、已修改的派生源码或部分任务不会被覆盖。

在当前已配置原TorchAir环境的服务器中，先执行[当前命令](current-status.md#用户现在执行什么)的源码准备和控制输入只读检查。原工程路径不同，仅替换`--source-root`；config内权重路径照原值使用。

准备产物的`export.sh`会从派生源码目录调用原入口，保持 `mode=export quant_mode=enable num_hidden_layers=40 seq_len=55 topK=16`。这与原AIR的导出形状一致，编译仍选64/1动态档。脚本拒绝覆盖已存在的导出目录；它尚未在本轮实际执行，TorchAir及设备仍待验证。

仅在原权重路径和磁盘预算确认后执行一次导出：

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

这是新target的一次构建命令，当前不要求立即启动。已有完整编译任务时用`--resume "$AI_RUN_DIR/full-target-features"`继续，配合`--target-features --inspect-only`或`--run-board`；不重新编译。默认无`--target-features`的旧81路接口保持严格校验。新接口只接受末路FP16 `[1,T,16384]`；仅该轴为`-1`时按请求档位解析，固定64不会被推断成支持1。

## 连续状态与运行时输入

原target共有89个输入，包括预量化INT8 embedding、FP32 embedding scale、mask、position、每层两个state及6个控制输入。原源码确认embedding按token查行、只乘一次scale，无需再次量化。状态必须按80个独立槽逐层回灌，不能复用同类型初始文件来代表连续状态。

快照没有实际输入生成器。原导出mask为全`-inf`、`output_pos=[seq_len]`、`conv_state_pos`和`effective_length`为零，均是导出占位，不能当作连续推理配方。`inspect_original_target_inputs.py`只读实际七组小控制文件，可选流式统计mask，明确标记dtype来自原exporter；不读embedding、state或权重，不生成默认值。

`conv_state_pos`128个INT32的真实编码和`allQLen`在decode时的含义仍需实际gear1样本、原输入生成器或所选DDK实现确认。新82路接口检查通过也只证明可见shape/dtype，不证明hidden语义、80状态接续或设备数值正确。

## draft 复用与当前证据

端到端采用现有`full-draft/residual-fix/bundle`。它是完整6层、block16、248320词表，只有原Top1和hidden两路生产输出。用户报告165/165编译精度检查通过、固定合成特征fixture下15/15 Top1一致，hidden相对L2=0.0126796675；尚无真实特征多轮精度或语言质量结论。

本轮主机检查验证了原Cpro类的81输出保持、8层真实特征追加和`torch.compile(fullgraph=True, backend="eager")`控制流；没有执行TorchAir导出或新target设备推理。主机串行调度器具备模型、prefix、状态和features绑定检查，真实HIAI适配与多轮设备执行尚未闭环。
