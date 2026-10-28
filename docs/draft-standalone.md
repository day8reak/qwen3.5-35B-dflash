# 完整 draft 独立执行

当前优先跑通完整六层 DFlash，不等待40层 target 编译。沿用已经配置的 `env.sh`、主机GE、目标DDK和HDC环境。这里的“权重目录”指含 `config.json`、safetensors文件及可能的索引文件的下载目录，不是AIR或OMC输出目录。

```bash
git pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/run_full_draft.py" \
  --draft-weights "$QWEN35_DRAFT_WEIGHTS" \
  --target-weights "$QWEN35_TARGET_WEIGHTS" \
  --tbe-diagnosis "$AI_RUN_DIR/fused-all-f6b81ead7ae8/diagnosis"
```

两个权重参数也可直接填写服务器上的实际目录；省略时读取 `QWEN35_RESOURCES` 中对应配置并展开环境变量。target目录使用原始模型权重，只读取其embedding和LM head，不读取40层decoder权重，不重新量化原target。

独立draft不要求target目录有 `model.safetensors.index.json`。索引缺失时，入口有界扫描同级 `.safetensors` 文件头，包括 `model.safetensors-00009-of-00014.safetensors` 这类名称；先检查全部所需张量的键名、形状、BF16类型及数据范围，再开始转换。重复键、缺失张量或截断会直接报错，不替换权重，也不生成假索引。配置仍须与项目锁定版本匹配；生成配置和索引是否存在分别记录，配置哈希匹配不等于上游权重版本已经验证。

项目锁定的[官方target索引](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/resolve/59d61f3ce65a6d9863b86d2e96597125219dc754/model.safetensors.index.json)将 `model.language_model.embed_tokens.weight` 和 `lm_head.weight` 均映射到第9片。仅这两个张量完整即可供独立draft使用；不要求其余target decoder分片齐全，也不据此宣称完整target下载已完成。

默认上下文容量64、有效前缀6，完整6层、block16、248320词表保持不变。输出 `int32[1,15]` 候选ID和 `float16[1,15,2048]` 的head前hidden。FP16权重包含共享embedding/head约2.61GiB，这是存储量估计，不是板端峰值内存。

入口自动建立新的 `full-draft-*` 目录，依次执行：

1. 核对锁定配置，流式准备71个FP16权重张量；同一读取过程中记录原始张量和转换后数据哈希。
2. 使用全部draft层和完整LM head生成PyTorch参考。输入是固定的合成target features、token IDs和有效前缀mask。
3. 只导出draft AIR，采用已有基础GE运算与FP16计算约定。
4. 只编译draft OMC；默认保留外置数据，也可显式选择 `--weight-storage embedded` 将权重编入OMC。恢复成功tiny路径的 `--use_origin_format=true --weight_data_type=FP16`。每30秒显示编译等待及日志信息。
5. 构建当前HIAI runner，上板执行一次draft，回收输出并比较。原生报告应记录 `target_calls=0`、`draft_calls=1`。

`result.json`记录总状态和各阶段路径。`board/comparison.json`记录Top1差异数、hidden最大绝对误差和relative L2；Top1要求零差异，hidden目前仅作误差诊断，尚无完整数值验收阈值。所有输入、模型、外置数据、参考输出和回收文件均保留哈希。输入回读一致不能证明设备缓存可见性，运行成功也不能证明NPU执行位置。

这一步验证完整draft在固定输入上的执行与输出，不是独立文本生成，也不证明原始BF16质量、与量化target的接受率或加速。当前实现是prefix recomputation，没有新增draft KV增量缓存。后续将固定features换成真实target的八层特征，再做接入验证。

如果编译或HDC失败，直接保留目录并恢复：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_full_draft.py" \
  --resume "$AI_RUN_DIR/full-draft-实际批次号" \
  --tbe-diagnosis "$AI_RUN_DIR/fused-all-f6b81ead7ae8/diagnosis"
```

恢复创建新目录并验证复用阶段的清单哈希；进程中断时可从 `job.json` 和 `completed-*.json` 恢复。与当前导出契约一致的成功导出和编译不重跑。旧版 AIR 的外置权重序列化契约失效时，自动保留 prepared 权重和 PyTorch reference，重新导出 AIR、编译和上板；新结果记录 `resume_invalidations`，旧目录不改动。更换权重或输入形状需新建任务。只准备权重和参考可用 `--prepare-only`；准备至OMC但暂不上板可用 `--compile-only`。不要为本入口重跑旧算子探针或平台诊断矩阵。

最新 `full-draft-732b26335347` 已完成v4 AIR编译和bundle，用户HiLog显示PID46426的CPUCL初始化成功，已越过此前 `v501/BroadcastTo` 失败。随后NPU `SubGraph_1` 的 `InitRuntimeParams` 报权重内存大小不匹配：`weightsMemBaseSize=16`，`modelWeightMemorySize=1151486976`。模型尚未进入Process，不能把CPUCL初始化成功当作掩码或模型数值通过。

v4将BOOL掩码转成INT32的0/1，GatherV2使用全零索引复制单例query/head维，最后转回BOOL；覆盖六层显式扩展和Select补出的六层head扩展，共12处。掩码值、权重、PyTorch参考、浮点计算和输出ABI不变。此前v3的INT32形状参数修改没有解决设备加载问题，历史失败记录保留；v4目前只取得编译及CPUCL初始化进展，完整执行仍失败。

保留最新732批次的AIR、compilation、bundle和board证据，以及其引用的 `d7b4db958850/prepared` 与fixture。运行时15条子图记录都显示 `weight hash:0x1dabbe2881ec5472`，尺寸却不同；这是运行时日志字段，不是15份文件的SHA256相同证据。用户现已选择尝试权重内嵌OMC：

```bash
git pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/run_full_draft.py" \
  --resume "$AI_RUN_DIR/full-draft-732b26335347" \
  --weight-storage embedded \
  --tbe-diagnosis "$AI_RUN_DIR/fused-all-f6b81ead7ae8/diagnosis"
```

`embedded` 显式设置OMG的 `--save_weights_as_external_data=false`，`external` 设置为true。新任务默认external；续跑不指定时继承已记录模式。模式变化只使compilation/bundle失效，复用732的v4 AIR以及d7 prepared/fixture，在新目录编译、打包和执行，旧证据不改。这不重算参考、不重新量化，也不修改图或权重；只是另一种装载方式，所选DDK编译及设备Load能否成功仍待验证。

用户追加的编译文件大小显示 `SubGraph_1.weight` 为1,151,486,976字节，恰好等于模型要求；`model.omc` 为2,905,990字节。这排除了所列编译权重文件只有16字节的解释，尚未独立验证文件SHA256、设备副本或HIAI实际选择的权重。

15份原sidecar共2,849,468,580字节，低于 `SetModelBuffer` 的UINT32长度上限4,294,967,295字节；该合计不能保证内嵌后的OMC大小或编译布局。打包和原生加载按新OMC的实际长度检查，超限明确报错；最终是否内嵌及是否仍产生外置数据，以真实编译产物为准。

发生板端失败时，入口会显示 `board_stage`、通过身份校验的 `failure_stage`/原生原因，以及独立的传输或比较错误，并输出有界stderr/stdout日志。旧版没有打印原因时，可直接读取已有文件，无需再次运行模型：

```bash
cat "$AI_RUN_DIR/full-draft-732b26335347/board/results/result.json"
cat "$AI_RUN_DIR/full-draft-732b26335347/board/comparison.json"
tail -n 100 "$AI_RUN_DIR/full-draft-732b26335347/board/draft.stderr"
```

`LOAD_OR_SETUP` 包含模型加载和IO建立，`INPUT_BINDING` 是输入绑定，`PROCESS` 是推理调用；需结合 `reason` 与原生日志定位，不能仅用阶段名断言根因。失败记录不要求生成输出bin，缺失成功输出不再掩盖已有原生错误。产物/运行时身份不匹配和证据回收失败仍使任务失败。

后续runner报告实际 `process_id`。失败时入口自动读取一次 `hilog -x` 并保留 `board/draft-hilog.txt`、stderr、退出码；校验后按PID生成候选摘要。每次HDC诊断调用最多10秒，不清空设备日志或修改设置，也不自动重跑模型。系统日志及PID可能对应其他会话，需结合时间和路径确认；诊断失败不会覆盖模型原生错误。

用户回传 `full-draft-d7b4db958850` 已完成71个权重、PyTorch参考和 AIR 导出；OMG 在首个 `FileConstant` 报告未识别 `dtype`，并将绝对权重路径拼在编译目录之后。新 draft 导出将权重引用写为 `weights/wNNNNN.bin`，从 AIR 父目录执行序列化，并显式使用整数 GE dtype ID 的 CPro 兼容策略。它不改变张量内容；通用 GE 默认 Type 属性保持原样。该兼容策略已纳入清单版本，后续 `full-draft-5133743f7c9c` 的日志已越过外置权重读取。

`full-draft-5133743f7c9c` 在首个 MatMul 兼容检查处失败。补齐tiny两个编译参数后，`full-draft-23abc47e25e7` 仍失败；用户追加上下文明确显示该实现拒绝 K>8000，而首个 `draft.fc` 的 K=16384。新导出仅将这一个投影表达成一次无bias、无padding的4×4 Conv2D。输入和权重只是reshape，显式NCHW描述；全点积和一次FP16结果边界保留。其他投影、六层结构和完整词表均不变，PyTorch参考仍使用原MatMul。

`full-draft-a091f16d13d8` 的用户日志已显示OMG编译和保存成功。外置权重模式实际生成 `compile-work/generated/model/model.omc`，而旧脚本只检查 `generated/model.omc`，因此误报失败。现在只接受这两种明确布局中的唯一非空普通模型文件，拒绝歧义、符号链接和空文件；保存并上传整个相对目录，保持 `SubGraph_*.weight` 与OMC的位置关系。

旧版输出检查错误的恢复会核验成功日志、原导出及编译输入、DDK/参数、全部编译产物和TBE审计，在新目录保存独立快照与成功记录；旧失败证据不变。旧日志没有成功退出码，恢复记录明确标注由检查阶段及成功日志推断。该恢复不重跑OMG，但不能复用已经过期的导出契约。

当前导出契约仍为 `cpro-fileconstant-relative-int-fc-conv2d-mask-gather-v4`，最新732批次与它一致，AIR继续复用。本次显式切换embedded会改变编译参数，仅重新编译和打包；同一模式后续续跑可复用已完成产物。更早的v2/v3批次恢复仍会触发重导和编译，因此不作为当前恢复起点。

`air/draft/projection-lowering.json`继续记录fc卷积表达的形状和等价范围。Conv2D与MatMul的累加顺序可能不同，板端仍须对照原参考，未宣称逐位一致或20ms性能。recipe为显式NCHW使用Q35GE002，普通ND图保持Q35GE001并兼容读取。

外置权重路径审查只覆盖本仓库源码：编译清单逐文件记录大小和SHA256，bundle保留 `generated` 下相对路径，上传前再核验本地文件，设备执行前调用 `sha256sum -c`。文件按路径匹配，没有按自然数或字典序重新分配子图权重；例如 `generated/model/SubGraph_1.weight` 对应 `bundle/model/model/SubGraph_1.weight`，与主OMC同级。当前未独立回收732批次的实际文件清单及校验调用记录，也不知道HIAI最终打开的文件；不能以源码审查代替远端身份或加载路径验证。

加载接口仍是待核对线索：runner先调用 `SetModelPath`，随后调用 `SetModelBuffer`。较旧的[Huawei署名公开源码镜像](https://github.com/946166920/hiai_ddk/blob/42142161fb414a8d53973b0d5ccbd0de7872d5d9/src/framework/model_manager/compatible/AiModelDescription.cpp#L113-L131)中，`SetModelBuffer` 会清空模型路径；这只是定位线索，尚未确认当前所选DDK有相同行为。用户回传的所选头文件没有显式外置权重setter，但声明不能确定getter语义或路径是否保留。此前成功tiny未传 `--save_weights_as_external_data=true`，完整draft则明确启用外置权重，因此tiny成功没有覆盖这项差异。本次embedded候选沿用现有加载API，不据旧源码猜测并修改调用行为；没有新增导出契约或已验证的加载修复。

需要只读检查已有日志、recipe和编译图dump时，可选：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_full_draft.py" \
  --inspect-only --resume "$AI_RUN_DIR/full-draft-23abc47e25e7"
```

此入口不准备权重、不导出、不编译、不上板。它核对 AIR/recipe 清单和编译输入记录，显示失败节点及权重生产者的声明描述、精确节点日志上下文；尝试从该编译记录的 DDK protobuf 库静态读取 schema，解码有界的已保存 `.irpb`。库/文件缺失、过大、schema 不匹配或节点不明确均单独报告，不猜测 dtype 编号含义，也不把 recipe 描述当作编译后描述。可用 `--schema-library` 显式指定该 DDK 的 protobuf 库；不会加载执行库或调用 OMG。诊断结果写入新的 `full-draft-inspect-*`，旧任务保持不变。

后续需要执行时，从最新正确 AIR 的批次恢复即可复用导出。已完成的 compilation 若参数不同，则只使 compilation/bundle 失效，并保留权重、参考和 AIR。具体完整尺寸的编译支持不能从 tiny 成功直接推定。

如果首次准备权重尚未成功（`components` 为空），直接修正路径或文件后重跑上面的完整命令；这时没有已完成的权重、导出或编译需要复用。

主机验证覆盖draft-only准备、六层参考计算、GE构图/导出、阶段恢复、外置权重绑定、原生输入检查与模拟执行失败。用户日志确认v4完整OMG与CPUCL初始化成功；模型Load仍在NPU子图权重内存检查失败，完整执行和数值结果尚未取得。

## 与已用过的 4B draft 对照

用户指定参考为 [`feature/gdr-chunk-verify`，提交9018673](https://github.com/day8reak/qwen3.5-4B-dflash/tree/90186732d915e8b058b2a053d205991d805eae76)。此前检查的 `framework/quant-air-om` 和 `main` 仅是历史对照，不再用于解释20ms路径。

该分支[结果记录](https://github.com/day8reak/qwen3.5-4B-dflash/blob/90186732d915e8b058b2a053d205991d805eae76/docs/RESULTS.md)给出六层FP16 draft **20.22 ms/call**，是C16/C64 context加16行block的同步OM混合均值；W8 runtime为63.52ms、W8离线NZ为47.89ms、W4 runtime为94.68ms，后三者使用另外的五层检查点。数据来自用户设备报告，状态为 `PASS_WITH_DIFFERENCES`，ordinary token一致性未通过，任务质量未评估。这些是310P的观测，不能直接当作Cpro性能。

源码可借鉴的结构有：`PackedDraftLayer`合并K/V及gate/up权重，context与block合并执行一次KV投影；GQA将Q分组折入行维，避免四份KV复制；每轮只持久写入新增已提交context的KV，block KV只临时参与注意力。当前六层图的逻辑投影调用为56次，合并后可为32次；这是源码级计数，不是已编译kernel数量或耗时承诺。当前独立入口仍为prefix-recompute，增量缓存需要同步修改ABI、状态和调度。

针对本次BOOL掩码故障，已检查该参考分支的integrated与incremental实现，两者仍通过 `key_valid.expand` 或mask expand复制维度，没有看到针对Cpro CPUCL BroadcastTo初始化的显式替代。它的310P执行记录不能证明当前Cpro节点兼容；本次选择沿用tiny中Gather精确复制的思路，增加BOOL/INT32转换以满足数据类型契约，另行上板验证。

两版FP16 draft均为6层、block16、32个Q头/8个KV头、head_dim128；4B hidden/intermediate/feature_width为2560/9216/20480，本版为2048/6144/16384。4B路径保留FP32 RMS归约、score/softmax，与当前FP16参考需分别核对。其W4/W8是group128、FP16激活的WeightQuantBatchMatmul路径，不能套用原target的A8W4/I4ZN。融合head与新W8 kernel在参考仓库仍是规划；当前head是完整词表linear加argmax。
