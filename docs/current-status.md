# 当前更新与执行命令

[首页](README.md) · [环境配置](environment-setup.md) · [验证记录](validation.md)

更新日期：2026-10-28。用户将当前优先级调整为先独立跑通完整6层DFlash draft；保留正在进行的40层target编译。最终仍需接通target验证、连续生成与性能比较。

## 用户现在执行什么

**下一步直接执行完整 draft：V2 文件恢复、初始化、推理和参考比较在同一次命令中完成。** 已有固定目录保存成功的external OMC与bundle时，复用全部模型阶段，只重建runner和本次board结果：

```bash
git pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/run_full_draft.py" --model-load-api file-v2
```

用户回传的所选DDK头文件及动态符号已经确认 `CreateBuiltModel`、`IBuiltModel::RestoreFromFile`、`CreateModelManager` 和 `IModelManager::Init/Run`。新入口用完整OMC路径恢复模型，保留与15份外置权重的相对目录关系，随后初始化和执行。它不再使用互相清空字段的legacy描述符setter组合；`InitWeights` 的调用时序与用途未由当前声明证明，因此没有猜测调用或自动切换加载方式。实际外置权重支持、加载和数值结果仍需这次设备运行确认。

本次不再运行描述符或API探针、不重编译AIR/OMC、不重新准备权重或PyTorch参考。板端上传当前runner、所选运行库及已有完整bundle（包括外置权重），执行一次完整draft；成功后自动回收并比较输出。失败自动显示 `runtime_stage`、本进程日志，以及明确提及runner PID的hiaiserver失败上下文；20MB以上HiLog也按流读取并选择有界摘要。若当前SDK编译接口不同，构建错误及所选ND头文件自动保留并输出，不需要先手工查询。

默认external选择 `file-v2`，embedded仍选择 `legacy-buffer`；显式选择加载API仅影响board阶段。固定目录会替换旧board结果，成功模型阶段保持复用。同步V2 `Run` 没有超时参数；HDC等待超时不保证远端进程已被取消。

[独立 draft 指南](draft-standalone.md)保留完整模型入口。它只读取draft全部权重及target的embedding/LM head，不构造或运行40层target。默认完整6层、block16、248320词表，输出15个候选ID和pre-head hidden诊断值。使用固定合成target features，因此这一步不证明语言质量、接受率或加速。

最新内嵌重试 **`full-draft-6599025a0b74`** 在 `Compile end` 后保存模型失败：`SaveToBuffer/CreateCompiledModelBuffer` 中的 `LargeMemoryOps::Copy` 报 `memcpy_s` 失败，经 `SetModelHeader/CopyDataToBuffer` 返回保存失败。它没有进入板端执行；目前只能确认大缓冲区序列化失败，不能确定是某个固定大小上限或其他原因，也不会自动改回external重跑。

按用户要求，入口现在默认使用固定 **`$AI_RUN_DIR/full-draft`**。此前一次迁移命令如下；已经完成迁移时不再执行它。它从732迁入已完成权重、参考、v4 AIR，以及已经成功编译的external OMC和bundle，不编译或上板：

```bash
git pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/run_full_draft.py" \
  --resume "$AI_RUN_DIR/full-draft-732b26335347" \
  --prepare-only
```

迁移采用同一文件系统内的目录重命名，不复制大权重；保留原清单字节及哈希，并将组件目录改为固定目录下的路径。从732继承external模式，保留现有成功编译和bundle；6599的embedded保存失败仍作为最新故障记录。成功后后续命令省略 `--resume`，自动从固定目录继续。成功阶段复用；失败或失效阶段及旧board输出会在原位置清理重建，最新日志和报告覆盖旧版，不再默认生成新批次目录。并发运行会被锁拒绝，中断的迁移通过日志恢复。迁入完成且组件均已位于固定目录后，旧批次目录可按用户选择清理；入口不自动删除整个旧目录，也不删除独立TBE诊断目录。固定目录仍需容纳一次编译的输入暂存和输出，不能替代释放已满磁盘空间。

此前 **`full-draft-732b26335347`** 的external模式已完成v4 AIR编译和打包。PID46426日志显示CPUCL初始化成功，越过 `v501/BroadcastTo` 障碍；随后NPU `SubGraph_1` 在 `InitRuntimeParams` 报 `weightsMemBaseSize=16`、`modelWeightMemorySize=1151486976`。尚无Process或数值结果。6599继续复用该v4 AIR和 `d7b4db958850` prepared/fixture，本次迁移不改变图、权重、参考或加载行为。

运行时给15个不同子图打印了同一个 `weight hash:0x1dabbe2881ec5472`，但它不是文件SHA256，不能据此断言15份权重内容相同或确定错配原因。实际文件哈希尚未回传。本仓库只读审查未发现按编号重排或丢失目录：编译到bundle再到上传均按完整相对路径处理；这仍不能证明设备HIAI打开了这些路径。

用户随后列出的编译文件中，`SubGraph_1.weight` 是 **1,151,486,976字节**，与模型要求一致，`model.omc` 是2,905,990字节；因此这份编译文件本身不是16字节。文件大小不能替代SHA256或证明运行时选中了正确文件。runner的“路径→buffer”调用会丢失路径，已由本次getter观察确认；外置权重文件的实际选择和加载契约仍待核对，见[验证记录](validation.md#外置模型加载描述符诊断入口)。此前成功tiny没有开启外置权重选项，不能直接验证这条加载契约；目前没有已确认的运行时修复。

原生失败保存在 `board/results/result.json`、`board/comparison.json` 与 `board/draft.stderr`。错误回传已修正：保留原生失败原因，单独记录传输/证据问题，运行失败时不要求成功输出bin。此前 `a091f16d13d8` 的嵌套OMC输出路径问题已修复。

4B参考已按用户指定更正为 `feature/gdr-chunk-verify`、提交 `9018673`。该分支记录六层FP16 draft为20.22 ms/次（310P、C16/C64混合均值）；源码包含合并K/V、gate/up投影和增量缓存。先完成当前独立draft的板端输出验证，再按这版迁移图结构与缓存，性能须在Cpro重新测量。

## 保留的 target 编译与诊断

**已经在编译的任务先保留，用另一个终端观察。** 旧脚本在 `Compiling existing AIR` 后将输出写入 `compilation/compile.log`，直到 OMG 退出才刷新终端；默认7200秒只是超时上限，不是预计完成时间。十几分钟没有终端输出不能单独判定卡死。查看本次目录里的日志尾部、文件更新时间以及 OMG/编译子进程的 CPU 累计耗时、内存和等待状态；日志静默也可能来自缓冲。先定位当前阶段，不重复启动另一个整模编译。

新入口等待期间每30秒显示已用时间、日志字节数、最近写入时间距今秒数及有长度限制的最后一行。这是等待观察，不是完成百分比或编译正常的判定；监控失败不会重启、终止或改写编译结果。它只影响以后启动的任务，正在运行的旧进程继续通过日志观察。

**上一轮外置权重暂存问题已修正。** `full-target-compile-e019e4a7ad53` 已通过输出参数校验并开始解析 AIR，在 `arg3_1` 的 `ConvertFileConstant` 处找不到 `./glm_export_air_cpro_40block_55_quant/model_model_layers_0_input_layernorm_weight`。用户确认这些权重就在原 `export.air` 同级目录；旧收集逻辑漏掉无扩展名文件，暂存布局也缺少引用里的导出目录前缀。

入口现在收集同级无扩展名的普通文件，保持 AIR 位于编译工作目录，将权重快照放在其下的原导出目录名中。默认 `--input-layout export-directory`；其它 AIR 若使用不带导出目录名的相对路径，可显式选择 `--input-layout air-directory`。不改写 AIR，不借用原目录链接，也不重新转换 YAML 指定的量化权重。以下保留启动命令供后续需要时使用；已有编译运行时不要重复启动：

```bash
git pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/compile_full_target.py" \
  --air /home/l00886453/qwen3_5_moe/glm_export_air_cpro_40block_55_quant/export.air \
  --tbe-diagnosis "$AI_RUN_DIR/fused-all-f6b81ead7ae8/diagnosis"
```

脚本内置用户提供的89个输入形状、80个`RefData`状态输入和5个动态轴，使用 `--dynamic_dims="64,64,64,64,64;1,1,1,1,1"`，保留 `--limit_partition_size=false` 和 `--save_weights_as_external_data=true`。原命令中的 `--d11ynamic_dims` 按拼写错误处理；执行前记录所选OMG帮助。帮助未列出的内部参数仍按用户命令原样交给实际编译器验证，不偷偷删参数。默认调用当前DDK的 `tools/tools_omg/master/omg` 实际程序，沿用已经成功的隔离环境，避免外层脚本修改共享工具目录；不添加FP16权重强制转换或探针的布局选项。

前一批次 `full-target-compile-771648b65da5` 的 `CheckOutputPathValid` 错误已通过短名称 `--output=model` 绕过，最新日志确认已进入 AIR 解析。本次权重修复继续保持短输出名。编译前会显示所选文件数量、总字节和权重相对前缀；`input-inventory.json` 记录快照哈希，输入文件不会被当作新生成 OMC 外置数据。文件收集仍有目录/数量限制，完整依赖尚需实际编译验证。

失败时入口直接打印有长度限制的日志摘要；完整异常和参数保留在 `result.json`、`compilation/compile-command.json`。帮助阶段失败显示 `help.log`，无需为了查看错误重新运行工具。

每次输出到新的 `full-target-compile-*`，保留 `compilation/compile.log`、完整命令、输入快照、OMC及生成的外置数据文件。编译成功后尝试读取编译图接口，报告在 `interface/report.txt`；工具不支持或转换失败单独报告，不改写编译成功状态。文件名和用户形状声明不等于实际AIR验证。原AIR保持原图，**本仓库基础Conv/RMS替代不会自动修改这份AIR**；以真实编译及图检查结果确认节点、状态和量化接口。

这条入口开始整模编译，不自动套用探针runner。原AIR的动态档位、输入顺序/dtype、RefData持久状态和外置权重装载还需与实际HIAI接口对齐后才能上板。

**三填充诊断已完成，不需要再续跑或重跑这批探针。** 用户回传 `gdn-diagnostic-138b60d3eba7` 为 `DIAGNOSIS_COMPLETE`：原 `gdn-diagnostic-c0cbac5ac9cf` 的零填充结果复核完成，另两轮执行完成，三轮host输入回读SHA256一致。保留这两个诊断目录和原 `fused-all-f6b81ead7ae8` 批次，后续直接分析已有证据。

同一OMC/输入的GDN输出随预填充值变化。零填充时prefill/decode核心relative L2约为0.00392/0.0564，末态约0.146/0.155；A5填充时核心升至3.44/11.19；5A填充时出现约203.25的误差，与FP16位模式 `0x5A5A` 的值一致。有效token区域已排除padding。当前证据不能区分部分写、输出累加和缓存可见性；清零不是已验证修复，数值验收仍未通过。上述结果来自用户报告，原始远端文件尚未在本地重新核验。

target原定里程碑为OMC编译与一次普通prefill/decode；当前先完成上述独立draft步骤，保留target已有编译及诊断。现成AIR路线绕过当前直接GE的W4A8重建缺口，但不能据此宣称整模已可上板。作为备选，新增加的权重准备器可流式生成完整target的逻辑INT4/INT8/FP16缓存，支持中断后校验复用；其约18.1GB数据尚非I4ZN/QPARAM，也不是板端峰值内存。原量化权重的 PyTorch/GE 路径现已接入，详见[原量化模型接入](original-quantization.md)；真实DDK量化绑定和QPARAM转换仍需实际环境提供。FP16约69.3GB不适合32GB板端。`--expert-dispatch selected`只减少计算，不减少常驻权重。现有自建stateful图的 `past_length` 静态；原AIR的动态状态契约需要独立检查。

target后续工作：按用户原转换契约完成现有AIR编译、读取实际图接口、接通状态执行；继续分析GDN读写问题。暂不增加Conv/RMS候选、重复平台诊断或新的性能矩阵。需要实机验证时给单一入口命令。

模型范围仍保留GDN、CacheUpdate、Attention三类六例，Conv/RMS使用基础算子。只有确需重新执行全部六例时才使用 `run_fused_probes.py --model-ops`；旧九例 `--all-ops` 保留为可选诊断，原Conv/RMS失败不改为通过。

原工程中 `LinearAttentionConvstateUpdate` 用于Cpro，`AdnRmsNorm` 用于ND，Cpro的RMS调用内置 `npu_rms_norm`。本仓库当前选择现有基础算子实现Conv历史拼接/截取和RMS，生产GE只启用 `--fused-ops gdn`。CacheUpdate/Attention的独立探针通过尚不等于已接入生产GE模型；完整模型及数值验证继续保留。

若再次进入Docker，恢复conda/DDK后执行 `source targets/cpro/scripts/env.sh` 即可；配置含自动生成的HIAI头文件及三库路径，详见[环境恢复指南](environment-setup.md#docker-重进后的配置恢复)。上述命令要求保留当前批次，保留权重无需重新准备。

## 当前状态

| 项目 | 状态 |
|---|---|
| PyTorch完整结构 | 原量化 target 按 Cpro forward 对齐 FP32 residual/门控/RoPE；新增原风格 embedding/位置选择入口，功能式状态接续通过主机测试 |
| GE完整结构 | 原W4量化target完整40层64/1构图；同步 FP32 计算边界与指定位置完整 logits，实际DDK量化绑定和设备验证待完成 |
| 自定义算子探针 | 当前选择GDN/CacheUpdate/Attention三类6例，用户回传导出、编译、加载及执行通过；原Conv/RMS三例失败保留为可选诊断 |
| 真实权重AIR导出 | 旧双图支持完整FP16；新GE target支持显式缓存 |
| 原工程完整AIR | 已进入 AIR 解析；缺失外置权重快照及目录前缀的问题已修复，真实 OMG 重试待执行；未执行设备推理 |
| 原量化缓存 | 按原YAML读取已有W4及scale，I4ZN逐字节参考对照通过；专家QPARAM缺失时明确报告pending，不重新量化 |
| 状态执行 | 主机fake prefill→decode接续通过；真实板端待验证 |
| 尚未完成 | 真实DDK量化契约/目标执行、常驻缓存与native DFlash调度、完整模型设备精度和性能 |

## 已确认与剩余问题

| 项目 | 结论 |
|---|---|
| GDN两例 | 三填充诊断已完成，输出依赖预填充值；零填充改善但末态relL2仍约0.146/0.155，未通过精度验收，原因未定位 |
| 编译目标 | 始终为 `kirinx90`；`Ascend310B1` 是DDK初始化后返回的内部名称，`Ascend310P3` 是原工程平台 |
| GDN状态 | 安装版初态/末态为FP16；本次runner修正了末态只接受FP32的旧校验 |
| conv | 当前走基础算子的历史拼接/截取路径；原Cpro自定义Conv两例在rank-4 y2维推后格式配置失败，历史结论不变 |
| RMS | 当前走基础算子RMS路径；原ND用的 `AdnRmsNorm` 探针缺维推函数，保留失败；不再尝试其他融合RMS候选 |
| CacheUpdate | prefill/decode调用通过，比较仅为SHAPE_ONLY_UNVERIFIED_CONTRACT，无数值精度通过结论 |
| Attention | prefill/decode调用通过，比较仅为SHAPE_ONLY_UNVERIFIED_CONTRACT；先前导出器问题不再阻止调用，精度仍未验证 |
| 尚未验证 | 当前六例的正确数值及NPU执行位置；CacheUpdate/Attention生产GE接入、基础Conv/RMS的完整模型目标验证与连续状态验证仍待完成 |

新增调用源码的[核对与勘误](qwen35-moe-operators-and-quantization-analysis.md#调用源码核对与勘误2026-10-27)确认：Attention converter把Python `pse_shift`实参接到GE端口16，端口3留空，当前探针方向正确；Conv的输出rank、GDN的Meta/安装版状态dtype和Cache的Functionalize调用名仍有差异，不能凭前端声明判定安装版行为。

## 完整模型构图（按需）

需要完整模型结构时选择下面一条路径；`sequence-length=8`只限定输入长度，模型层数、专家和词表均为完整尺寸。

**PyTorch：** 生成完整meta模型和层级FX结构，不需要检查点或TorchAir。

```bash
python3 -m qwen35_moe_dflash.cli build-full-pytorch \
  --sequence-length 8 --output "$AI_RUN_DIR/torch-full-$(date +%Y%m%d-%H%M%S)"
```

**GE：** 生成完整target/draft构图源码，仅选择GDN融合，不需要检查点或GE库。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/direct_model_air.py" build-full-structure \
  --sequence-length 8 --expert-dispatch selected --fused-ops gdn \
  --bindings "$AI_RUN_DIR/fused-all-f6b81ead7ae8/bindings.json" \
  --output "$AI_RUN_DIR/ge-full-$(date +%Y%m%d-%H%M%S)"
```

这些是未绑定真实权重的结构产物。当前生产GE融合接口支持 `gdn,conv_state,rms` 三个角色，本次只选择 `gdn`，Conv状态和RMS保留基础算子。CacheUpdate/Attention尚未通过该接口接入生产图。实际导出AIR：按 [PyTorch](pytorch-guide.md)或 [GE](ge-guide.md)指南继续；无需重跑旧探针。

## 文档融合接口

本次fresh批次的 `bindings.json` 可复用于GE融合构图，需要时在构图命令前设置其路径。绑定包含文档端口与安装源码的GDN状态声明，不代表全部kernel已安装或已在目标芯片验证；融合构图命令见[GE指南](ge-guide.md)。

完整W4A8 + FP16 DFlash先按24–28GiB可用RAM规划；这不是实测峰值，也不适用于当前FP16权重导出，见[内存预算](environment-setup.md#内存预算文档-w4a8--dflash)。
