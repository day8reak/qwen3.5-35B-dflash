# 当前阶段测试命令

当前任务：用已有算子组合实现六层Draft attention，重新验证完整Draft精度。配置为C16／cap192／K15、17输入／14输出；学习权重、GQA分组、mask、KV提交规则和验收门槛保持。特征投影仍为16384→2048的4×4 Conv2D，本轮未处理其性能问题。

新导出为 Gather → QK MatMul → FP32缩放／mask／Softmax → FP16概率 → PV MatMul，不注册或调用 AdnFusedInferAttention。计算192个context槽＋16个临时候选槽，共208行，通过原有mask选择有效行，不再使用NZ分页或补齐224行。无效query保留anchor witness，RMS保留 npu_rms_norm → RmsNorm。

旧融合矩阵两轮合计比较127／129个不同case：114失败、13通过；KV1执行失败、KV15未执行。Q16／32／64／1分别匹配对应长度的KV前缀，mask和页表有响应，物理原因未定。用户选择暂缓该调查，先做组合版完整精度。历史结果见[当前结果](results.md)。

本地440次缩小宽度六层调用与原组合实现逐位一致，四档真实尺寸FX捕获为12 MatMul、6 FP32 Softmax、32 RMS、17／14接口。独立完整前缀oracle的hidden／KV通过原容差，但有4个Top1差异，旧组合实现同样存在，故未标为整体精度通过。新TorchAir／OMG和板端结果待回传，原≤50ms目标仍未达到。

## 1. 原Target导出环境：重新导出Draft

在此前成功导出Target／Draft AIR、已有PyTorch／torch_npu／TorchAir的权重服务器执行。REPO_DIR、AI_RUN_DIR沿用该机路径，运行目录位于仓库外；若换终端，先设置这两个变量。

已有 draft/source/prepared/prepared.json 时直接复用。仅首次缺少prepared时，设置 TARGET_WEIGHTS 为原Target YAML的 model_file 对应的原始BF16权重目录，DRAFT_WEIGHTS 为原始DFlash检查点目录，再执行：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/prepare_draft.py" \
  --target-weights "$TARGET_WEIGHTS" --draft-weights "$DRAFT_WEIGHTS" \
  --output "$AI_RUN_DIR/draft/source/prepared"
```

更新代码并重新导出；旧融合AIR不能作为本轮输入：

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/export_incremental_draft.py" \
  --frontend pytorch \
  --prepared "$AI_RUN_DIR/draft/source/prepared" \
  --context-rows 16 --cache-capacity 192 --max-draft-tokens 15 \
  --output "$AI_RUN_DIR/draft/air" --replace
```

预期 PYTORCH_DRAFT_AIR_EXPORTED、attention.policy=composed-attention-v1、attention_nodes=0、composed_attention_layers=6、rms_nodes=32、17输入／14输出。air-inspection.json记录六条QK → FP32 Softmax → PV连接；发现融合attention、缺失连接、Softmax变成FP16或已记录KV维度缩为前缀时拒绝发布。

导出先写固定 draft/tests/export，成功后移动到 draft/air，失败保留旧公共AIR。源图缺失中间shape不被当作真实形状证明。

## 2. 传输新AIR并安装

将权重服务器的 draft/air 整目录传到DDK服务器的 $AI_RUN_DIR/draft/.air-incoming，直接包含 export.air、weight/、air-schema.pb、manifest.json、air-inspection.json。两机根路径可以不同，权重相对路径保持；若同机导出、编译则跳过此步骤。

在DDK服务器 t00826215_atc_mobile 的 custom_pypto 环境执行：

```bash
git -C "$REPO_DIR" pull --ff-only
python3 "$REPO_DIR/targets/cpro/scripts/export_incremental_draft.py" \
  --import-air "$AI_RUN_DIR/draft/.air-incoming" \
  --output "$AI_RUN_DIR/draft/air" --replace
```

返回 PYTORCH_DRAFT_AIR_INSTALLED 后再编译。安装使用目录移动，不额外复制学习权重。

## 3. DDK服务器：编译完整候选OMC

沿用 CPRO_DDK_ROOT、CPRO_DDK_ID 与已存在的 common/tbe-diagnosis 完整目录。仅当诊断目录缺失时执行：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/direct_model_air.py" check-tbe-platform \
  --ddk-root "$CPRO_DDK_ROOT" --python "$(command -v python3)" \
  --diagnose --jobs 4 --output "$AI_RUN_DIR/common/tbe-diagnosis"
```

编译复用固定候选槽，原公共 draft/omc 与运行配置保留：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/compile_incremental_draft.py" \
  --air-export "$AI_RUN_DIR/draft/air" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --tbe-diagnosis "$AI_RUN_DIR/common/tbe-diagnosis" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-omc" --replace
```

预期 PYTORCH_DRAFT_OMC_COMPILED。继续执行 draft-static-preparation-v5：Reshape链简化、静态ZerosLike／Fill物化、二维权重转置吸收到MatMul属性、已验证的embedding单位Muls旁路。

可读OMC主图若出现融合attention则拒绝打包；GraphOp不透明内核未解码，不能由主图证明内部指令或精度。编译失败时只读查看原因：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/compile_incremental_draft.py" \
  --inspect-compile "$AI_RUN_DIR/draft/tests/pytorch-omc"
```

## 4. DDK服务器：完整Draft独立精度

使用现有 runtime/runtime.json 和已成功运行过的 qwen35_draft_accuracy native；本轮未改runner，无需重建。--source必须与AIR学习权重一致。入口绑定实际编译槽位，生成独立完整前缀参考，再执行 isolated 和 feedback 两组，共38次调用：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --config "$AI_RUN_DIR/runtime/runtime.json" \
  --compiled "$AI_RUN_DIR/draft/tests/pytorch-omc" \
  --source "$AI_RUN_DIR/draft/source/prepared" \
  --air-export "$AI_RUN_DIR/draft/air" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-run" --apply
```

启动返回后回收同一次请求；PENDING时继续recover，不重复apply：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --recover "$AI_RUN_DIR/draft/tests/pytorch-run" \
  --output "$AI_RUN_DIR/draft/tests/pytorch-result"
```

入口只复用已知结束的固定测试槽，运行中或状态不明的目录不可覆盖。通过条件：两组各19例全部通过，各86个有效候选Top1完全一致，hidden及每层K/V维持 atol=0.01、rtol=0.01，KV未写区逐位保持。覆盖空context、初始追赶、部分块、拒绝候选、重复调用、增长与容量边界。

这里验证完整Draft在相同确定性非零合成features下的计算精度，尚不证明Target特征精度、文本质量、接受率或端侧闭环。新候选不自动替换生产模型；先回传本轮编译摘要和整体精度结果。
