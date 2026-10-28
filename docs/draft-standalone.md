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
4. 只编译draft OMC，保留外置数据；恢复成功tiny路径的 `--use_origin_format=true --weight_data_type=FP16`。每30秒显示编译等待及日志信息。
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

用户回传 `full-draft-d7b4db958850` 已完成71个权重、PyTorch参考和 AIR 导出；OMG 在首个 `FileConstant` 报告未识别 `dtype`，并将绝对权重路径拼在编译目录之后。新 draft 导出将权重引用写为 `weights/wNNNNN.bin`，从 AIR 父目录执行序列化，并显式使用整数 GE dtype ID 的 CPro 兼容策略。它不改变张量内容；通用 GE 默认 Type 属性保持原样。该兼容策略已纳入清单版本，后续 `full-draft-5133743f7c9c` 的日志已越过外置权重读取。

最新批次 `full-draft-5133743f7c9c` 在首个 MatMul 兼容检查处失败。完整 draft 编译已恢复 tiny 成功路径的两个 FP16 参数；从此批次恢复会复用现有 AIR，无需再次导出。已完成的 compilation 若参数不同，则只使 compilation/bundle 失效，并保留权重、参考和 AIR。具体完整尺寸的编译支持仍由实际 OMG 重试确认，不能从 tiny 成功直接推定。

如果首次准备权重尚未成功（`components` 为空），直接修正路径或文件后重跑上面的完整命令；这时没有已完成的权重、导出或编译需要复用。

主机验证覆盖draft-only准备、六层参考计算、GE构图/导出、阶段恢复、外置权重绑定、原生输入检查与模拟执行失败；真实DDK和板端执行仍待服务器结果。
