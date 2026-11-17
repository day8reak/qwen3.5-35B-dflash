# 四层 Target Prefill：同次输入与 golden

单次执行原 310P 的前四层、原 norm 和 LM head，在 wrapper 入口和返回时复制输入／输出，避免 Decode 覆盖 Prefill 文件或缓存别名改变 golden。保留原 W4 量化和算子调用，不重新导出 AIR／OMC。

## 1. 准备

使用**原来已跑通四层推理的 310P Python／torch_npu 环境**，保留原自定义算子加载配置；仓库路径约定见[环境配置](environment-setup.md)。此步骤不是 DDK 的仅导出环境。原源码、权重和新 bin 都留在仓库外。

```bash
export REPO_DIR='/实际路径/qwen3.5-35B-dflash'
export AI_RUN_DIR="${REPO_DIR}-run"
export TARGET_SOURCE='/实际路径/qwen35_moe'
export TARGET_CONFIG="$TARGET_SOURCE/config/qwen3_5_moe.yaml"
export PREFILL_FIXTURE="$AI_RUN_DIR/target/tests/prefill-fixture"
```

脚本默认从 YAML 的 `prompts` 列表取第 1 条，也支持单个 `prompt` 字符串，用 `model_file` 对应的 tokenizer 编码。`input_ids` 就是编码后的 token 编号，长度自动计算；无需自己准备 bin。YAML 例如：

```yaml
prompts:
  - "南京著名的景点有哪些，请列出五个。"
```

默认使用原文一次；不补齐、不截断，实际 token 数必须在 2 到配置缓存容量之间。原 inference.py 会用 `prompt * 32` 覆盖聊天模板结果；本入口同样不套聊天模板，保留原 tokenizer 的特殊 token 默认设置，但只有显式传 `--prompt-repeat 32` 才重复原文。选择其他条目用 `--prompt-index 1`（从 0 开始）。

YAML 的相对资产路径沿原程序规则，相对于 `TARGET_SOURCE`；使用绝对权重路径最直接。脚本构造四层后量化并搬到 NPU，原 loader 仍可能读取完整 checkpoint。

## 2. 一次生成

`PREFILL_FIXTURE` 应为尚不存在的输出目录。

```bash
python "$REPO_DIR/targets/cpro/tools/generate_target_prefill.py" \
  --source-root "$TARGET_SOURCE" \
  --config "$TARGET_CONFIG" \
  --device npu:0 --intermediates --capture-boundaries \
  --output "$PREFILL_FIXTURE"
```

成功显示 `status=CAPTURED`、实际 `sequence_length`、16 个语义输入、9 个输出，并保存 `input_ids.bin`。对应整网 OMC 的 Prefill 档位须匹配这个实际长度；指定 `--sequence-length 64` 只检查 token 数恰好为 64，不会改 prompt。已有 token 文件时，仍可添加 `--input-ids /实际路径/input_ids.bin`，文件必须为无 padding 的小端 INT32，长度从字节数自动计算。

`--intermediates` 额外保存各层输入、Norm、GDN／Attention、MoE 和层输出；`--capture-boundaries` 同时记录已有 probe 所需的边界与投影权重，供随后导出整层 GDN／完整 Attention 对照。整网 Prefill 只执行一次，不执行 Decode。所有缓存初态为零，实际 causal mask、position、`allQLen`、`effective_length`、`output_pos` 都在调用前核对。

启用 `--capture-boundaries` 时，采集结束后还会用独立复制的输入／旧缓存重放一次 L3 完整 Attention 模块，仅将 Attention core 换为已审查的独立数学公式，另存三路模块参考。原整网输出及原模块观测不覆盖；这份对照用于区分原 310P Attention core 的偏差与 CPro 整模块偏差，不包括 MoE。

采集与独立参考计算的耗时不作为模型性能数据。若边界采集或模块参考不完整，命令显示 `PARTIAL` 并保留已完成的整网快照与具体原因。

若仅辅助模块重放报 K/V 不一致，而 `attention_audit=PASSED`、边界为 `CAPTURED`，可复用本次数据按[模块分组对比](target-attention-pair.md)的 `captured-audited` 模式继续，无需重跑整网。重放失败不会覆盖原输出；新采集还会在 `attention_math_replays` 中保存差异指标、有效／未写缓存区统计及 `diagnostic_outputs`，这些诊断文件不作为 golden。

| 文件 | 用途 |
|---|---|
| `manifest.json` | 实际 shape/dtype/字节数、源文件身份、输入转换、审计结论 |
| `input_ids.bin` | 本次实际 token 编号，小端 INT32；来源、prompt 选项与实际长度记录在 manifest |
| `inputs/*.bin` | 同次 Prefill 的 16 路具名输入，用于绑定 CPro 实际输入槽位 |
| `source_inputs/*.bin` | 原 310P wrapper 实际参数快照，保留源 dtype |
| `golden/*.bin` | 同次原 wrapper 返回的 9 个输出，立即独立复制 |
| `intermediates/*.bin` | 指定 `--intermediates` 时的层边界快照 |
| `capture/manifest.json` | 指定 `--capture-boundaries` 时，供 `target_layer_probe.py export --capture` 直接使用 |
| `attention-audit/` | 原 Attention 调用与独立数学参考，仅诊断，不替换原输出 |
| `original/` | 原 inference.py 自行写出的文件；比较使用上面的具名快照 |

## 3. 对齐板端输入、输出

`inputs/` 的八个非状态输入是 `embed_scale`、`inputs_embeds`、`new_kv_cache_pos`、`effective_length`、`position_ids`、`attention_mask`、`allQLen`、`output_pos`。另有三对 `conv_state_0..2`／`recurrent_state_0..2` 和 `key_cache_3`／`value_cache_3`。

`position_ids`、`new_kv_cache_pos` 从原 INT64 按值转为 CPro 的 INT32；原值及原 dtype 留在 `source_inputs/`。`allQLen` 从原 `[SEQ_LEN]` 列表序列化为 INT64[1]，其余输入保留实际 dtype。这里匹配显式展开 ConvState 的新模型，**没有 `linear_update_tensor`**。

从实际 OMC 查询 I/O 后，按模型语义绑定这些文件，保持 `--input`、`--input_type` 和 runtime 槽位一致；不要按文件名排序、`Data.index` 或同 shape 猜状态次序。16 是该参考的具名输入数，最终编译接口仍以实际模型为准。

```bash
python "$REPO_DIR/targets/cpro/tools/parse_model_json.py" \
  '/实际路径/四层模型.omc' \
  --output-dir "$AI_RUN_DIR/target/tests/prefill-io"
```

原 wrapper 输出顺序为：`golden_logits`，L0 conv/recurrent，L1 conv/recurrent，L2 conv/recurrent，L3 key/value。文件写在 `manifest.json.outputs` 的 `source_tuple_index` 中；CPro 的实际输出槽位也须核对。前三层的状态不是普通 Attention K/V。比较 L3 KV 时还要单独检查本次有效写入区域，不能用大量未写入零值的相等比例判通过。

`CAPTURED` 只表示同次输入／输出已保存。原 NZ Prefill Attention 曾与独立数学参考不符，所以脚本额外审计实际 Attention 输入与输出：`attention_audit=FAILED` 表示参考边界仍有数值偏差，`UNAVAILABLE` 表示当前调用不满足已审查的参考合同；两者均不能当成可信整网 golden 已验证。即使审计 `PASSED`，也只验证该 Attention 边界，整网精度仍需与 CPro 同输入比较。回传 `manifest.json` 和新的板端比较结果即可继续定位。

模块定位的下一步见[分别验证 Linear Attention 与 Full Attention](target-attention-pair.md)，直接复用这次 `capture/`。
