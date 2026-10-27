# 直接 GE 构图与执行

[首页](README.md) · [环境配置](environment-setup.md) · [端到端流程](project-structure.md#端到端如何运行)

统一命令入口是 `targets/cpro/scripts/direct_model_air.py`。Python/NumPy 描述图，生成 C++ 并调用真实 GE 库保存 AIR；构图不经过 PyTorch/TorchAir。PyTorch 只在需要对应框架参考时使用。

## 选择哪条路径

| 路径 | 命令 | 产物及用途 |
|---|---|---|
| 原始 target/draft 双图 | `prepare → export → compile` | HIAI ABI v2，可交给现有 runner；按前缀重算 |
| 新 target 状态图 | `plan-stateful` / `export-stateful` / `compile-stateful` / `prepare-stateful-step` | 显式 conv/recurrent/KV，融合可选；独立单步runner模式，尚未真实上板 |
| 文档融合接口 | `bindings-from-document` | 根据用户 MD §4.6 写绑定 JSON，不安装 kernel |

## 原始双图：准备、导出、编译

以下用于从头复现 tiny；已有 prepared/AIR/OMC 时复用对应目录，不必重复前序步骤。先设置环境页中的 `CANN_ROOT` 和目标 DDK 变量。

```bash
export GE_MODEL_TOOL="$REPO_DIR/targets/cpro/scripts/direct_model_air.py"
export GE_PAIR_RUN="$(mktemp -d "$AI_RUN_DIR/ge-pair-XXXXXXXX")"
"$MODEL_PYTHON" "$GE_MODEL_TOOL" prepare --tiny --sequence-length 8 \
  --output "$GE_PAIR_RUN/prepared"
"$MODEL_PYTHON" "$GE_MODEL_TOOL" export \
  --prepared "$GE_PAIR_RUN/prepared" --ge-root "$CANN_ROOT" \
  --output "$GE_PAIR_RUN/air"
"$MODEL_PYTHON" "$GE_MODEL_TOOL" compile \
  --air "$GE_PAIR_RUN/air" --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$GE_PAIR_RUN/omc"
```

依次生成 `PREPARED`、`EXPORTED`、`COMPILED` 清单。prepare 是权重和构图输入准备，不能当成 AIR；export 需要真正的主机 GE 库。默认采用 native lowering；显式的 `--fp16-reductions matmul`、`--fp16-row-mul compact`、`--gdn-indexed-copy gather` 是已有诊断/改写选项，不是最终精度或性能保证。

在模型 Python 环境生成同一份 prepared 的参考，再交给[板端指南](board-guide.md)：

```bash
"$MODEL_PYTHON" "$GE_MODEL_TOOL" reference-tiny \
  --prepared "$GE_PAIR_RUN/prepared" --prompt-token-ids 1,2,3 --max-new-tokens 4 \
  --output "$GE_PAIR_RUN/reference.json"
```

tiny 为 target4层、draft2层、hidden16、词表64、block4，只验证结构和接口。完整结构使用 `plan --full`；准备完整 FP16 权重须显式 `prepare --full --float-baseline` 并提供 `--target-weights`、`--draft-weights`。这是服务器转换诊断，尚非32GB板机部署方案；直接 GE 的压缩量化映射未完成。

## 状态图与可选融合

已有 prepared 时，先生成源码和 ABI，不调用 GE 或设备：

```bash
"$MODEL_PYTHON" "$REPO_DIR/targets/cpro/scripts/direct_model_air.py" plan-stateful \
  --prepared "$GE_PAIR_RUN/prepared" --sequence-length 3 --past-length 0 \
  --fused-ops none --output "$AI_RUN_DIR/ge-stateful-source-$(date +%Y%m%d-%H%M%S)"
```

`--fused-ops` 可选 `none`，或 `gdn,conv_state,rms` 的任意不重复组合；选用的角色缺少绑定时直接报错。

| 角色 | 关闭时 | 开启时 |
|---|---|---|
| gdn | FP16 chunk/recurrent 基础图 | `ChunkGatedDeltaRule`，含 Q/K L2、显式有效长度和状态 |
| conv_state | 历史与新 QKV 拼接 | `LinearAttentionConvstateUpdate`，从 y2 尾部取规范化历史 |
| rms | 基础 RMS 运算 | `AdnRmsNorm`，主 norm 的 gamma=1+w，gated norm 的 gamma=w |

卷积运算、SiLU、gated norm 外的门控、投影、MoE、全注意力和 KV 追加仍在各自图边界。GDN 融合不代表整层全部融合。

接口生成命令集中在[当前待执行命令](current-status.md#用户现在执行什么)。生成器按文档绑定 GDN 的 `last_recurrent_state`（FP32）、3个GE Int属性，RMS 的 `self`/FP32 `rstd`，以及 conv 的 `x1/x2/x3 → y1/y2`。卷积 y1 排列未说明，因此逻辑缓存取 y2（历史‖当前）最后 K−1 项。

绑定限定实际模型32头×128维、8192卷积通道，不能套用 tiny。完整形状 prepared 和匹配 GE/DDK 就绪时可按下例导出；`GE_FULL_PAIR_RUN` 指完整权重 prepared，`GE_CUSTOM_BINDINGS` 指生成的接口文件：

```bash
export GE_STATEFUL_RUN="$(mktemp -d "$AI_RUN_DIR/ge-stateful-XXXXXXXX")"
"$MODEL_PYTHON" "$REPO_DIR/targets/cpro/scripts/direct_model_air.py" export-stateful \
  --prepared "$GE_FULL_PAIR_RUN/prepared" --ge-root "$CANN_ROOT" \
  --sequence-length 3 --past-length 0 \
  --fused-ops gdn,conv_state,rms --bindings "$GE_CUSTOM_BINDINGS" \
  --recurrent-dtype float16 --gdn-state-casts allow \
  --output "$GE_STATEFUL_RUN/air"
```

该例使用文档 Cpro 的 FP16 初态、FP32 末态，显式允许末态转回 FP16 缓存；Cast 会记录于 ABI。也可显式选用其他文档初态类型与 FP32 缓存。转换不是精度验证，不自动改变原路径的精度合同。

## 状态图：编译和输入打包

获得真实 `EXPORTED` 的状态AIR后，使用独立命令，保留导出目录和外部权重原路径：

```bash
"$MODEL_PYTHON" "$REPO_DIR/targets/cpro/scripts/direct_model_air.py" compile-stateful \
  --air "$GE_STATEFUL_RUN/air" --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$GE_STATEFUL_RUN/omc"
"$MODEL_PYTHON" "$REPO_DIR/targets/cpro/scripts/direct_model_air.py" prepare-stateful-step \
  --omc "$GE_STATEFUL_RUN/omc" --token-ids 1,2,3 \
  --output "$GE_STATEFUL_RUN/step-prefill"
```

编译入口校验AIR、权重、recipe、状态ABI和bindings；source-only产物会被拒绝。打包生成 `step.tsv`、状态ABI、模型载荷和有哈希的输入文件，供正式runner的 `--stateful-step` 模式使用，见[板端说明](board-guide.md#6-新状态图的单步执行)。

prefill为S=3/T=0。继续decode需用相同权重与算子策略导出、编译S=1/T=3的状态图；将其目录设为 `GE_DECODE_OMC`，将上一步回收的完整结果目录设为 `PREFILL_RESULT`：

```bash
"$MODEL_PYTHON" "$REPO_DIR/targets/cpro/scripts/direct_model_air.py" prepare-stateful-step \
  --omc "$GE_DECODE_OMC" --token-ids "$NEXT_TOKEN_ID" \
  --previous-step "$GE_STATEFUL_RUN/step-prefill" --previous-result "$PREFILL_RESULT" \
  --output "$GE_STATEFUL_RUN/step-decode"
```

`NEXT_TOKEN_ID` 为要消费的下一个token。打包检查模型身份、位置、全部输出哈希及逐层state形状，接续上一轮conv/recurrent/KV；不接受把整段投机末态作为部分接受前缀。host-fake结果默认拒绝，主机联调需显式 `--allow-host-fake`。

这是状态接续的单步结构，尚不是常驻模型的高性能生成循环。完整35B部署还需真实量化权重映射与内存验证，不能直接以FP16全权重图作为32GB板端方案。

## 缓存、产物和运行边界

batch=1，S个输入均有效且连续。`--past-length T` 是静态参数：prefill为T=0，消费3个token后单token decode为S=1/T=3；T变化需要相应图。输出前两项为Top1和target特征，随后是逐层状态；`target/state-abi.json`记录全部输入输出的名称、顺序、shape、dtype与状态策略。

源码输出类型为 `qwen35-stateful-ge-source`，AIR输出为 `qwen35-stateful-ge-air`；均不是旧的target/draft双图ABI v2。**它们不能传给旧 `compile`、`run_hdc.py`；请使用 `compile-stateful` 与runner的独立 `--stateful-step` 模式。** 常驻状态bank和native DFlash调度尚未完成。 DFlash部分接受必须提交接受前缀的状态，不能直接提交整段verify的末态。

`CacheUpdate` 与 `AdnFusedInferAttention` 还需动态列表端口、可选口和NZ/分页布局；用户文档的长度列表与 `pse_shift/all_seq_lengths_q` 分支存在需确认的表述。当前固定ND端口实现没有仅凭名字启用这两个算子。

## 保留的离线诊断

统一入口的 `inspect-compile`、`inspect-omc`、`inspect-omc-cpu`、`compare-target-features` 用于读取已有构图、编译或输出资料。其参数以 `命令 --help` 为准。独立批量算子探针已经归档，不属于当前执行流程。
