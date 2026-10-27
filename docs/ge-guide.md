# 直接 GE 构图与执行

[首页](README.md) · [环境配置](environment-setup.md) · [板端执行](board-guide.md)

本路径使用 Python/NumPy 描述图，生成 C++ 并调用 GE 保存 AIR，不经过 PyTorch/TorchAir。先设置环境页中的公共变量。

## 1. 生成完整模型结构

构造完整 40 层 target 和 6 层 DFlash，保留 256 专家/top8、hidden 2048、词表 248320；S=8 只表示输入长度。此步无需权重或 GE，输出源码、recipe、节点连边和权重需求，状态为 `STRUCTURE_READY`，不能直接编译 OMC。

```bash
export GE_MODEL_TOOL="$REPO_DIR/targets/cpro/scripts/direct_model_air.py"
export GE_FULL_RUN="$(mktemp -d "$AI_RUN_DIR/ge-full-XXXXXXXX")"
"$MODEL_PYTHON" "$GE_MODEL_TOOL" build-full-structure \
  --sequence-length 8 --past-length 0 --expert-dispatch selected \
  --fused-ops none --output "$GE_FULL_RUN/structure"
```

Target 带显式缓存 ABI；draft 仍按完整上下文重算。`selected` 按 token 选取专家，但保留全部专家权重，尚非供应商 grouped GEMM 实现。

## 2. 准备完整真实权重

按[权重准备](weights.md)设置两套检查点目录。此命令流式转换 BF16→FP16 并使用外部权重文件；去重权重约 65.28 GiB，不适用 W4A8 板端内存预算。

```bash
export GE_FULL_PAIR_RUN="$(mktemp -d "$AI_RUN_DIR/ge-full-weights-XXXXXXXX")"
"$MODEL_PYTHON" "$GE_MODEL_TOOL" prepare --full --float-baseline \
  --sequence-length 8 --expert-dispatch selected \
  --target-weights "$QWEN35_TARGET_WEIGHTS" --draft-weights "$QWEN35_DRAFT_WEIGHTS" \
  --output "$GE_FULL_PAIR_RUN/prepared"
```

## 状态图与可选融合

先生成文档 §4.6 的接口绑定；它不安装算子，也不证明所选 DDK 支持这些算子。下面选择 Cpro 的 FP16 初态，GDN 输出 state 为 FP32。

```bash
export GE_BINDING_RUN="$(mktemp -d "$AI_RUN_DIR/ge-bindings-XXXXXXXX")"
export GE_CUSTOM_BINDINGS="$GE_BINDING_RUN/bindings.json"
"$MODEL_PYTHON" "$GE_MODEL_TOOL" bindings-from-document \
  --target "${CPRO_TARGET:-kirinx90}" --ddk-id "$CPRO_DDK_ID" \
  --gdn-initial-state-dtype float16 --output "$GE_CUSTOM_BINDINGS"
```

融合角色可选 `gdn,conv_state,rms` 的任意不重复组合；不用融合时改为 `--fused-ops none`，省略 `--bindings` 和 `--gdn-state-casts allow`。同样的选项也适用于 `build-full-structure`，但每次使用新输出目录。

```bash
export GE_STATEFUL_RUN="$(mktemp -d "$AI_RUN_DIR/ge-stateful-XXXXXXXX")"
"$MODEL_PYTHON" "$GE_MODEL_TOOL" export-stateful \
  --prepared "$GE_FULL_PAIR_RUN/prepared" --ge-root "$CANN_ROOT" \
  --sequence-length 3 --past-length 0 \
  --fused-ops gdn,conv_state,rms --bindings "$GE_CUSTOM_BINDINGS" \
  --recurrent-dtype float16 --gdn-state-casts allow \
  --output "$GE_STATEFUL_RUN/air"
```

`allow` 显式允许 GDN 末态转回 FP16 缓存并记录于 ABI；仅生成源码时将 `export-stateful` 改为 `plan-stateful`，去掉 `--ge-root`。文档融合绑定只接受完整模型尺寸，不适用于 tiny。

## 状态图：编译和输入打包

编译只接受真实 `EXPORTED` AIR；保留 AIR、外部权重和导出目录原路径。以下打包 S=3/T=0 的 prefill，完成后按[板端单步模式](board-guide.md#6-新状态图的单步执行)执行。

```bash
"$MODEL_PYTHON" "$GE_MODEL_TOOL" compile-stateful \
  --air "$GE_STATEFUL_RUN/air" --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$GE_STATEFUL_RUN/omc"
"$MODEL_PYTHON" "$GE_MODEL_TOOL" prepare-stateful-step \
  --omc "$GE_STATEFUL_RUN/omc" --token-ids 1,2,3 \
  --output "$GE_STATEFUL_RUN/step-prefill"
```

继续 decode 时，用相同 prepared 和融合选项导出 S=1/T=3 的图；`PREFILL_RESULT` 是回收的完整 prefill 结果目录，`NEXT_TOKEN_ID` 是下一步要消费的 token。

```bash
"$MODEL_PYTHON" "$GE_MODEL_TOOL" export-stateful \
  --prepared "$GE_FULL_PAIR_RUN/prepared" --ge-root "$CANN_ROOT" \
  --sequence-length 1 --past-length 3 \
  --fused-ops gdn,conv_state,rms --bindings "$GE_CUSTOM_BINDINGS" \
  --recurrent-dtype float16 --gdn-state-casts allow \
  --output "$GE_STATEFUL_RUN/decode-air"
"$MODEL_PYTHON" "$GE_MODEL_TOOL" compile-stateful \
  --air "$GE_STATEFUL_RUN/decode-air" --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$GE_STATEFUL_RUN/decode-omc"
"$MODEL_PYTHON" "$GE_MODEL_TOOL" prepare-stateful-step \
  --omc "$GE_STATEFUL_RUN/decode-omc" --token-ids "$NEXT_TOKEN_ID" \
  --previous-step "$GE_STATEFUL_RUN/step-prefill" --previous-result "$PREFILL_RESULT" \
  --output "$GE_STATEFUL_RUN/step-decode"
```

每个 T 需要匹配的静态图，打包会校验模型身份和逐层状态；该单步流程尚未形成常驻模型与 native DFlash 调度。

## 原始完整双图 AIR / OMC

需要旧 ABI v2 的 target/draft 时，复用第2节的完整 prepared 执行以下命令；此路径按前缀重算，不使用新缓存 ABI。

```bash
"$MODEL_PYTHON" "$GE_MODEL_TOOL" export \
  --prepared "$GE_FULL_PAIR_RUN/prepared" --ge-root "$CANN_ROOT" \
  --output "$GE_FULL_PAIR_RUN/air"
"$MODEL_PYTHON" "$GE_MODEL_TOOL" compile \
  --air "$GE_FULL_PAIR_RUN/air" --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$GE_FULL_PAIR_RUN/omc"
```

## 参数与接口参考

| 参数 / 角色 | 含义 |
|---|---|
| `sequence-length S` / `past-length T` | batch=1，输入 S 个连续有效 token，已有 T 个 token 状态 |
| `gdn` | `ChunkGatedDeltaRule`，含 Q/K L2 和 recurrent state；不融合整层 |
| `conv_state` | `LinearAttentionConvstateUpdate`，从 y2 尾部取 K−1 项逻辑历史 |
| `rms` | `AdnRmsNorm`；主 norm gamma=`1+w`，gated norm gamma=`w` |
| `fp16-reductions` / `fp16-row-mul` / `gdn-indexed-copy` | 默认 native；保留已有 opt-in 改写，详见 `--help` |

新状态图只接 `compile-stateful` 和 runner 的 `--stateful-step`，不能传给旧 `compile/run_hdc.py`。`CacheUpdate`、`AdnFusedInferAttention` 尚未在此 GE 路径启用；DFlash 部分接受必须提交接受前缀的状态。

`inspect-compile`、`inspect-omc`、`inspect-omc-cpu`、`compare-target-features` 可读取已有资料；当前不要求重跑旧探针。验证范围见 [validation.md](validation.md)。

## 融合算子调用探针

覆盖文档的五类自定义算子，共9例：GDN、conv-state、CacheUpdate、分页Attention各prefill/decode，以及RMS。使用完整模型的头数和通道尺寸，不需要模型权重。已有GDN编译产物时直接执行[当前更新的一条命令](current-status.md#用户现在执行什么)；下面仅用于从绑定新建整批探针：

```bash
export FUSED_PROBE_JOB="$(mktemp -d "$AI_RUN_DIR/fused-probes-XXXXXXXX")"
export FUSED_PROBE_RUN="$FUSED_PROBE_JOB/probes"
"$MODEL_PYTHON" "$GE_MODEL_TOOL" probe-fused-ops \
  --bindings "$GE_CUSTOM_BINDINGS" --ge-root "$CANN_ROOT" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --cases all --conv-history-width 4 \
  --output "$FUSED_PROBE_RUN"
cat "$FUSED_PROBE_RUN/report.txt"
```

`--cases all` 启用9例，省略则保持原GDN/conv/RMS五例。`--conv-history-width 4` 仅测试日志确认的输入宽度，输出顺序尚未验证；该候选、CacheUpdate和Attention当前只做调用及形状/有限值诊断。Attention使用独立文档端口契约，动态K/V列表各一个分页张量，可选端口保留原索引，尚未接入生产模型。

仅生成源码时省略 `--ge-root`、`--ddk-root`、`--ddk-id`；需要当前DDK名称适配时加 `--tbe-diagnosis "$TBE_DIAG_JOB/diagnosis"`。编译、加载、调用和数值诊断分别记录，不据此宣称NPU运行位置。

## tiny 回归

仅检查旧小模型链路；参考必须来自本次 prepared，不能替换成 PyTorch `demo` 的随机模型。

```bash
export GE_PAIR_RUN="$(mktemp -d "$AI_RUN_DIR/ge-tiny-XXXXXXXX")"
"$MODEL_PYTHON" "$GE_MODEL_TOOL" prepare --tiny --sequence-length 8 \
  --output "$GE_PAIR_RUN/prepared"
"$MODEL_PYTHON" "$GE_MODEL_TOOL" export \
  --prepared "$GE_PAIR_RUN/prepared" --ge-root "$CANN_ROOT" \
  --output "$GE_PAIR_RUN/air"
"$MODEL_PYTHON" "$GE_MODEL_TOOL" compile \
  --air "$GE_PAIR_RUN/air" --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$GE_PAIR_RUN/omc"
"$MODEL_PYTHON" "$GE_MODEL_TOOL" reference-tiny \
  --prepared "$GE_PAIR_RUN/prepared" --prompt-token-ids 1,2,3 --max-new-tokens 4 \
  --output "$GE_PAIR_RUN/reference.json"
```
