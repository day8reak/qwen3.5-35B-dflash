# 先核验 Target golden

真实 T64 Prefill 中，310P 的 L0–L2 QKV 和 L3 V 均通过 CPU 对照。L3 Attention core 的原输出与同次输入的独立数学参考超差。T8 下另观察到 L0 QKV 融合调用的行数相关偏差。[T64 golden 核验](#t64-golden) · [行数对照](#l0-rows)

## 1. CPU 独立重算

在保存原 NZ 源码、量化权重和 `GOLDEN_CAPTURE` 的主机执行，沿用 [Prefill 环境与变量](target-prefill-fixture.md#1-准备)，先同步 `main`。需要 Python、NumPy、PyYAML 和 CPU 可用的 PyTorch；不导入模型、不调用 NPU、不编译 AIR/OMC。无需重新运行 Prefill。

`TARGET_SOURCE` 在本步骤指生成 golden 的**原 NZ/310P 源码根目录**，必须包含 `models/modeling_qwen3_5_moe_hiai_nz.py`，不是 CPro 目录。脚本核对它与 capture 的源文件哈希；YAML 的 `quanted_pth` 相对路径也按该根目录解析。

```bash
export GOLDEN_CAPTURE="$PREFILL_FIXTURE/capture"
export PAIR_ROOT="${PAIR_ROOT:-$AI_RUN_DIR/target/tests/attention-pair}"
export GOLDEN_CHECK="$PAIR_ROOT/golden-check"

python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" verify-qlinear-golden \
  --capture "$GOLDEN_CAPTURE" \
  --source-root "$TARGET_SOURCE" --target-config "$TARGET_CONFIG" \
  --output "$GOLDEN_CHECK/cpu"

cat "$GOLDEN_CHECK/cpu/summary.txt"
```

输出目录须尚不存在。脚本按原始 INT4 权重 `[N,K]`、原 FP32 group scale 计算，先核对重新打包的权重和 scale 与 capture 完全一致，再做每组 256 元素的精确整数点积、FP64 缩放求和、最终转 FP16。另报告固定 FP32 累加结果与 FP64 参考的差异，不按 golden 选择更接近的一份。

CPU 动态量化使用明确约定：FP32 逐行 `absmax/127`、除 scale 后 ties-to-even 舍入、零行输出零。逐行缩放和舍入步骤参考[官方测试](https://github.com/Ascend/op-plugin/blob/master/test/test_custom_ops/test_npu_dynamic_quant.py)，零行策略由本工具显式定义。这不能认证当前定制 310P 内核的舍入；官方测试本身允许量化整数有 1 的差异。当前成功 capture 未保存实际 A8 激活与 token scale，小偏差不能直接判 golden 错误。

`INSPECTED` 表示计算与记录完成。各行 `AGREES/DIFFERS` 使用现有精度指标，仅描述所选投影与声明数学参考的一致性；整网 golden 保持 `UNVERIFIED`，原输入和 golden 不改写。

## 2. 有偏差时，仅重放 DynamicQuant

在**原来生成 golden 的 310P/torch_npu 环境**执行，保留原自定义算子环境配置。只对四份已保存的 hidden 调用动态量化，不加载完整模型、权重，不调用 `npu_quant_matmul`。

```bash
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" replay-dynamic-quant \
  --capture "$GOLDEN_CAPTURE" --device npu:0 \
  --output "$GOLDEN_CHECK/dynamic-quant"

python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" verify-qlinear-golden \
  --capture "$GOLDEN_CAPTURE" \
  --source-root "$TARGET_SOURCE" --target-config "$TARGET_CONFIG" \
  --quant-replay "$GOLDEN_CHECK/dynamic-quant" \
  --output "$GOLDEN_CHECK/with-dynamic-quant"

cat "$GOLDEN_CHECK/with-dynamic-quant/summary.txt"
```

第二次核验同时报告纯 CPU 参考、使用新 A8/scale 的 CPU 矩阵参考，以及两套量化中间值的差异。新调用恢复的是冻结输入的逻辑值，不是历史同次调用的 NPU format/stride，也不冒充原调用 A8。

| 结果 | 定位方向 |
| --- | --- |
| 原始权重或 scale 绑定失败 | 先核对 checkpoint、capture 与 YAML，不作精度归因 |
| CPU 量化参考有偏差，新量化加 CPU 矩阵参考一致 | 检查动态量化规则／实现差异 |
| 两套矩阵参考均与原投影明显不符 | 查原 QuantMatmul 的 group、packing、调用环境；暂不把这些 golden 当正确参考 |
| 所选投影一致 | 再推进 GDN 后续计算、FullAttention 及整网参考核验；四个投影不覆盖全部模型 |

本次四个投影的 CPU 对 310P 捕获均为 `DIFFERS`。使用重放 A8/scale 后，相对 L2 在 L0、L1、L3 的显示值完全相同，L2 从 1.194516731 变为 1.194516718。可在 `with-dynamic-quant/report.json` 的 `quantizer_comparison` 查看 A8 与 scale 的逐元素比较。

## 3. 用已有板端输出直接比 CPU

在保存了 `execution/fixture`、`recovery/native`、`omc/manifest.json` 和 `comparison/report.json` 的主机比较。如果分布在两台机器，把完整的 `golden-check/cpu/` 目录复制过去。

```bash
for group in linear full-fp32-pos; do
  python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" compare-projection-values \
    --cpu "$GOLDEN_CHECK/cpu" --group "$PAIR_ROOT/$group"
done
```

`board_vs_cpu` 直接比较同一 token、同一通道的值。L0–L2 的 ConvState 可观测最后 4 个 token，L3 按实际 block table 取出 8 个有效 token 的 V。另两行分别给出 CPU、板端在相同区域对 310P 的误差。已有第二步结果时，`--cpu "$GOLDEN_CHECK/with-dynamic-quant"` 可一次比较两套参考。

| T8 可观测区域 | 元素数 | 板端对 CPU 相对 L2 | CPU 对 310P 相对 L2 |
| --- | ---: | ---: | ---: |
| L0 QKV 末 4 token | 32768 | 0.05520% | 135.64% |
| L1 QKV 末 4 token | 32768 | 0.06185% | 144.92% |
| L2 QKV 末 4 token | 32768 | 0.07757% | 135.42% |
| L3 V 全部 8 token | 4096 | 0.05653% | 121.73% |

<a id="step4"></a>

## 4. 在原 310P 环境检查单次矩阵乘

四层 fixture 在 CPU 上执行 `quant_model_inplace(..., device='cpu')`，再用 `wrapper.to(device)` 搬到 NPU。排查时以能正常生成文本的原 310P 推理入口为基线，核对实际 QLinear 类、权重准备设备、格式转换、checkpoint 和运行库。正常吐字提供端到端证据，四层 fixture 的中间张量还需要在同一输入上逐项对齐。

默认选 L3 V，用重放所得 A8、token scale、捕获的 packed W 和 FP32 group scale 重建一次原矩阵乘。原调用省略分组或传入 None 时，再显式传 `group_sizes=[0,0,256]`。两次调用共用同一组张量。CPU 另外检验一种布局假设，把每个 INT32 按连续 8 个输出通道的有符号 INT4 解释。

```bash
git -C "$REPO_DIR" pull --ff-only
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" probe-310p-matmul \
  --capture "$GOLDEN_CAPTURE" --quant-replay "$GOLDEN_CHECK/dynamic-quant" \
  --cpu "$GOLDEN_CHECK/with-dynamic-quant" --layer 3 --device npu:0 \
  --output "$GOLDEN_CHECK/l3-matmul"
cat "$GOLDEN_CHECK/l3-matmul/summary.txt"
```

摘要逐项列出四层 capture 与单投影重放的 dtype、stride 和 NPU format。`None` 表示缺少记录。格式号相同的两个 INT32 张量仍可能装有排列方式不同的 INT4 权重。

- 先看 `original_vs_capture`，其中 `equal_bits` 表示是否逐位复现。若明显偏离，优先从正常吐字进程的同次 Prefill 取实参作对照，包括 A8、两份 scale、packed W、设备 format 和算子返回值。
- 若复现原偏差，再比较 `original_vs_cpu` 与 `explicit_group256_vs_cpu`。只改 group 参数就通过时，可把差异定位到这次调用的分组或后端选择。
- 若 `nd_hypothesis_vs_capture` 很接近，检查加载和后端对 packed W 的解释。该 CPU 实验只检验一种具体排列。全网判定仍需同输入 logits 和后续 decode 对照。

本次 L3 V 的四个参数在四层捕获和独立重放之间 dtype、stride、format 全部一致（format 均为 2）。原参数调用 V5 在 groupSize 检查报 161002，显式 group256 在内部 TransData 查找 binary 时报 561103。两次均无输出。连续输出通道 INT4 的 CPU 假设也与捕获不符（相对 L2 1.3121、余弦 −0.03176）。

<a id="step5"></a>

## 5. 在正常吐字的 310P 进程观察 L3 V

用原来成功的 Python 环境、工作目录、源码、配置和启动参数。下例的 `--` 后是传给原 `inference.py` 的参数，请沿用成功命令。原程序照常写入自身配置的输出位置。探针只在第一次 Prefill 的 L3 V 前向中旁路记录实际 A8、packed W、两份 scale、矩阵乘返回和模块返回。采集包含设备回读，本次耗时仅供调试。

```bash
git -C "$REPO_DIR" pull --ff-only
cd "$TARGET_SOURCE"
TRACE="$REPO_DIR/targets/cpro/tools/trace_target_prefill_matmul.py"
python "$TRACE" --source-root "$TARGET_SOURCE" --list-configs -- --config "$TARGET_CONFIG"
```

如果 tokenizer 报 `num_hidden_layers (4)` 与 `layer_types (40)` 冲突，错误发生在 Prefill 之前。上面的命令列出原项目根目录、`config/` 及当前 YAML 同目录的候选，打印实际 HF 配置路径、声明层数、层类型列表长度、W4 目录和权重头中出现的层号。路径按原推理工作目录解析。`w4_layer_ids` 表示这些层有权重记录，每层各投影的完整性仍需加载验证。

**40 层正常推理**使用当时正常吐字的 YAML。若上一步确认当前 `TARGET_CONFIG` 指向 40 层 checkpoint，沿用该变量即可。本模型的 `layer_types` 有 40 项，W4 应覆盖索引 0–39。

```bash
LIVE_PARENT=$(mktemp -d "$GOLDEN_CHECK/l3-live-40.XXXXXXXX")
python "$TRACE" --reference-layers 40 \
  --source-root "$TARGET_SOURCE" --capture "$GOLDEN_CAPTURE" \
  --cpu "$GOLDEN_CHECK/with-dynamic-quant" --output "$LIVE_PARENT/trace" -- \
  --config "$TARGET_CONFIG" --quant_mode enable --quant_bits 4 --max_token 2
```

**40 层模型在构造阶段显存不足时**，在同一个 310P 推理环境运行下面的 L3 单投影实验。工具向原加载器指定第 0–3 层，保留原脚本的设备选择和 W4 初始化，在第一次 Prefill 入口取出已初始化的 `v_proj`，用四层 capture 中保存的 hidden 调用一次，然后退出。完整 Prefill 和 Decode 此次均未执行。

```bash
LIVE_PARENT=$(mktemp -d "$GOLDEN_CHECK/l3-prefix.XXXXXXXX")
python "$TRACE" --reference-layers 40 --replay-l3-prefix \
  --source-root "$TARGET_SOURCE" --capture "$GOLDEN_CAPTURE" \
  --cpu "$GOLDEN_CHECK/with-dynamic-quant" --output "$LIVE_PARENT/trace" -- \
  --config "$TARGET_CONFIG" --quant_mode enable --quant_bits 4 --max_token 2
```

原脚本可能先打印 256 个 token。单投影重放只用这份 `input_ids` 确定设备，输入值取自 capture 的 L3 hidden。`live.json` 记录设备来源、模块张量的名称、dtype、shape、stride 和可读的 NPU format；若原调用报错，也会保留当时的描述。

当 `hidden_equal_bits=true` 时，`module_vs_four_layer` 对比历史 golden，`module_vs_cpu_reference` 对比同一 hidden 的已存 CPU 参考。捕获到已知签名、权重及 scale 的矩阵乘时，另有 `matmul_vs_same_call_cpu`，使用现场 A8 和 token scale 重算。仅取得模块输入输出时报告为 `PARTIAL`，可在 `report.json` 中查看已完成的比较。

**T8 前缀重放结果。** L3 V 使用相同的 hidden，现场模块输出与同次矩阵乘返回逐位一致。现场对历史 golden 的相对 L2 为 1.13231，余弦为 0.25882。现场对同次实参的 CPU group256 计算，相对 L2 为 1.40265，余弦为 0.10193。packed W 和 group scale 的逻辑值与 checkpoint 重建结果逐位一致，现场 A8 和 token scale 与独立 DynamicQuant 重放逐位一致。历史调用的 A8 内容未采集。

在保存 trace 和原权重的服务器执行离线比较，`LIVE_PARENT` 取上一步成功运行的目录。

```bash
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" inspect-live-layout \
  --trace "$LIVE_PARENT/trace" --capture "$GOLDEN_CAPTURE" \
  --cpu "$GOLDEN_CHECK/with-dynamic-quant"
```

`comparisons` 给出两种 CPU 预测。两者使用同一份现场 A8、group scale 和 token scale。`checkpoint_i4zn` 使用与现场 packed W 核对过的 checkpoint 逻辑权重，`nd_output_channel_nibbles` 将每个 INT32 从低到高解成八个连续输出通道。`live_vs_prediction` 的相对 L2 以 CPU 预测为分母，`prediction_vs_capture` 以历史 golden 为分母。`capture` 和 `live` 列出两次调用的 dtype、shape、stride、NPU format 和传参。库路径分别取自历史采集初始化和现场算子返回后的快照；实际选用的 kernel 仍需验证。

这份 T8 trace 中，现场输出对连续通道预测的相对 L2 为 1.44412，余弦为 −0.00359。两种排列都与现场及历史输出偏离。历史与现场的四个操作数的 dtype、shape、stride、NPU format（均为 2）、记录的 kwargs、软件版本及 opapi 库路径均一致。

**已声明四层的模型**可以用 `--reference-layers 4` 运行自然 Prefill 观察，省略 `--replay-l3-prefix`，另取新的输出目录。对于已声明四层的 Qwen 配置，工具仅在当前进程内取过长层类型列表的前四项。报告记录配置适配和实际观测的模型层数。该结果只覆盖四层路径。

`matmul_vs_same_call_cpu` 用本次观测的 A8 和 token scale 重算，先逐位核对本次权重与 group scale。`hidden_equal_bits` 同时覆盖形状、dtype 和内容，只有它为 true 才比较新旧模块输出以及新 A8 与重放 A8。原入口若把 prompt 重复 32 次，很可能与 T8 捕获长度不同，此时同次矩阵乘对照仍有效。采集细节在输出目录的 `live.json`，比较在 `report.json`。

已有独立重放报告也可直接对比进程信息。四层 capture 的快照早于实际投影，库映射清单可提供线索，单凭清单不能确认 V5 由哪个库实现。

```bash
python - "$GOLDEN_CHECK/l3-matmul/report.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    r = json.load(f)
for key in ("capture_runtime", "runtime_before", "runtime_after"):
    v = r.get(key) or {}
    print(key, v.get("python"), v.get("torch_version"), v.get("torch_npu_version"))
    print("ASCEND_CUSTOM_OPP_PATH =", v.get("environment", {}).get("ASCEND_CUSTOM_OPP_PATH"))
    print(*(v.get("loaded_libraries") or []), sep="\n")
PY
```

<a id="step6"></a>

## 6. 在生成 golden 的四层 Prefill 中复测 L3 V

在原 310P 环境运行。工具校验旧 fixture 的 token 文件、历史 L3 hidden 及前缀 trace，沿用旧 token 和中间量采集选项，从 CPU 构造并量化前四层，按原顺序执行一次 Prefill。第一次 L3 V matmul 的实参与返回值保存在新目录。Prefill 返回后，在同一进程把原张量对象及参数再交给同一算子入口一次。如果新 Prefill 到达 L3 的 hidden 已变，工具还会对当前初始化的 V 投影单独送入历史 hidden。

```bash
CHECK_PARENT=$(mktemp -d "$GOLDEN_CHECK/l3-repeat.XXXXXXXX")
CHECK="$CHECK_PARENT/check"
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" check-prefill-matmul \
  --fixture "$PREFILL_FIXTURE" --trace "$LIVE_PARENT/trace" \
  --source-root "$TARGET_SOURCE" --target-config "$TARGET_CONFIG" \
  --device npu:0 --output "$CHECK"
```

先看 `observation.operands_equal_before_repeat` 和 `observation.repeat_vs_first.equal_bits`。四个实参逐位一致而输出不同，表示这两次调用的结果有波动。输出一致时，看 `hidden_equal_to_historical`、`operands_equal_to_prefix`、`signature_equal_to_prefix` 以及 `first_vs_historical_golden` 和 `first_vs_prefix`。若 hidden 改变，用 `frozen_hidden_control` 看固定输入对照。这里新采的 A8 只代表本次 Prefill。完整报告位于 `$CHECK_PARENT/check/report.json`，旧 fixture 和前缀 trace 仅读取。

**T8 实测。** 两轮都复现了同一现象。每轮在首次调用和 Prefill 结束后复跑前，四个操作数均逐位相同。两轮各自的输出相对 L2 分别为 1.16503 和 1.07370，余弦分别为 0.22432 和 0.29921。两轮自然 Prefill 的 L3 hidden 均与历史捕获不同。固定历史 hidden 后，四个实参均与前缀实验逐位相同，输出相对前缀的 L2 仍分别为 0.86869 和 0.86347。此 L3 V golden 暂停用于判定 CPro 精度。

在保存两份 fixture 的主机上，复用同一终端的 `CHECK` 定位最早发生变化的捕获点。如果换了终端，或上一步在子 shell 中运行，把报告中 `report` 的完整路径赋给 `CHECK`。下面两条命令都接受 `report.json` 路径或其所在目录。

```bash
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" inspect-prefill-drift \
  --fixture "$PREFILL_FIXTURE" --check "$CHECK"
```

`wrapper_input_differences` 列出发生变化的具名输入。`first_changed_boundary` 按捕获顺序指向第一次输入或输出字节变化，`first_output_outside_gate` 指向第一个有变化且超出既有精度标准的输出所在边界。对应的 `observations` 给出差异张量、指标、属性和已记录权重的逐位对比。两个字段可以指向同一边界。捕获点之间仍可能包含未观测的计算。

在原 310P 自定义算子环境中，使用同一 `CHECK` 单独重放保存的 L3 矩阵乘。只从 `first_call.pt` 恢复一次设备张量，按原参数连续调用三次，每次同步、回读并复核操作数。Prefill 和板端 OMC 均无需重跑。

```bash
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" replay-saved-matmul \
  --check "$CHECK" --device npu:0
```

`calls[1:]` 中的 `vs_previous` 比较相邻两次返回，`vs_prefill` 对比原首次调用，`vs_delayed` 对比原 Prefill 结束后的复跑。格式、stride 和还原的逻辑字节可在同一份输出中核对。独立进程使用重新分配的设备张量；三次结果一致时，原 Prefill 进程的差异仍须单独解释。若算子调度报错，报告会保留错误码和已完成的调用数。

<a id="step7"></a>

## 7. 定位 T8 Prefill 的首个投影偏差

两次四层 Prefill 的 wrapper 输入一致。按捕获顺序检查到第 2 个边界时，L0 `linear_attn.in_proj_qkv` 已经超差。该模块的 hidden、属性、packed W 和 group scale 一致，输出相对 L2 为 1.08216，余弦为 0.39999。原始两次调用均缺少该层 A8 和 per-token scale 的值。

在 `qwen3_5_w4` 环境独立重放 L3 V，四个实参的还原字节、dtype、shape、stride 和格式均匹配，每次同步后实参保持一致。第 2 对第 1 次、第 3 对第 2 次的输出相对 L2 分别为 1.04879、1.07200，余弦分别为 0.45271、0.41593。该结果只覆盖这份 T8 调用，具体执行内核及原因仍待确认。

在同一 310P 环境分两步复测 L0。第一步用同一份捕获 hidden 连续量化三次，分别检查 A8 和 per-token scale。第二步固定已有量化重放保存的 A8、per-token scale 和捕获的两份权重参数，按原调用签名连续做三次矩阵乘。两步都逐次同步并检查输入。输出目录由本次命令新建。

```bash
git -C "$REPO_DIR" pull --ff-only
L0_RUN=$(mktemp -d "${GOLDEN_CHECK:?}/l0-stages.XXXXXXXX")
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" probe-310p-matmul \
  --capture "$PREFILL_FIXTURE/capture" --quant-replay "$GOLDEN_CHECK/dynamic-quant" \
  --cpu "$GOLDEN_CHECK/with-dynamic-quant" --layer 0 --device npu:0 \
  --repeat-original --dequant-control --output "$L0_RUN/probe"
cat "$L0_RUN/probe/summary.txt"
```

用户在 `qwen3_5_w4` 回传的 L0 QKV 结果覆盖全部 8 个 token。

| 310P 路径 | 对照结果 | 重复执行 |
| --- | --- | --- |
| DynamicQuant | 三次 A8 和 per-token scale 均与已保存的量化重放逐位一致 | 三次 hidden 均保持原值 |
| 显式反量化 checkpoint W4，FP32 矩阵乘，乘 token scale 后转 FP16 | 两次均通过 CPU 五项指标，相对 L2 0.01737%，最大绝对误差 0.015625 | 两次输出逐位一致 |
| 原 A8W4 融合矩阵乘 | 三次对 CPU 的相对 L2 分别为 1.25139、1.03471、1.03471（比值） | 第 2 次对第 1 次最大绝对差 24.03906，第 3 次与第 2 次逐位一致 |

要检验原调用省略分组参数是否影响结果，在同一 310P 环境复用上一步的 capture、量化重放和 CPU 参考。只改变 `group_sizes` 参数，交错运行原签名和显式 `[0,0,256]` 各三次。每次同步后比较 CPU 数值、四个操作数的逻辑字节及同签名的上次输出。

```bash
L0_GROUP=$(mktemp -d "${GOLDEN_CHECK:?}/l0-group.XXXXXXXX")
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" probe-310p-matmul \
  --capture "$PREFILL_FIXTURE/capture" --quant-replay "$GOLDEN_CHECK/dynamic-quant" \
  --cpu "$GOLDEN_CHECK/with-dynamic-quant" --layer 0 --device npu:0 \
  --compare-group256 --output "$L0_GROUP/probe"
cat "$L0_GROUP/probe/summary.txt"
```

摘要末尾分别给出两个签名的 `group check`。通过要求同签名三次均对齐 CPU 且输出逐位稳定。调度报错会保留原始错误和已完成次数。该对照只判定固定输入的 L0 QKV，整网 golden 仍待验证。

<a id="l0-rows"></a>

### 对比融合矩阵乘的 M=8、64、256

L0 两种签名各完成三次，都未通过。原调用对 CPU 的相对 L2 为 1.018–1.184，显式 group256 为 1.012–1.018。后者两次与上一次同签名输出比较的相对 L2 分别为 0.422 和 0.408。

在 310P 的 `qwen3_5_w4` 环境，将捕获的 8 行 A8 和对应 token scale 按原顺序重复到指定行数，沿用同一 packed W、group scale 和原调用签名。每个进程只测一个 M，运行三次。CPU 期望值使用同一批 8 行投影逐块重复。这是独立 QKV 投影对照。

```bash
L0_ROWS=$(mktemp -d "${GOLDEN_CHECK:?}/l0-rows.XXXXXXXX")
for m in 8 64 256; do
  profiler=()
  if command -v msprof >/dev/null; then profiler=(msprof "--output=$L0_ROWS/profile-$m"); fi
  "${profiler[@]}" python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" probe-310p-matmul \
    --capture "$PREFILL_FIXTURE/capture" --quant-replay "$GOLDEN_CHECK/dynamic-quant" \
    --cpu "$GOLDEN_CHECK/with-dynamic-quant" --layer 0 --device npu:0 \
    --probe-rows "$m" --output "$L0_ROWS/m$m"
done
grep 'original check:' "$L0_ROWS"/m*/summary.txt
```

`PASSED` 要求每次的整体及每个 8 行块均对齐 CPU，且三次输出逐位一致。若只有较长 M 通过，记录哪些行数能复现偏差。有 profiling 时对照各行数的 `op_summary_*.csv` 和 `task_time_*.csv` 里实际记录的任务名、类型和输入形状（如有）。仅凭张量格式号和已加载的库还无法确定具体内核。工具的采集和导出字段见[华为 msprof 文档](https://www.hiascend.com/document/detail/en/mindstudio/700/TITools/Profiling/atlasprofiling_16_0005.html)。

L0 QKV 实测中，M=8 的首次输出对 CPU 的相对 L2 为 1.01228，余弦为 0.22522，三次调用综合判定失败。M=64 和 M=256 均通过，首次相对 L2 均为 0.00074388，余弦均为 0.99999975。

三个 M 各有三条 AI_CORE 任务记录。`op_summary` 的算子名均为 `aclnnQuantMatmulV4_QuantBatchMatmulV4444_QuantBatchMatmulV4444`，`task_time` 的内核名均为 `QuantBatchMatmulV4444_b5528a498525995346b260ca8f9eb8c4_21`。记录的 dtype 和 ND 格式一致；变化的形状为 A8 的 `[M,2048]`、token scale 的 `[M,1]` 和输出的 `[M,8192]`。回传结果未给出 Block Dim 或 Tiling Key，具体分块与尾行处理仍待核对。以上结果只覆盖 L0 独立投影。

直接在同一台 310P 主机读取这次的 profiling：

```bash
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" inspect-row-profiles --root "$L0_ROWS"
```

按 M 和来源 CSV 对照设备侧记录的算子名、内核名、Block Dim 及 tiling key 等可用列。`NO_EXPORTED_CSV` 时查看 `raw_profile_dirs`；已有原始 `PROF_*` 目录可用 `msprof --export=on --output=<原始目录>` 导出后再读。[msprof 导出说明](https://www.hiascend.com/doc_center/source/en/mindstudio/700/TITools/Profiling/atlasprofiling_16_0055.html)

继续验证另外三个已捕获的投影时，共用原 T8 的逐层输入和 CPU 参考，只测试已在 L0 通过的 M=64：

```bash
for layer in 1 2 3; do
  python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" probe-310p-matmul \
    --capture "$PREFILL_FIXTURE/capture" --quant-replay "$GOLDEN_CHECK/dynamic-quant" \
    --cpu "$GOLDEN_CHECK/with-dynamic-quant" --layer "$layer" --device npu:0 \
    --probe-rows 64 --output "$L0_ROWS/layer-$layer-m64"
done
grep -E 'original_vs_cpu:|original check:' "$L0_ROWS"/layer-*-m64/summary.txt
```

在上述 M=64 对照中，L0、L1、L2 的 QKV 和 L3 的 V 首次对 CPU 的相对 L2 依次为 0.07439%、0.07531%、0.07399%、0.09220%。每个投影的三次调用均通过，且各自三次输出逐位一致。

<a id="t64-golden"></a>

### 核验真实四层 T64 Prefill 的投影

在 310P 的 `qwen3_5_w4` 环境，按[采集命令](target-attention-pair.md#t64-gdr)将 YAML 的 prompt 重复 8 次，执行一次四层 Prefill。已有该次采集时直接复用。确认 `sequence_length=64`、`boundary_capture.status=CAPTURED`。若总状态为 `PARTIAL` 且仅 Full Attention 模块参考重放失败，仍可核验原前向捕获的四个投影。

```bash
PROBE="$REPO_DIR/targets/cpro/tools/target_layer_probe.py"
T64_FIXTURE="$AI_RUN_DIR/target/tests/prefill-fixture-t64"
T64_CHECK="$GOLDEN_CHECK/t64"

python "$PROBE" replay-dynamic-quant \
  --capture "$T64_FIXTURE/capture" --device npu:0 \
  --output "$T64_CHECK/dynamic-quant"
python "$PROBE" verify-qlinear-golden \
  --capture "$T64_FIXTURE/capture" \
  --source-root "$TARGET_SOURCE" --target-config "$TARGET_CONFIG" \
  --quant-replay "$T64_CHECK/dynamic-quant" --output "$T64_CHECK/with-dynamic-quant"
cat "$T64_CHECK/with-dynamic-quant/summary.csv"
```

本次两种参考的八行均为 `AGREES`。按纯 CPU 参考的报告口径，L0、L1、L2 的 QKV 和 L3 的 V 相对 L2 依次为 0.07437%、0.09832%、0.11289%、0.22203%。这些数值覆盖每个投影在该次前向的全部 64 行。

L3 的 `attention_audit=FAILED` 表示原 fused Attention core 输出超出同次 Q/K/V、mask、页表与有效长度计算的 CPU 参考门槛。`attention_module_reference=COMPUTED` 表示用独立数学 core 重放整个注意力模块已成功，重放 K/V 与原捕获逐位一致。可直接读取这份 fixture，查看失败指标、最大误差的逻辑坐标（batch、head、query、channel）、每 16 个 query 的误差及原模块 hidden 对重放的偏差。

```bash
python "$PROBE" inspect-prefill-audit --fixture "$T64_FIXTURE"
```

对照板端 Full Attention 时选择 `--full-attention-reference math-replay`，用重放输出作主参考，同时保留原模块输出作对照。整网 logits 和 GDN 状态的精度仍需单独验证。[两类 Attention 的验证流程](target-attention-pair.md#t64-full)
