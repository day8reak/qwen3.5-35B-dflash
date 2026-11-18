# 先核验 Target golden

本次 T8 Prefill，板端可观测的四个投影与独立 CPU group256 参考均通过，原四层捕获在这些位置偏离参考。独立重放 L3 矩阵乘停在 V5 参数校验和 TransData，当前从[步骤 5](#step5)记录正常吐字进程的同次实参和返回值。

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
