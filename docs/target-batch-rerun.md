# Target：整层 GatedDeltaNet 测试与 Attention 验证

**当前整层 GDN FP16 六图：18 个输出全部 FAILED，全部重复稳定。已确认 Prefill ConvState 等于全零旧输入，Decode 与旧输入不同；六图 linear_update_tensor 都是全零。下一步执行[第 7.6 节](#76-整层-gdn-数值失败后的只读检查)的布局及控制来源检查，复用已有 dump。** Attention 两图已对独立数学参考五项通过：Prefill L2=0.11175%，Decode L2=0.11438%，各自 10 次输出一致；原 Prefill captured golden 错误已确认。Attention、RoPE、Prefill Cache 不重复编译。完整结论见[精度状态](target-accuracy-status.md)。

采集覆盖四层，供 Attention 和整层 GDN 共用。第 2 节为已完成的 Attention 准备流程，第 7 节切换为当前 GDN 测试：

| 模块 | 图数 | 每张图输出 |
| --- | ---: | --- |
| L3 AdnFusedInferAttention（本轮已通过） | 2 | attention_nz |
| L0–L2 整层 GatedDeltaNet（当前） | 6 | hidden、new_conv_state、recurrent_state（FP16） |

GDN 包含五个 QLinear、ConvState、卷积、SiLU、g/beta、Q/K 重复、ChunkGatedDeltaRule、gated norm 和 out_proj，层内真实串联。它不是把 core/state 再分成小图，也不包含 Decoder 的外部残差或 MoE。QLinear 沿用已验证的 FP16 scale 适配；卷积沿用显式单高度 Conv2d。

输入和三个 golden 从同一 CAPTURED layer/call/phase 的已有边界组合，检查四个输入投影的 hidden 一致。其中 linear_update_tensor 是捕获工具按原 exporter 合成的 INT32[128] 全零占位，并非参考执行时观测的控制量，当前需核对其运行契约。使用显式选项 --gdn-state-dtype float16：初态、GDN 末态和层内 writeback 为 FP16；g 沿用原计算，在 GE 算子入口数值 Cast 为 FP16。原始 raw-state FP32 golden 字节保留，实际 FP16 输出提升到比较精度后计算误差，五项门槛保持。CSV 的 actual_dtype/golden_dtype 分别记录两端类型，state 的 dtype_preserved=false 是已选择的接口差异，不单独判数值失败；输出仍须满足本次 FP16 ABI 和重复稳定性。

Attention 沿用捕获 Q/K/V、mask、block_table 和长度参数，新增 Q 轴、Q 长度常量字节与 allQLen 传递检查。当前范围为 B=1 静态图。Q 为 `[B, heads×D/16, S, 16]`，序列轴是 2；旧捕获 Prefill S256、Decode S1。源码全图 S64 对应 `[1,256,64,16]`，旧捕获小图对应 `[1,256,256,16]`；两者均没有把 Prefill Q 变成 1。新采集的 S 以实际 manifest 为准，不为了复现旧长度补齐或截断输入。`allQLen` 单独记录，不能强制等于当前 Q 长度；它在用户提供的内核中控制 KV 块循环。实际 tiling 和完整生产图仍需编译产物核对。

每张图每次都使用捕获输入；整层 GDN 包含层内误差传递，仍不包含跨图连续 Prefill→Decode 状态反馈。48 个 GDN 独立边界已回传 43 PASS / 5 FAIL，整层 GDN 当前 18 个输出全部失败。L0–L2 的独立函数测试可以继续复用捕获；L3 Prefill Attention 下游的整图 golden 需要参考路径修复后重建。

## 1. 初始化路径

各阶段使用各自已跑通的环境。用户确认：成功采集 golden 的 **310P 参考推理环境与 qwen3_5_cpro 不同**；参考采集必须恢复那套 Python、CANN 和自定义算子初始化方式。qwen3_5_cpro 用于此前已成功的 CPro AIR 导出，不能因导出成功就用于生成参考。编译主机使用已选 DDK；上板主机沿用 runtime.json、HDC 和 runner。

```bash
export REPO_DIR="$(git rev-parse --show-toplevel)"
git -C "$REPO_DIR" pull --ff-only origin main
export AI_RUN_DIR="${AI_RUN_DIR:-${REPO_DIR}-run}"
export GOLDEN_CAPTURE="$AI_RUN_DIR/target/tests/golden-capture"
export BATCH_ROOT="$AI_RUN_DIR/target/tests/attention"
export BATCH="$BATCH_ROOT/air"
export OMC_BATCH="$BATCH_ROOT/omc"
export COMPARISON="$BATCH_ROOT/comparison"
export RUNTIME_CONFIG="${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}"
export EXECUTION="$BATCH_ROOT/execution"
export RECOVERY="$BATCH_ROOT/recovery"
export PYTHONDONTWRITEBYTECODE=1
```

各主机设置各自实际的 REPO_DIR／AI_RUN_DIR，并保持 BATCH_ROOT 内 air／omc 的相对位置。已有本组 AIR／OMC 时，从缺失的下一阶段继续。新建命令要求空输出槽位。固定目录复测前先确认旧请求结束、已回收，再清理该测试的临时产物，不覆盖活动任务。golden-capture 是后续测试的共同输入，不能随单个 Attention 测试目录清理。

### 1.1 重新采集 golden（310P 主机）

在原先已跑通的 PyTorch／torch_npu 环境中执行。TARGET_SOURCE 和 TARGET_CONFIG 分别设置为原私有项目根目录和原权重 YAML 的绝对路径，保留原配置中的权重、prompt 与量化参数。使用已验证的 inference_target_probe.py，实际构造前四层并执行原 inference.py；采集 Prefill 和第一次 Decode。此步执行参考模型，但不编译 CPro 的 QLinear、RoPE、Cache 等独立测试图。

在参考主机的新终端按原成功方式初始化 CANN／自定义算子环境，并激活原参考 Conda 环境。REFERENCE_CONDA_ENV 填原成功环境的实际名称；不能填当前报 V5 groupSize 错误的 qwen3_5_cpro。只切 Conda 不能保证清除上一套环境留下的库搜索路径，因此仍需沿用原初始化方式。

```bash
: "${REFERENCE_CONDA_ENV:?请设置此前成功采集 golden 的 310P Conda 环境名}"
conda activate "$REFERENCE_CONDA_ENV"
python - <<'PY_ENV'
import sys, torch, torch_npu
print('python:', sys.executable)
print('torch:', torch.__version__)
print('torch_npu:', getattr(torch_npu, '__version__', 'unknown'), torch_npu.__file__)
PY_ENV
```

```bash
: "${TARGET_SOURCE:?请设置原 Target 源码根目录的绝对路径}"
: "${TARGET_CONFIG:?请设置原 Target 权重 YAML 的绝对路径}"
(
set -e
test -f "$TARGET_SOURCE/inference.py"
test -f "$TARGET_CONFIG"
test ! -e "$GOLDEN_CAPTURE"
cd "$TARGET_SOURCE"
python "$REPO_DIR/targets/cpro/tools/inference_target_probe.py" \
  --source-root "$TARGET_SOURCE" \
  --output "$GOLDEN_CAPTURE" \
  --model-layers 4 --layers 0,1,2,3 --calls 2 -- \
  --config "$TARGET_CONFIG" --quant_mode enable --quant_bits 4 --max_token 2
)

python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" list \
  --capture "$GOLDEN_CAPTURE"
```

成功条件是 manifest.json 的 status=CAPTURED，且包含 L3 call0/prefill 与 call1/decode 的 fused_attention；下一步会检查。采集后可跳过本节复用同一 capture；缺失时必须重新采集，不能从 CSV 或指标表反推 golden。原 inference.py 的其他已验证参数继续放在 `--` 后，不切换 pseudo_quant 或额外量化策略。

### 1.2 四层配置校验失败后的重试

若报 `num_hidden_layers (4) must be equal to the number of layer_types (40)`，且 list 显示 FAILED、regions=[]，失败发生在 AutoTokenizer → AutoConfig 的配置构造阶段，尚未执行模型计算。Transformers 的[配置校验](https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py)要求两者一致；pkg_resources 警告不是这次退出原因。

当前采集入口已做进程内兼容：只针对 YAML 的 model_file 指向的 checkpoint，且文本配置已经声明本次请求的四层时，保留 layer_types 前四项及原顺序，再交给正常 HF 校验。合法的 40/40 配置保持原样；模型工厂仍只构造四层，并同步构造后配置的层数与层类型列表。原 config.json、权重和 tokenizer 文件均不写入，不需要降级 Transformers 或关闭校验。manifest.json 的 config_compatibility 记录实际适配。

先更新代码，清理本次配置错误产生的空失败目录，然后重新执行第 1.1 节。下面只允许删除本次 run 内、因该错误失败且无 regions 的 manifest／qlinear-audit 两类文件；已有采集、活动状态或其他文件会停止清理。仍使用原 GOLDEN_CAPTURE 路径。

```bash
git -C "$REPO_DIR" pull --ff-only origin main
python - "$GOLDEN_CAPTURE" <<'PY_RETRY'
import json, os, sys
from pathlib import Path
p = Path(sys.argv[1])
run = Path(os.environ['AI_RUN_DIR']).resolve()
if p.is_symlink() or run not in p.resolve().parents:
    raise SystemExit('拒绝清理 run 目录外或符号链接路径')
m = json.loads((p/'manifest.json').read_text())
error = '\n'.join(m.get('errors', []))
if (m.get('artifact_kind') != 'qwen35-target-cpro-boundary-probe-v1'
        or m.get('status') != 'FAILED' or m.get('regions') != []
        or 'num_hidden_layers' not in error or 'layer_types' not in error):
    raise SystemExit('只清理本次层数配置错误产生的空 FAILED capture')
files = list(p.iterdir())
if any(f.name not in {'manifest.json', 'qlinear-audit.json'}
       or f.is_symlink() or not f.is_file() for f in files):
    raise SystemExit('目录包含其他产物，停止清理')
for f in files:
    f.unlink()
p.rmdir()
print('已清理空失败 capture；重新执行第 1.1 节采集命令')
PY_RETRY
```

在本地 Transformers 5.13.0 复现相同异常并验证修复后 AutoConfig／AutoTokenizer 可加载、token ID 不变、原文件字节不变，两个四层构造入口同步了列表。本地检查不代表 310P 采集已经成功；仍以重跑的 CAPTURED 结果为准。

### 1.3 首个 QLinear 报 V5 groupSize 错误

2026-11-13 回传已进入四层模型，在 L0 in_proj_qkv 首次调用失败：K=2048、scale 有 8 组，对应 group size=256；original_kwargs 和 effective_kwargs 均未传 group_sizes，实际调用 aclnnQuantMatmulV5 报 161002。公开 [op-plugin 7.1.0 实现](https://github.com/Ascend/op-plugin/blob/7.1.0/op_plugin/ops/opapi/QuantMatmulKernelNpuOpApi.cpp)的未传分组分支返回 0，并将该值交给 V5；与本次错误一致，安装包的完整行为仍以本机日志为准。

用户已确认旧 golden 来自另一个环境，当前先恢复原参考环境，保持原 npu_quant_matmul 参数。不要把 CPro QuantBatchMatmul 已验证的 FP16 scale 适配应用到 310P 参考，也不通过添加 --qlinear-group-size 256 声称恢复原 golden。显式 group256 是另一个已有诊断选项，其基线等价性未验证；本次暂不启用。

本次 FAILED capture 含真实 QLinear 输入，不能使用第 1.2 节的空目录清理。先移到固定 qlinear-failure 槽位保留，再用原 GOLDEN_CAPTURE 路径重采集；不覆盖已有槽位：

```bash
git -C "$REPO_DIR" pull --ff-only origin main
python - "$GOLDEN_CAPTURE" <<'PY_QMATMUL_RETRY'
import json, os, sys
from pathlib import Path
p = Path(sys.argv[1])
run = Path(os.environ['AI_RUN_DIR']).resolve()
dest = run/'target/tests/qlinear-failure'
if p.is_symlink() or run not in p.resolve().parents:
    raise SystemExit('失败 capture 必须位于本次 run 内且不是符号链接')
m = json.loads((p/'manifest.json').read_text())
audit = json.loads((p/'qlinear-audit.json').read_text())
failed = audit.get('failed_call') or {}
error = failed.get('error', '')
if (m.get('artifact_kind') != 'qwen35-target-cpro-boundary-probe-v1'
        or m.get('status') != 'FAILED' or failed.get('status') != 'FAILED'
        or 'aclnnQuantMatmulV5' not in error or 'groupSize' not in error
        or not (p/'qlinear-failure.pt').is_file()):
    raise SystemExit('目录不是本次带重放输入的 V5 groupSize 失败采集')
if dest.exists() or dest.is_symlink():
    raise SystemExit('qlinear-failure 槽位已存在，不覆盖；请先核对其内容')
dest.parent.mkdir(parents=True, exist_ok=True)
p.rename(dest)
print('失败输入保留在:', dest)
print('恢复原 310P 参考环境后，重新执行第 1.1 节')
PY_QMATMUL_RETRY
```

用户随后回传换回原参考环境后采集成功，本次无需显式 group256 或继续重放失败算子。第 2 节仍会检查所选新 capture 的 CAPTURED 状态与两次 Attention 边界；原始 manifest 未回传本地，成功状态来自用户报告。这证明原参考环境的采集流程恢复，尚未完成新 Attention AIR／OMC 的数值验证。

## 2. 导出 AIR

切回此前已成功导出 AIR 的环境（qwen3_5_cpro），读取第 1.1 节生成的 capture，不在这个环境重新采集。TARGET_SOURCE 为原私有源码根目录，TARGET_CONFIG 为此前成功的原权重 YAML。先核对新 capture 包含两次 Attention 边界，缺项即停止。

```bash
: "${TARGET_SOURCE:?请设置原 Target 源码根目录}"
: "${TARGET_CONFIG:?请设置原 Target 权重 YAML 路径}"
export CPRO_EXPORT_SCRIPT="${CPRO_EXPORT_SCRIPT:-$TARGET_SOURCE/export_glm_model_qwen3_5_moe.py}"
(
set -e
test -f "$TARGET_CONFIG"
test -f "$CPRO_EXPORT_SCRIPT"
test ! -e "$BATCH"
PYTHONPATH="$REPO_DIR/model${PYTHONPATH:+:$PYTHONPATH}" python - "$GOLDEN_CAPTURE/manifest.json" <<'PY_CHECK'
import json, sys
m = json.load(open(sys.argv[1]))
attention = [r for r in m['regions'] if r['point'] == 'fused_attention']
if (m.get('status') != 'CAPTURED' or len(attention) != 2
        or {(r['layer'], r['call'], r['phase']) for r in attention} != {(3,0,'prefill'), (3,1,'decode')}):
    raise SystemExit('需要 L3 Prefill/call0、Decode/call1 的 Attention 捕获。')
print('CAPTURE: 2 attention graphs; 2 outputs')
PY_CHECK
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" export \
  --capture "$GOLDEN_CAPTURE" \
  --source-root "$TARGET_SOURCE" --target-config "$TARGET_CONFIG" \
  --converter-script "$CPRO_EXPORT_SCRIPT" \
  --points fused_attention \
  --phases prefill decode --output "$BATCH"
)
```

应为 EXPORTED 2/2。失败看 manifest.json 的 error 和 `$BATCH/<case>.log`。每份 export.json 保留 `source_contract.status=BOUND_TO_SOURCE`、源码表达式、输入与输出类型；不能修改描述符来掩盖 dtype 差异。跨主机复制整个 BATCH，包含清单、schema、输入、golden，不只复制 export.air。

## 3. 编译 OMC

沿用此前成功的 DDK（本轮 ID 为 100.520.020.010）和已完成的 TBE 诊断。GDN 曾遇到 TE worker 初始化失败，继续使用现有兼容配置。

```bash
: "${CPRO_DDK_ROOT:?请设置该主机此前成功的 DDK 根目录}"
export CPRO_DDK_ID="${CPRO_DDK_ID:-100.520.020.010}"
export TBE_DIAGNOSIS="${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}"
(
set -e
test -f "$TBE_DIAGNOSIS/report.json"
test ! -e "$OMC_BATCH"
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" compile \
  --batch "$BATCH" --output "$OMC_BATCH" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --compiler-profile isolated-kirinx90 \
  --tbe-diagnosis "$TBE_DIAGNOSIS" --use-origin-format
)
```

摘要在 `$OMC_BATCH/compile.json`，日志在 `models/<case>/omc/compile.log`。Attention 应为 COMPILED 2/2，整层 GDN 应为 COMPILED 6/6；部分失败时，下一步自动执行已 COMPILED 的项，缺项保留。跨主机复制整个 BATCH_ROOT，保持 air／omc 相对位置。

## 4. 上板与回收

沿用已成功的 runtime.json 和 runner，无需重建运行器。每图预热 3 次、采样 10 次，每次恢复捕获输入。

```bash
if test -f "${QWEN35_ENV_CONFIG:-$REPO_DIR/../qwen35-cpro.env}"; then
  source "$REPO_DIR/targets/cpro/scripts/env.sh"
fi
export HDC="${HDC:-hdc}"
: "${BIP:?沿用原 HDC 服务端 IP:端口}" "${SN:?沿用原设备序列号}"
: "${CPRO_DDK_ROOT:?上板主机实际 DDK 根目录}"
test -f "$RUNTIME_CONFIG"
python "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --compiled "$OMC_BATCH" --config "$RUNTIME_CONFIG" \
  --ddk-root "$CPRO_DDK_ROOT" --output "$EXECUTION" --apply
```

入口名虽然包含 draft，也支持 Target。启动后只回收同一个请求：

```bash
python "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --recover "$EXECUTION" --output "$RECOVERY"
```

返回 PENDING 时稍后重复 recover，不重复 apply。数值失败可能让回收返回非零；有 native 输出后仍可继续比较。

## 5. 五项指标比较

```bash
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" compare \
  --fixture "$EXECUTION/fixture" --native "$RECOVERY/native" \
  --output "$COMPARISON"
```

回传 `summary.csv` 和 `report.json`。数值通过要求 cosine >0.99、relative_l2 <0.05、KL <0.005、RMSE <1、MeanRelativeError <1，所有重复调用通过且稳定。CSV 同时记录 failed_metrics 和 dtype_preserved。raw state 即使数值通过，也须核实最终状态接口，不能忽略类型变化。

Attention 批次 summary.csv 为 2 行，整层 GDN 批次为 18 行。任何输出或任意一次重复失败，整组不能通过。Attention 原 captured-golden 比较仍会复现 Prefill 失败；它已由第 5.1 节独立参考与范围约束确认属于参考错误，当前模块结果明确采用独立参考，不静默更换 fixture golden。

Attention 的 `export.json.attention_contract` 会记录 Q shape、actual_seq_lengths_q、actual_seq_lengths_kv、all_seq_lengths_q 及 AIR 传递链。`query_length_constant` 验证实际 INT64 常量内容，不只看 `_readable_value`。本轮保持算子实现不变，先核对调用与导出。

### 5.1 Attention 输出检查

对现有产物执行离线检查。工具记录源 AIR 的长度常量、各 head／16 行块／逐行统计与零行位置。本次增加 `--reference`，按源 AIR 端口绑定 Q/K/V、mask、页表和 allQLen，独立计算 `softmax(scale × QKᵀ + 原 additive mask) × V`，FP64 算术后转 FP16，分别比较 actual 与 captured_golden；五项标准保持不变。输出坐标从 0 开始，按已核对的源码 packed 布局解释。

参考按页表重建有效 KV、按 GQA 关系选 KV head，用 allQLen 控制 KV 长度；`actual_seq_lengths_kv=2048` 保持 mask 行跨度。不会自动添加 causal mask、裁掉 query 行或覆盖原 golden。若整行 mask 都为 -inf，报告 `UNDEFINED_ALL_MASKED_ROWS`，不擅自填零或给整输出通过结论。还会检查两侧输出是否超出对应 V 通道范围；很小的浮点越界只作诊断，不能代替五项验收。原始 capture 无需带到此主机，但 BATCH_ROOT/air 中的源 AIR、air-schema.pb、golden 及捕获输入仍需保留。仅需 NumPy 与原检查工具依赖，不加载 torch_npu。

先同步仓库 main。以下重建已结束的 inspection 报告目录；其他测试产物继续复用，不执行导出、编译或上板：

```bash
rm -rf -- "${BATCH_ROOT:?}/inspection"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" inspect-attention --reference \
  --batch "$OMC_BATCH" \
  --fixture "$EXECUTION/fixture" --native "$RECOVERY/native" \
  --output "$BATCH_ROOT/inspection"

cat "$BATCH_ROOT/inspection/reference-summary.txt"
```

此检查已完成，下面的结论来自用户回传 `inspection/reference-summary.txt`。复核时重点看 `active_V_bounds`、`full_value_cache_abs_max`、mask 第 16–31 行统计，以及 `actual_vs_reference` / `captured_golden_vs_reference` 五项指标。原逐行、逐块和两侧比较仍在 report.json、query-rows.csv、query-blocks.csv。

最新结果：CPro 两阶段均五项通过；原 Prefill golden 五项全失败（relative-L2=60.141885，为比值），Decode golden 五项通过。输入 V cache 绝对值最大3.494140625，Prefill golden=1769，且588,094个元素超出 V 通道范围；CPro两阶段越界数均为0。golden 的第16–31行虽全零，mask却分别允许17–32个零偏置key，不能解释为全遮罩。错误已定位到参考计算／采集路径，具体内部原因仍待查；保留原captured golden与诊断记录。

expected_total_tasks 依据确认的源码 step=16 推导，observed_total_tasks 保留 null；工具未读取实际 tiling。本次数值结论依据完整输出对独立参考的五项比较，不用非零数量或函数名证明调度；不外推到未测形状或连续状态反馈。

## 6. Attention Prefill 调用合同

全图 `Qwen3_5MoeAttention.forward` 和单模块复用的源码片段都已经传入多行 Q、Python 列表 `actual_seq_lengths_q=[S]`、`inner_precise=2`、BNSD 和 block_table。保留这些参数；不增加不存在的 `prefill` 开关，也不把 Q reshape 成标准 ND `[B,N,S,D]` 后直接交给这个 packed 接口。

导出必须保持 `dynamic=False`，并让 `actual_seq_lengths_q` 直接成为 INT64 `Const([S])`。原 converter 已调用 `dtype_promote(..., DT_INT64)`；若实际 AIR 已是正确常量，则不需要重复修改。若它变成 Data 或运行时 Shape 路径，应在调用处恢复静态 Python 列表，在 converter 中按该列表创建 INT64 Const；不能把 `.cpu()` 的运行时 tensor 当成编译常量。新单模块检查遇到非直接 Const 或内容不符会停止导出。

`allQLen` 仍走既有的导出 carrier：Python `pse_shift=allQLen` → GE `all_seq_lengths_q`，GE `pse_shift=None`。它决定 KV 遍历长度；Decode 通常大于当前 Q 的 1。`actual_seq_lengths_kv` 在所贴内核中也用于 mask 行跨度，当前 mask 最后一维为 2048，因此不要为了切 Prefill 把这个参数随意改成 Q 长度。

用户于 2026-11-13 确认这份源码的 `DEFAULT_QUERY_SEQ_STEP=16`。所贴内核统一调用 `pa.Decode`，其中 `mActual>1` 是多行计算路径，`mActual==1` 是单行路径。因此当前 T64/T256 Prefill 的所有 Q 块均应为 `mActual=16`，Decode 为 1；无需通过修改 Q 布局或额外模式参数来切换。其他非 16 整倍数的 Prefill 尾块仍可能只有一行。此结论依据用户确认的源码常量，实际 OMC 的 tiling 尚未回传。

| 场景（B1、16 Q heads） | Q 块数 / head | 每块 mActual | 总任务数 / max(endBlk) |
| --- | ---: | ---: | ---: |
| T64 Prefill | 4 | 16 | 64 |
| T256 Prefill | 16 | 16 | 256 |
| T1 Decode | 1 | 1 | 16 |

总任务数为 `B × heads × ceil(S/16)`；blockDim 只是启动核数，不能代替总任务数。对于当前 block_size=32、head_dim=256、roundM=16 的完整块，所贴 QK 两个 16×16 块和 PV 十六个 16×16 块在块数上足够覆盖，先前 mActual>16 的覆盖疑虑不适用于这两组 Prefill；这不代表其他搬运、布局或同步已经验证。

源码 AIR 中有 Const 只证明调用侧满足静态输入合同，不保证当前编译器让 tiling 的 `GetData<int64_t>()` 非空。所贴 host 在该指针为空时退回 `batch×heads`，当 S 大于 querySeqStep 时存在任务不足风险。这是漏算任务，不会因此自动把 mActual 变成 1：例如 B1、16 heads、S256、step16 时，16 个任务按所贴索引只覆盖 head0 的 16 个 Q 块，其余 heads 未覆盖。先检查现有编译日志/tiling：`qTokens=S`、`actualSeqLengthsQ=S`、`querySeqStep`、各核 `endBlk` 的最大值。只有实际任务数覆盖全部 Q 块，才能认定 Prefill 调度完整；不根据函数名或一个 blockDim 值下结论。

## 7. 整层 GDN：FP16 状态测试

本节替代此前 FP32 请求流程。复用成功采集的 golden-capture；导出环境仍用已跑通的 qwen3_5_cpro，编译环境仍用 custom_pypto 与选定 DDK。只重导出 L0–L2 Prefill／Decode 六图，不重新采集 golden。

**A. 更新代码，清理已结束的旧 FP32 请求产物**

在持有旧 gdn-layer/air 或 omc 的各主机各执行一次；同一主机只执行一次。下面只接受原六图、FP32 请求，以及未上板的 prepare 失败。先保存编译与准备摘要，再释放本组 air/omc/execution 槽位；保留 golden-capture、state-inspection 和其他测试。已导出的 FP16 批次会被拒绝清理，直接继续其下一阶段。

```bash
export REPO_DIR="$(git rev-parse --show-toplevel)"
git -C "$REPO_DIR" pull --ff-only origin main
export AI_RUN_DIR="${AI_RUN_DIR:-${REPO_DIR}-run}"
export BATCH_ROOT="$AI_RUN_DIR/target/tests/gdn-layer"
export PYTHONDONTWRITEBYTECODE=1
python - "$BATCH_ROOT" <<'PY_RESET'
import fcntl, json, os, shutil, sys
from pathlib import Path
root = Path(sys.argv[1])
if root.is_symlink() or root.resolve() != Path(os.environ['AI_RUN_DIR']).resolve()/'target/tests/gdn-layer':
    raise SystemExit('只处理当前 run 的固定 gdn-layer 目录')
root.mkdir(parents=True, exist_ok=True)
with (root/'.omc.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    expected = {f'l{layer:02d}_c{call:02d}_gdn_layer_all' for layer in range(3) for call in range(2)}
    paths = [root/n for n in ('air', 'omc', 'execution')]
    if any(p.is_symlink() for p in paths):
        raise SystemExit('不清理符号链接')
    if any((root/n).exists() or (root/n).is_symlink() for n in ('recovery', 'comparison')):
        raise SystemExit('已有回收或比较记录，先核对本组任务状态')
    saved = {}
    for name in ('air', 'omc'):
        folder = root/name
        if not folder.exists():
            continue
        m = json.loads((folder/'manifest.json').read_text())
        if (m.get('artifact_kind') != 'qwen35-target-cpro-boundary-probe-v1'
                or m.get('status') != 'EXPORTED'
                or m.get('candidate_options', {}).get('gdn_state_dtype') != 'float32'
                or len(m.get('cases', [])) != 6
                or {c['id'] for c in m['cases']} != expected
                or any(c.get('kind') != 'gdn_layer' or c.get('status') != 'EXPORTED' for c in m['cases'])):
            raise SystemExit('不是已结束的原六图 FP32 请求批次: ' + str(folder))
        saved[name+'_manifest'] = m
    if (root/'omc').exists():
        compiled = json.loads((root/'omc/compile.json').read_text())
        if (compiled.get('status') != 'COMPILED' or set(compiled.get('cases', {})) != expected
                or any(c['status'] != 'COMPILED' for c in compiled['cases'].values())):
            raise SystemExit('旧编译批次不是已确认的六图完成状态')
        saved['compile'] = compiled
    if (root/'execution').exists():
        result = json.loads((root/'execution/result.json').read_text())
        if (result.get('status') != 'FAILED' or result.get('stage') != 'prepare'
                or result.get('device_execution') != 'NOT_RUN'
                or result.get('lock_state') != 'NO_KNOWN_OWNED_LOCK'
                or result.get('launch_requested') or result.get('board_attempt')):
            raise SystemExit('执行记录不是已结束的上板前失败，先回收原任务')
        saved['prepare'] = result
    if saved:
        (root/'fp32-contract-summary.json').write_text(json.dumps(saved, ensure_ascii=False, indent=2)+'\n')
    for folder in paths:
        if folder.exists():
            shutil.rmtree(folder)
print('旧 FP32 请求产物已处理，可以导出 FP16 六图')
PY_RESET
```

**B. 在 CPro AIR 导出主机执行**

TARGET_SOURCE/TARGET_CONFIG 沿用此前成功导出的私有源码和权重配置。保持原 T256/T1、chunk64/1、effective_length 和权重。

```bash
export REPO_DIR="$(git rev-parse --show-toplevel)"
export AI_RUN_DIR="${AI_RUN_DIR:-${REPO_DIR}-run}"
export GOLDEN_CAPTURE="${GOLDEN_CAPTURE:-$AI_RUN_DIR/target/tests/golden-capture}"
: "${TARGET_SOURCE:?请设置原 Target 源码根目录}"
: "${TARGET_CONFIG:?请设置原 Target 权重 YAML 路径}"
export BATCH_ROOT="$AI_RUN_DIR/target/tests/gdn-layer"
export BATCH="$BATCH_ROOT/air"
export PYTHONDONTWRITEBYTECODE=1
export CPRO_EXPORT_SCRIPT="${CPRO_EXPORT_SCRIPT:-$TARGET_SOURCE/export_glm_model_qwen3_5_moe.py}"
(
set -e
test -f "$GOLDEN_CAPTURE/manifest.json"
test -f "$TARGET_CONFIG"
test -f "$CPRO_EXPORT_SCRIPT"
test ! -e "$BATCH"
PYTHONPATH="$REPO_DIR/model${PYTHONPATH:+:$PYTHONPATH}" python - "$GOLDEN_CAPTURE/manifest.json" <<'PY_CHECK'
import json, sys
from qwen35_moe_dflash.target.layer_probe.whole_gdn import assemble
m = json.load(open(sys.argv[1]))
gdn = assemble(m)
expected = {(layer, call, phase) for layer in range(3)
            for call, phase in enumerate(('prefill', 'decode'))}
if (m.get('status') != 'CAPTURED' or len(gdn) != 6
        or {(r['layer'], r['call'], r['phase']) for r in gdn} != expected):
    raise SystemExit('需要 L0–L2 Prefill/call0、Decode/call1 的完整 GDN 捕获')
print('CAPTURE: 6 whole GDN graphs; 18 outputs')
PY_CHECK
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" export \
  --capture "$GOLDEN_CAPTURE" \
  --source-root "$TARGET_SOURCE" --target-config "$TARGET_CONFIG" \
  --converter-script "$CPRO_EXPORT_SCRIPT" \
  --points gdn_layer --qlinear-scales fp16 --gdn-state-dtype float16 \
  --phases prefill decode --output "$BATCH"
)
```

预期 EXPORTED 6/6。每图 export.json 的 candidate.state_dtype、candidate_graph_check.initial_state/last_recurrent_state/writeback 和 output_contract.required_dtypes["2"] 均应为 float16，golden_state_dtype 仍为 float32。源图还检查 g 的真实 FP32→FP16 Cast，以及 query/key/value/beta 为 FP16、effective_length 为 INT16。

私有 exporter 的 stock Meta 将末态声明为 FP32，因此进程内 converter 用显式 Cast 将真实 FP16 算子输出接回该前端类型，原 forward 再写回 FP16。此桥接没有恢复精度，也没有声称内核使用 FP32；直接算子输出与整图状态输出都检查为 FP16。旧 --gdn-state-dtype float32 仍执行严格 FP32 检查，不能用新代码直接放行旧批次。

### 7.1 AIR 导出后：编译与精度测试

完成上节 FP16 六图导出后，顺序执行编译 → 上板 → 回收 → 比较。使用新 target/tests/gdn-layer/air，不复用原 FP32 请求的 AIR／OMC。跨主机时更新代码并传输新 air，保留其输入与 golden。

**A. 在编译主机初始化路径**

进入仓库，使用此前 Attention 编译成功的环境（custom_pypto）和同一 DDK。AI_RUN_DIR 指向存放这批 AIR 的运行目录，默认沿用仓库旁的运行目录。跨主机时复制整个 air 目录，包含 manifest、schema、输入和 golden。

```bash
export REPO_DIR="$(git rev-parse --show-toplevel)"
git -C "$REPO_DIR" pull --ff-only origin main
export AI_RUN_DIR="${AI_RUN_DIR:-${REPO_DIR}-run}"
export BATCH_ROOT="$AI_RUN_DIR/target/tests/gdn-layer"
export BATCH="$BATCH_ROOT/air"
export OMC_BATCH="$BATCH_ROOT/omc"
export PYTHONDONTWRITEBYTECODE=1
: "${CPRO_DDK_ROOT:?请设置此前成功编译 Attention 的 DDK 根目录}"
export CPRO_DDK_ID="${CPRO_DDK_ID:-100.520.020.010}"
export TBE_DIAGNOSIS="${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}"
```

**B. 编译这六张 GDN 图**

```bash
(
set -e
test -f "$BATCH/manifest.json"
test -f "$TBE_DIAGNOSIS/report.json"
test ! -e "$OMC_BATCH"
python - "$BATCH/manifest.json" <<'PY_DTYPE'
import json, sys
m = json.load(open(sys.argv[1]))
if m.get('candidate_options', {}).get('gdn_state_dtype') != 'float16':
    raise SystemExit('需要第 7 节新导出的 FP16 批次；不能复用旧 FP32 请求的 AIR')
PY_DTYPE
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" compile \
  --batch "$BATCH" --output "$OMC_BATCH" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --compiler-profile isolated-kirinx90 \
  --tbe-diagnosis "$TBE_DIAGNOSIS" --use-origin-format
)
cat "$OMC_BATCH/compile.json"
```

预期 `COMPILED 6/6`。失败项看 `$OMC_BATCH/models/<case>/omc/compile.log`，保留现有 AIR；回传 `compile.json` 和失败日志即可继续定位。OMC 目录若已经存在，先查看其中 compile.json，不覆盖重编。跨主机上板时复制整个 BATCH_ROOT，保留 air／omc 相对位置；上板会读取两者。

**C. 在上板主机执行**

沿用此前 Attention 成功的 HDC、runtime.json 和 runner。在仓库目录执行；如果编译和上板是同一主机，继续使用当前环境。

```bash
export REPO_DIR="$(git rev-parse --show-toplevel)"
git -C "$REPO_DIR" pull --ff-only origin main
if test -f "${QWEN35_ENV_CONFIG:-$REPO_DIR/../qwen35-cpro.env}"; then
  source "$REPO_DIR/targets/cpro/scripts/env.sh"
fi
export AI_RUN_DIR="${AI_RUN_DIR:-${REPO_DIR}-run}"
export BATCH_ROOT="$AI_RUN_DIR/target/tests/gdn-layer"
export OMC_BATCH="$BATCH_ROOT/omc"
export EXECUTION="$BATCH_ROOT/execution"
export RECOVERY="$BATCH_ROOT/recovery"
export COMPARISON="$BATCH_ROOT/comparison"
export RUNTIME_CONFIG="${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}"
export HDC="${HDC:-hdc}"
export PYTHONDONTWRITEBYTECODE=1
(
set -e
: "${BIP:?沿用原 HDC 服务端 IP:端口}" "${SN:?沿用原设备序列号}"
: "${CPRO_DDK_ROOT:?请设置上板主机实际 DDK 根目录}"
test -f "$RUNTIME_CONFIG"
test -f "$OMC_BATCH/compile.json"
test -f "$BATCH_ROOT/air/manifest.json"
python "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --compiled "$OMC_BATCH" --config "$RUNTIME_CONFIG" \
  --ddk-root "$CPRO_DDK_ROOT" --output "$EXECUTION" --apply
)
```

每图预热 3 次、采样 10 次，每次恢复捕获输入。部分图编译失败时只执行已编译项，缺项不算通过。启动后使用下面的 recover 回收同一个请求，不重复 apply。

**D. 回收结果**

```bash
python "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --recover "$EXECUTION" --output "$RECOVERY"
```

若返回 PENDING，稍后重复本条 recover。回收完成并取得 native 输出后进入比较；数值失败导致回收非零时，仍可使用已回收输出进行五项比较。

**E. 比较三个输出**

```bash
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" compare \
  --fixture "$EXECUTION/fixture" --native "$RECOVERY/native" \
  --output "$COMPARISON"
cat "$COMPARISON/summary.csv"
```

完整结果应有 18 行：L0–L2 × Prefill／Decode × hidden／new_conv_state／recurrent_state。state 的 output 名称沿用 golden_recurrent_state_fp32，表示参考数据类型；实际输出为 FP16，golden 为 FP32，可直接看新增 actual_dtype/golden_dtype 两列。五项仍按 cosine >0.99、relative-L2 <0.05、KL <0.005、RMSE <1、MeanRelativeError <1，且检查状态 dtype 与重复稳定性。

回传 `gdn-layer/comparison/summary.csv` 和 `report.json`。这些是整层串联结果，仍未测试跨调用的连续状态反馈。

### 7.2 编译失败时查看日志

仅在新 FP16 批次编译失败时读取对应 $OMC_BATCH/models/<case>/omc/compile.log。已有 AIR 不必重新导出；回传 compile.json 与首个失败日志末尾即可定位。当前安装检查已经结束，无需重复扫描 DDK。

### 7.3 修复 ConvState → Conv2D 的重复升维

此前 DDK padded 为 [1,8192,1,259]／[1,8192,1,4]，旧 unsqueeze 又升为五维。现使用 padded.reshape(batch_size, conv_dim, 1, -1)，兼容源函数三维与 DDK 四维 padded；卷积权重和元素顺序保持。随后原 FP32 请求批次已 6/6 编译成功。本次 FP16 导出保留此修复。

### 7.4 编译后 state 仍为 FP16：只读定位

此轮定位已结束，原批次记录见[状态接口证据](../specs/target-gdn-state-contract-results.json)。六图源 AIR／保留 dump 声明 FP32，最终 OMC 由 NPUCL SubGraph_0 直接输出 FP16；prepare 按原 FP32 要求拦截，设备未执行。

当前安装 DDK 100.520.020.010 的两份 ChunkGatedDeltaRule 注册都声明 g、initial_state、last_recurrent_state 为 FP16。C++ 源码 SHA256 01634d23da746856c1119a9743d7f6db5f66aef9560d745423556a9c3162125a 与历史相同，第 1855／1858 行均直接用 half 指针绑定状态缓冲区。扫描 PARTIAL 仅来自两个候选目录不存在，未否定这些已读取的声明和源码。编译二进制选择与内核内部累加 dtype 仍未独立核验。

用户已确认按 FP16 跑，状态 shape [1,32,128,128] 的实际 FP16 输出为 1,048,576 bytes，原 FP32 golden 为 2,097,152 bytes；读取时分别使用各自 dtype/bytes，保持五项比较门槛。无需使用 OMG --output_type 补 FP32 输出。整层新批次已回传 18 项数值失败，当前继续第 7.6 节。

### 7.5 准备阶段 torch_npu 加载失败后重试

适用于本次 `stage=prepare`、`device_execution=NOT_RUN`、`Failed to load the backend extension: torch_npu`。原因是 compiled_state_contract 所在模块曾在导入时加载 Torch；现在只有源 AIR 导出的 converter 适配才导入 Torch。准备和 ABI 检查不再触发该后端。整层六图的 `_gdn_layer_all` 用例名也已纳入启动、回收共用的校验。

保留现有 FP16 AIR、OMC 和原始 golden，不需要重新采集、导出或编译，也不需要更换上板环境。下面只清理已确认未上板的失败 execution，并保存其 result.json；有设备请求或其他文件时停止清理。

**A. 更新代码与路径（原上板主机）**

```bash
export REPO_DIR="$(git rev-parse --show-toplevel)"
git -C "$REPO_DIR" pull --ff-only origin main
export AI_RUN_DIR="${AI_RUN_DIR:-${REPO_DIR}-run}"
export BATCH_ROOT="$AI_RUN_DIR/target/tests/gdn-layer"
export OMC_BATCH="$BATCH_ROOT/omc"
export EXECUTION="$BATCH_ROOT/execution"
export RECOVERY="$BATCH_ROOT/recovery"
export COMPARISON="$BATCH_ROOT/comparison"
export RUNTIME_CONFIG="${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}"
export PYTHONDONTWRITEBYTECODE=1
```

**B. 清理这次准备失败的 execution**

原命令已经退出后执行。它保留 `prepare-failure.json`，AIR／OMC 不变；不能用第 7 节 A 的旧 FP32 批次清理命令代替。

```bash
PYTHONPATH="$REPO_DIR/model${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY_RETRY'
import fcntl, json, os, shutil
from pathlib import Path
from qwen35_moe_dflash.execution.run_layout import _active_processes

run = Path(os.environ['AI_RUN_DIR']).resolve()
root = Path(os.environ['BATCH_ROOT']).absolute()
if root.is_symlink() or root.resolve() != run/'target/tests/gdn-layer':
    raise SystemExit('BATCH_ROOT 不是本轮整层 GDN 测试目录')
with (root/'.omc.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if _active_processes(root):
        raise SystemExit('本组测试仍有本地进程，保留 execution')
    manifest = json.loads((root/'air/manifest.json').read_text())
    if manifest.get('candidate_options', {}).get('gdn_state_dtype') != 'float16':
        raise SystemExit('需要已导出的 FP16 批次；旧 FP32 批次按第 7 节处理')
    folder, saved = root/'execution', root/'prepare-failure.json'
    if folder.is_symlink() or saved.is_symlink():
        raise SystemExit('不处理符号链接')
    if folder.exists():
        if any(p.name not in {'result.json', 'fixture'} for p in folder.iterdir()):
            raise SystemExit('发现准备阶段之外的文件，保留 execution')
        if any(p.is_symlink() for p in folder.rglob('*')):
            raise SystemExit('execution 内有符号链接，保留原目录')
        result = json.loads((folder/'result.json').read_text())
        if (result.get('artifact_kind') != 'qwen35-draft-vector-validation'
                or result.get('status') != 'FAILED' or result.get('stage') != 'prepare'
                or result.get('device_execution') != 'NOT_RUN'
                or result.get('lock_state') != 'NO_KNOWN_OWNED_LOCK'
                or result.get('launch_requested') or result.get('board_attempt')
                or result.get('uncertain') or result.get('active_child_count')
                or 'Failed to load the backend extension: torch_npu' not in result.get('error', '')):
            raise SystemExit('不是本次已结束的 Torch 导入失败，保留 execution')
        saved.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        shutil.rmtree(folder)
print('execution 可重试，现有 AIR／OMC 保留')
PY_RETRY
```

**C. 重试上板**

```bash
(
set -e
: "${BIP:?沿用原 HDC 服务端 IP:端口}" "${SN:?沿用原设备序列号}"
: "${CPRO_DDK_ROOT:?沿用本机实际 DDK 根目录}"
test -f "$RUNTIME_CONFIG"
test -f "$OMC_BATCH/compile.json"
test -f "$BATCH_ROOT/air/manifest.json"
test ! -e "$EXECUTION"
python "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --compiled "$OMC_BATCH" --config "$RUNTIME_CONFIG" \
  --ddk-root "$CPRO_DDK_ROOT" --output "$EXECUTION" --apply
)
```

启动成功后按第 7.1 节 D／E 回收和比较同一个请求。准备阶段修复仅通过本地模拟 OMC 描述的六用例检查（禁止导入 Torch），不代表设备执行或整层精度已通过；仍需回传 18 个输出的 summary.csv。

### 7.6 整层 GDN 数值失败后的只读检查

最新只读检查已确认 Prefill 三图 ConvState 全零且 10 次均等于旧输入；Decode 三图与旧输入不同，不能继续按“六图都返回旧状态”定位。六图 linear_update_tensor 全零来自 capture.py 的 ORIGINAL_CPRO_EXPORT_ZERO_INT32_128 占位策略；effective_length=256／1 是另一个输入，不能视为 ConvState 已收到更新长度。

**A. 在保存 execution／recovery 的主机检查布局假设。** 更新后的原命令额外生成 conv-layout-summary.txt，列出源 BCK、按 BKC 序列化、按 [B,C/16,K,16] 序列化三种解读。它同时比较四个历史位置、Decode 左移历史与原输入的关系。矩阵每项是 B*C 个元素中的相等数量，行是输出历史位置，列是旧输入或 golden 的历史位置；完整逐位结果保存在 report.json。所有候选都报告，未选定任何布局，不修改真实输入、输出或 golden，不生成新精度通过结论。

下面使用本次固定目录 convstate-inspection；已完成的 output-inspection 保留。本步只需 NumPy，不需要 Torch、DDK 或设备。

```bash
export REPO_DIR="$(git rev-parse --show-toplevel)"
git -C "$REPO_DIR" pull --ff-only origin main
export AI_RUN_DIR="${AI_RUN_DIR:-${REPO_DIR}-run}"
export BATCH_ROOT="$AI_RUN_DIR/target/tests/gdn-layer"
export EXECUTION="$BATCH_ROOT/execution"
export RECOVERY="$BATCH_ROOT/recovery"
export INSPECTION="$BATCH_ROOT/convstate-inspection"
export PYTHONDONTWRITEBYTECODE=1
(
set -e
test -f "$EXECUTION/fixture/manifest.json"
test -d "$RECOVERY/native"
test ! -e "$INSPECTION"
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" inspect-gdn-output \
  --fixture "$EXECUTION/fixture" --native "$RECOVERY/native" \
  --output "$INSPECTION"
)
cat "$INSPECTION/conv-layout-summary.txt"
```

**B. 在编译主机读取当前 DDK 的 ConvState 文本定义。** 这是状态更新算子的控制／布局检查，区别于已完成的 GDN FP16 注册检查。只搜索所选 DDK 的 ddk、tools、include 文本目录，不执行编译器或修改安装。需要 rg；可与 A 在同一主机顺序执行。

```bash
export REPO_DIR="$(git rev-parse --show-toplevel)"
export AI_RUN_DIR="${AI_RUN_DIR:-${REPO_DIR}-run}"
export BATCH_ROOT="$AI_RUN_DIR/target/tests/gdn-layer"
(
set -e
: "${CPRO_DDK_ROOT:?沿用编译这六图的 DDK 根目录}"
command -v rg >/dev/null
mkdir -p "$BATCH_ROOT/convstate-inspection"
ddk_text_roots=()
for part in ddk tools include; do
  if test -d "$CPRO_DDK_ROOT/$part"; then
    ddk_text_roots+=("$CPRO_DDK_ROOT/$part")
  fi
done
test "${#ddk_text_roots[@]}" -gt 0
search_rc=0
rg -n -i -C 12 --max-filesize 4M --max-columns 400 --max-columns-preview \
  --glob '*.{cpp,c,cc,h,hpp,py,json,ini}' \
  'LinearAttentionConvstateUpdate|linear_attention_convstate_update' \
  "${ddk_text_roots[@]}" \
  > "$BATCH_ROOT/convstate-inspection/ddk-convstate.txt" \
  2> "$BATCH_ROOT/convstate-inspection/ddk-search-errors.txt" || search_rc=$?
if test "$search_rc" -eq 1; then
  echo '未找到匹配文本；不能据此断言算子不存在。' > "$BATCH_ROOT/convstate-inspection/ddk-convstate.txt"
elif test "$search_rc" -ne 0; then
  cat "$BATCH_ROOT/convstate-inspection/ddk-search-errors.txt"
  exit "$search_rc"
fi
)
cat "$BATCH_ROOT/convstate-inspection/ddk-convstate.txt"
```

回传 conv-layout-summary.txt 和 ddk-convstate.txt。如果 DDK 只有接口／shape 声明，还需要 LinearAttentionConvstateUpdate 的实现或 llmConfig[128] 字段定义；不能把零占位直接猜改为某个长度。源码中的候选声明也不单独证明编译后采用的实现。

当前 AIR／OMC、execution、recovery 和 comparison 均继续使用。A 的目录若已存在，直接读取已生成的报告。跨主机时先完成 A，再把 B 的两个文本结果放在对应诊断目录；没有新增导出、编译或上板任务。
