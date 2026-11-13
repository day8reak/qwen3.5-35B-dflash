# Target：Attention 定位与整层 GatedDeltaNet 测试

旧 target-layer-accuracy 产物目录及 golden capture 已删除。本页从重新采集开始，所有新产物放在 target/tests 下。当前先定位 Attention：**只导出、编译 L3 Prefill／Decode 两张 Attention 图**，保持算子实现和原数值调用不变，回收后按 head／16 行 Q 块查看误差。旧报告数值不能代替已删除的输入或输出张量，也不能据此重建原 golden。

采集仍覆盖四层，供 Attention 和后续整层 GDN 共用。已实现的测试范围如下；本页默认命令只选择 Attention：

| 模块 | 图数 | 每张图输出 |
| --- | ---: | --- |
| L3 AdnFusedInferAttention（本轮） | 2 | attention_nz |
| L0–L2 整层 GatedDeltaNet（后续切换） | 6 | hidden、new_conv_state、recurrent_state_fp32 |

GDN 包含五个 QLinear、ConvState、卷积、SiLU、g/beta、Q/K 重复、ChunkGatedDeltaRule、gated norm 和 out_proj，层内真实串联。它不是把 core/state 再分成小图，也不包含 Decoder 的外部残差或 MoE。QLinear 沿用已验证的 FP16 scale 适配；卷积沿用显式单高度 Conv2d。

输入和三个 golden 从同一 CAPTURED layer/call/phase 的已有边界组合，检查四个输入投影的 hidden 一致；没有重算 golden。原 FP16 递归输入数值转换成 FP32，ChunkGatedDeltaRule 的 initial_state、last_recurrent_state 和层内 writeback 均声明 FP32，末态对比原始 raw-state FP32 golden。**末端 Cast 不算内部精度修复**：准备上板时拒绝非 FP32 状态 ABI 及可见 FP16→FP32 补 Cast，保留 GraphOp 链；不声称能从不透明内核外部证明累加精度。若 DDK 算子仍只支持 FP16 state，需对应算子实现支持，不能改 AIR dtype 标签绕过。

Attention 沿用捕获 Q/K/V、mask、block_table 和长度参数，新增 Q 轴、Q 长度常量字节与 allQLen 传递检查。当前范围为 B=1 静态图。Q 为 `[B, heads×D/16, S, 16]`，序列轴是 2；旧捕获 Prefill S256、Decode S1。源码全图 S64 对应 `[1,256,64,16]`，旧捕获小图对应 `[1,256,256,16]`；两者均没有把 Prefill Q 变成 1。新采集的 S 以实际 manifest 为准，不为了复现旧长度补齐或截断输入。`allQLen` 单独记录，不能强制等于当前 Q 长度；它在用户提供的内核中控制 KV 块循环。实际 tiling 和完整生产图仍需编译产物核对。

每张图每次都使用捕获输入；此轮包含层内误差传递，仍不包含跨图连续 Prefill→Decode 状态反馈。旧 48 个独立边界已回传 43 PASS / 5 FAIL，本组是新实验，尚无设备结果。已通过的独立 QLinear、RoPE、Prefill Cache 不单独复跑。

## 1. 初始化路径

各主机沿用此前成功的 Python／vendor 环境。AIR 主机需要原捕获、私有 Target 源码和权重 YAML；编译主机使用已选 DDK；上板主机沿用 runtime.json、HDC 和 runner。

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

## 2. 导出 AIR

TARGET_SOURCE 为原私有源码根目录，TARGET_CONFIG 为此前成功的原权重 YAML。先核对新 capture 包含两次 Attention 边界，缺项即停止。

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

摘要在 `$OMC_BATCH/compile.json`，日志在 `models/<case>/omc/compile.log`。应为 COMPILED 2/2；部分失败时，下一步自动执行已 COMPILED 的项，缺项保留。跨主机复制整个 BATCH_ROOT，保持 air／omc 相对位置。

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

本轮 summary.csv 应有 2 行：L3 Attention Prefill、Decode 各一行。任何输出或任意一次重复失败，整组不能通过。

Attention 的 `export.json.attention_contract` 会记录 Q shape、actual_seq_lengths_q、actual_seq_lengths_kv、all_seq_lengths_q 及 AIR 传递链。`query_length_constant` 验证实际 INT64 常量内容，不只看 `_readable_value`。本轮保持算子实现不变，先核对调用与导出。

### 5.1 Attention 输出检查

对本轮新产物执行以下只读检查。它检查源 AIR 的长度常量、各 head 和每 16 行 Q 的非零数、max_abs、relative_l2，所有元素参与统计。重复调用只有输出 SHA256 相同才合并统计。原始 capture 无需带到此主机，但 BATCH_ROOT/air 中的源 AIR、air-schema.pb、golden 仍需保留。

```bash
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" inspect-attention \
  --batch "$OMC_BATCH" \
  --fixture "$EXECUTION/fixture" --native "$RECOVERY/native" \
  --output "$BATCH_ROOT/inspection"

cat "$BATCH_ROOT/inspection/summary.txt"
cat "$BATCH_ROOT/inspection/head-summary.csv"
```

回传 comparison/summary.csv、inspection/summary.txt 和 inspection/head-summary.csv。report.json 和 query-blocks.csv 留在 inspection 内。若 Prefill 只有 head0 接近 golden、其余 head 全零而 golden 非零，支持任务数不足的怀疑；若各 head 都有结果，则再看逐块误差。零输出也可能来自计算或写回，不能直接认定任务未调度。expected_total_tasks 依据确认的源码 step=16 推导，observed_total_tasks 保留 null；工具未读取实际 tiling，也未独立证明设备输出布局。

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

## 7. 后续整层 GDN 测试

复用第 1.1 节的同一 golden-capture，另设 GDN 测试目录。以下命令替代第 2 节的 Attention 导出，然后沿用第 3–5 节编译、执行、回收和比较；不执行第 5.1 节的 Attention 检查。无需再次采集或重复编译已经准备好的 Attention。

```bash
export BATCH_ROOT="$AI_RUN_DIR/target/tests/gdn-layer"
export BATCH="$BATCH_ROOT/air"
export OMC_BATCH="$BATCH_ROOT/omc"
export COMPARISON="$BATCH_ROOT/comparison"
export EXECUTION="$BATCH_ROOT/execution"
export RECOVERY="$BATCH_ROOT/recovery"
export CPRO_EXPORT_SCRIPT="${CPRO_EXPORT_SCRIPT:-$TARGET_SOURCE/export_glm_model_qwen3_5_moe.py}"
(
set -e
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
    raise SystemExit('需要 L0–L2 Prefill/call0、Decode/call1 的完整 GDN 捕获。')
print('CAPTURE: 6 whole GDN graphs; 18 outputs')
PY_CHECK
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" export \
  --capture "$GOLDEN_CAPTURE" \
  --source-root "$TARGET_SOURCE" --target-config "$TARGET_CONFIG" \
  --converter-script "$CPRO_EXPORT_SCRIPT" \
  --points gdn_layer --qlinear-scales fp16 --gdn-state-dtype float32 \
  --phases prefill decode --output "$BATCH"
)
```

应导出、编译各 6 张图，summary.csv 共 18 行。每个 case 同时返回 hidden、new_conv_state、recurrent_state_fp32 三个输出，按真实 dtype/bytes 拆分比较；不是三张独立边界图。末态数值与 FP32 ABI 均需核实，FP32 接口仍不证明内核内部累加精度。整层 GDN 和 Attention 均通过后，再验证它们在整图中的状态反馈。
