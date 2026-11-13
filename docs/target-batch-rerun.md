# Target：整层 GatedDeltaNet FP32 state 与 Fused Attention

Attention 当前定位先运行[已有输出检查](target-layer-accuracy.md#attention-输出范围检查)，读取产生 Prefill 99.67% 误差的那批 AIR／OMC、fixture 和 native，按 head／16 行 Q 块查看误差。此步骤不需要重新导出、编译或上板；下面是新一组整层 GDN／Attention 的完整执行流程。

当前新增一组 **8 张图、20 个输出**，覆盖 Prefill 和第一次 Decode：

| 模块 | 图数 | 每张图输出 |
| --- | ---: | --- |
| L0–L2 整层 GatedDeltaNet | 6 | hidden、new_conv_state、recurrent_state_fp32 |
| L3 AdnFusedInferAttention | 2 | attention_nz |

GDN 包含五个 QLinear、ConvState、卷积、SiLU、g/beta、Q/K 重复、ChunkGatedDeltaRule、gated norm 和 out_proj，层内真实串联。它不是把 core/state 再分成小图，也不包含 Decoder 的外部残差或 MoE。QLinear 沿用已验证的 FP16 scale 适配；卷积沿用显式单高度 Conv2d。

输入和三个 golden 从同一 CAPTURED layer/call/phase 的已有边界组合，检查四个输入投影的 hidden 一致；没有重算 golden。原 FP16 递归输入数值转换成 FP32，ChunkGatedDeltaRule 的 initial_state、last_recurrent_state 和层内 writeback 均声明 FP32，末态对比原始 raw-state FP32 golden。**末端 Cast 不算内部精度修复**：准备上板时拒绝非 FP32 状态 ABI 及可见 FP16→FP32 补 Cast，保留 GraphOp 链；不声称能从不透明内核外部证明累加精度。若 DDK 算子仍只支持 FP16 state，需对应算子实现支持，不能改 AIR dtype 标签绕过。

Attention 沿用捕获 Q/K/V、mask、block_table 和长度参数，新增 Q 轴、Q 长度常量字节与 allQLen 传递检查。当前范围为 B=1 静态图。Q 为 `[B, heads×D/16, S, 16]`，序列轴是 2；原捕获 Prefill S256、Decode S1。源码全图 S64 对应 `[1,256,64,16]`，捕获小图对应 `[1,256,256,16]`；两者均没有把 Prefill Q 变成 1。`allQLen` 单独记录，不能强制等于当前 Q 长度；它在用户提供的内核中控制 KV 块循环。实际 tiling 和完整生产图仍需编译产物核对。

每张图每次都使用捕获输入；此轮包含层内误差传递，仍不包含跨图连续 Prefill→Decode 状态反馈。旧 48 个独立边界已回传 43 PASS / 5 FAIL，本组是新实验，尚无设备结果。已通过的独立 QLinear、RoPE、Prefill Cache 不单独复跑。

## 1. 初始化路径

各主机沿用此前成功的 Python／vendor 环境。AIR 主机需要原捕获、私有 Target 源码和权重 YAML；编译主机使用已选 DDK；上板主机沿用 runtime.json、HDC 和 runner。

```bash
export REPO_DIR="$(git rev-parse --show-toplevel)"
git -C "$REPO_DIR" pull --ff-only origin main
export AI_RUN_DIR="${AI_RUN_DIR:-${REPO_DIR}-run}"
export GOLDEN_CAPTURE="${GOLDEN_CAPTURE:-${CAPTURE:-$AI_RUN_DIR/target-layer-accuracy/golden}}"
export BATCH_ROOT="$AI_RUN_DIR/target/tests/gdn-layer-attention"
export BATCH="$BATCH_ROOT/air"
export OMC_BATCH="$BATCH_ROOT/omc"
export COMPARISON="$BATCH_ROOT/comparison"
export RUNTIME_CONFIG="${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}"
export EXECUTION="$AI_RUN_DIR/fused/target-gdn-layer-attention-execution"
export RECOVERY="$AI_RUN_DIR/fused/target-gdn-layer-attention-recovery"
export PYTHONDONTWRITEBYTECODE=1
```

旧 48 图不能当作整层图复用；已有本组 AIR／OMC 时，从缺失的下一阶段继续。新建命令要求空输出槽位。固定目录复测前先确认旧请求结束、已回收，再清理该测试的临时产物，不覆盖活动任务。

## 2. 导出 AIR

TARGET_SOURCE 为原私有源码根目录，TARGET_CONFIG 为此前成功的原权重 YAML。先核对原捕获包含所需整层边界，缺项即停止。

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
from qwen35_moe_dflash.target.layer_probe.whole_gdn import assemble
m = json.load(open(sys.argv[1]))
gdn = assemble(m)
attention = [r for r in m['regions'] if r['point'] == 'fused_attention']
expected = {(layer, call, phase) for layer in range(3)
            for call, phase in enumerate(('prefill', 'decode'))}
if m.get('status') != 'CAPTURED' or len(gdn) != 6 or {(r['layer'], r['call'], r['phase']) for r in gdn} != expected:
    raise SystemExit('需要 L0–L2 Prefill/call0、Decode/call1 的完整 GDN 捕获。')
if len(attention) != 2 or {(r['layer'], r['call'], r['phase']) for r in attention} != {(3,0,'prefill'), (3,1,'decode')}:
    raise SystemExit('需要 L3 Prefill/call0、Decode/call1 的 Attention 捕获。')
print('CAPTURE: 6 whole GDN + 2 attention graphs; 20 outputs')
PY_CHECK
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" export \
  --capture "$GOLDEN_CAPTURE" \
  --source-root "$TARGET_SOURCE" --target-config "$TARGET_CONFIG" \
  --converter-script "$CPRO_EXPORT_SCRIPT" \
  --points gdn_layer fused_attention \
  --qlinear-scales fp16 --gdn-state-dtype float32 \
  --phases prefill decode --output "$BATCH"
)
```

应为 EXPORTED 8/8。失败看 manifest.json 的 error 和 `$BATCH/<case>.log`。每份 export.json 保留 `source_contract.status=BOUND_TO_SOURCE`、源码表达式、输入与输出类型；不能修改描述符来掩盖 dtype 差异。跨主机复制整个 BATCH，包含清单、schema、输入、golden，不只复制 export.air。

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

摘要在 `$OMC_BATCH/compile.json`，日志在 `models/<case>/omc/compile.log`。部分失败时，下一步自动执行已 COMPILED 的项，缺项保留，不代表 8 张图全部通过。跨主机复制整个 BATCH_ROOT，保持 air／omc 相对位置。

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

本组 summary.csv 应有 20 行：整层 GDN 18 行、Attention 2 行。同一 GDN case 的三个输出按固定槽位独立统计，完整 dump 按各槽位真实 dtype/bytes 拆分。任何输出或任意一次重复失败，整组不能通过。新输出的数值和类型都需检查；FP32 接口通过仍不证明内核内部每一步都以 FP32 计算。

Attention 的 `export.json.attention_contract` 会记录 Q shape、actual_seq_lengths_q、actual_seq_lengths_kv、all_seq_lengths_q 及 AIR 传递链。`query_length_constant` 验证实际 INT64 常量内容，不只看 `_readable_value`。本轮保持算子实现不变，先核对调用与导出。

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
