# Target GDN：独立边界精度测试

当前测试 L0–L2 的 GDN 及其门控、状态转换、归一化边界，覆盖原捕获的 Prefill 和第一次 Decode，**共 48 张单输出小图**。先复现实际 CPro 源码基线，按 CPro / CHS 共用五项标准判定，再据异常边界修改实现。

| point | 每层每阶段输出 | 图数（3 层 × 2 阶段） |
| --- | --- | ---: |
| decay_beta | golden_g、golden_beta | 12 |
| repeat_qk | golden_q_repeated、golden_k_repeated | 12 |
| gdn | golden_core、golden_state_raw | 12 |
| recurrent_writeback | golden_written_state | 6 |
| linear_attn_norm | golden_y | 6 |

GDN core 和 raw state 分图导出，都来自真实调用，保留 effective_length、chunk 参数和初始状态。各图独立使用原 CAPTURED 输入和 golden；Prefill T256／Decode T1 以捕获为准，不裁成 T64、不强制将 g／state 改成 FP16。writeback 通过不能证明 CPro GDN 的错误状态被修好。当前不执行 QLinear、RoPE、Cache 或 Attention，也不使用六输入合成测试替代真实边界。[历史数值及定位顺序](target-accuracy-status.md#gdn-当前独立测试)。

## 1. 初始化路径

各主机沿用此前成功的 Python／vendor 环境。AIR 主机需要原捕获、私有 Target 源码和权重 YAML；编译主机使用已选 DDK；上板主机沿用 runtime.json、HDC 和 runner。

```bash
export REPO_DIR="$(git rev-parse --show-toplevel)"
git -C "$REPO_DIR" pull --ff-only origin main
export AI_RUN_DIR="${AI_RUN_DIR:-${REPO_DIR}-run}"
export GOLDEN_CAPTURE="${GOLDEN_CAPTURE:-${CAPTURE:-$AI_RUN_DIR/target-layer-accuracy/golden}}"
export BATCH_ROOT="$AI_RUN_DIR/target/tests/gdn-boundaries"
export BATCH="$BATCH_ROOT/air"
export OMC_BATCH="$BATCH_ROOT/omc"
export COMPARISON="$BATCH_ROOT/comparison"
export RUNTIME_CONFIG="${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}"
export EXECUTION="$AI_RUN_DIR/fused/target-gdn-boundaries-execution"
export RECOVERY="$AI_RUN_DIR/fused/target-gdn-boundaries-recovery"
export PYTHONDONTWRITEBYTECODE=1
```

已有这 48 项的 fixture/native 时，直接执行下方“复用旧 dump”；已有本范围的 AIR／OMC 时，从缺失的下一阶段继续。新建命令要求空输出槽位。固定目录复测前先确认旧请求结束、已回收，再清理该测试的临时产物，不覆盖活动任务。

## 2. 导出 AIR

TARGET_SOURCE 为原私有源码根目录，TARGET_CONFIG 为此前成功的原权重 YAML。先核对原捕获包含完整 48 项，缺项即停止。

```bash
: "${TARGET_SOURCE:?请设置原 Target 源码根目录}"
: "${TARGET_CONFIG:?请设置原 Target 权重 YAML 路径}"
export CPRO_EXPORT_SCRIPT="${CPRO_EXPORT_SCRIPT:-$TARGET_SOURCE/export_glm_model_qwen3_5_moe.py}"
(
set -e
test -f "$TARGET_CONFIG"
test -f "$CPRO_EXPORT_SCRIPT"
test ! -e "$BATCH"
python - "$GOLDEN_CAPTURE/manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
outputs = {
    "decay_beta": ["golden_g", "golden_beta"],
    "repeat_qk": ["golden_q_repeated", "golden_k_repeated"],
    "gdn": ["golden_core", "golden_state_raw"],
    "recurrent_writeback": ["golden_written_state"],
    "linear_attn_norm": ["golden_y"],
}
expected = {(layer, call, phase, point, name)
            for layer in range(3) for call, phase in enumerate(("prefill", "decode"))
            for point, names in outputs.items() for name in names}
actual = [(r["layer"], r["call"], r["phase"], r["point"], t["name"])
          for r in m["regions"] if r["point"] in outputs for t in r["outputs"]]
if m.get("status") != "CAPTURED" or len(actual) != 48 or set(actual) != expected:
    raise SystemExit("需要 L0–L2、Prefill/call0 + Decode/call1、五类边界的完整 48 项捕获。")
print("CAPTURE: 48 GDN-related outputs")
PY
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" export \
  --capture "$GOLDEN_CAPTURE" \
  --source-root "$TARGET_SOURCE" --target-config "$TARGET_CONFIG" \
  --converter-script "$CPRO_EXPORT_SCRIPT" \
  --points decay_beta repeat_qk gdn recurrent_writeback linear_attn_norm \
  --phases prefill decode --output "$BATCH"
)
```

应为 EXPORTED 48/48。失败看 manifest.json 的 error 和 `$BATCH/<case>.log`。每份 export.json 保留 `source_contract.status=BOUND_TO_SOURCE`、源码表达式、输入与输出类型；不能修改描述符来掩盖 dtype 差异。跨主机复制整个 BATCH，包含清单、schema、输入、golden，不只复制 export.air。

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

摘要在 `$OMC_BATCH/compile.json`，日志在 `models/<case>/omc/compile.log`。部分失败时，下一步自动执行已 COMPILED 的项，缺项保留，不代表 48 项全部通过。跨主机复制整个 BATCH_ROOT，保持 air／omc 相对位置。

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

先看 decay_beta、repeat_qk，再分别看 gdn/core 和 raw state，最后看 writeback、norm。前两者对齐而 GDN 偏差大时，再查 GDN 布局、输入 dtype、effective_length、Q/K 归一化和累加精度；不能统一归因于量化投影，因为各图独立输入 golden。

### 复用旧 dump

若旧全边界测试的 fixture/native 完整，只补算这 48 项，跳过导出／编译／上板。COMPARISON 要求空目录；不能从只有 summary.csv 的旧结果反推新指标。

```bash
: "${GDN_FIXTURE:?设置保留的执行 fixture 目录}"
: "${GDN_NATIVE:?设置匹配的回收 native 目录}"
(
set -e
GDN_CASE_LIST="$(python - "$GDN_FIXTURE/manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
outputs = {
    "decay_beta": ["golden_g", "golden_beta"],
    "repeat_qk": ["golden_q_repeated", "golden_k_repeated"],
    "gdn": ["golden_core", "golden_state_raw"],
    "recurrent_writeback": ["golden_written_state"],
    "linear_attn_norm": ["golden_y"],
}
cases = [c for c in m["cases"] if c["point"] in outputs]
expected = {(layer, call, phase, point, name)
            for layer in range(3) for call, phase in enumerate(("prefill", "decode"))
            for point, names in outputs.items() for name in names}
actual = {(c["layer"], c["call"], c["phase"], c["point"], c["output_name"]) for c in cases}
if len(cases) != 48 or len({c["case"] for c in cases}) != 48 or actual != expected:
    raise SystemExit("已有 fixture 不包含完整 48 项 GDN 边界，请检查来源。")
print("\n".join(c["case"] for c in cases))
PY
)"
mapfile -t GDN_CASES <<< "$GDN_CASE_LIST"
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" compare \
  --fixture "$GDN_FIXTURE" --native "$GDN_NATIVE" \
  --cases "${GDN_CASES[@]}" --output "$COMPARISON"
)
```

选择旧批次的子集后，`coverage_complete=false` 仍表示未覆盖整个旧 fixture；这 48 项即使通过，总状态也可能是 PARTIAL。查看 `numerical_validation` 和各 case，不能把旧批次未测项改成通过。旧数据仅作为旧图基线；修改源码后仍需重新生成对应产物验证。
