# Target RoPE：已测结果与复现命令

用户已回传 **4 张 RoPE 小图：Prefill／Decode × Q／K，全部零误差**，旋转前缀与未旋转尾部均数值一致，FP16 输出且重复稳定。当前不再复跑 RoPE、已通过的 Prefill Cache 或 QLinear；下一步处理 AdnFusedInferAttention Prefill，Decode Cache 暂后。QuantBatchMatmul／RoPE 有效改动及 Prefill 剩余问题见[精度状态](target-accuracy-status.md)，最新指标见[交互 HTML](assets/target-architecture-interactive.html)。

私有源码 `models/modeling_qwen3_5_moe_hiai_cpro.py` 中，`rotate_half` 在第 60 行、`apply_rotary_pos_emb` 在第 66 行、`Qwen3_5MoeRotaryEmbedding` 在第 140 行，`_init_rope_cache()` 在第 184 行。行号对应本地旧快照，服务器当前源码以本次实际导出的文件为准。

该快照已经在 FP32 下计算 sin/cos，查表后才转类型。当前捕获点只含 Q、K、cos、sin，不能反推角度。本轮 `--rope-precision fp32` 是**旋转乘加候选**：四个输入先转 FP32，调用实际 CPro partial RoPE helper，最后恢复各自 golden 输出 dtype。输入文件、系数和 golden 不变，不把输出强制降为 Q/K 的输入类型，也不将它称为 sin/cos 生成修复。

本地主机检查覆盖源码系数生成顺序、两种长度／系数 dtype、partial rotary 尾部、PyTorch 导出、AIR 描述检查和四图筛选。用户随后在原环境完成执行并回传零误差；未在本地重新核验服务器完整产物。原始状态 METRICS_RECORDED 表示未配置自动浮点门槛，本次数值已经对齐。**以下命令仅供今后修改后需要回归时复用，当前不用重编或上板。**

## 1. 每台主机初始化

在该主机仓库中，激活此前成功的 Python／vendor 环境。AIR 主机需要原 CAPTURED 捕获和私有 Target 源码；编译主机需要完整 AIR；上板主机沿用已成功运行的 runtime.json、HDC 和 runner。

```bash
export REPO_DIR="$(git rev-parse --show-toplevel)"
git -C "$REPO_DIR" pull --ff-only origin main
export AI_RUN_DIR="${AI_RUN_DIR:-${REPO_DIR}-run}"
export GOLDEN_CAPTURE="${GOLDEN_CAPTURE:-${CAPTURE:-$AI_RUN_DIR/target-layer-accuracy/golden}}"
export BATCH_ROOT="$AI_RUN_DIR/target/tests/rope-candidate"
export BATCH="$BATCH_ROOT/air"
export OMC_BATCH="$BATCH_ROOT/omc"
export COMPARISON="$BATCH_ROOT/comparison"
export RUNTIME_CONFIG="${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}"
export EXECUTION="$AI_RUN_DIR/fused/target-rope-execution"
export RECOVERY="$AI_RUN_DIR/fused/target-rope-recovery"
export PYTHONDONTWRITEBYTECODE=1
```

AIR／OMC／comparison 要求空输出槽位。后续复测复用这些固定路径；先确认上次执行结束并回收，再处理它的临时产物，不覆盖活动请求或公共模型。

## 2. 导出四张 RoPE 图

在原 TorchAir 环境设置 TARGET_SOURCE 为私有 Target 源码根目录，converter 沿用此前成功路径。RoPE 无权重，不需要权重 YAML。

```bash
: "${TARGET_SOURCE:?请设置原 Target 源码根目录}"
export CPRO_EXPORT_SCRIPT="${CPRO_EXPORT_SCRIPT:-$TARGET_SOURCE/export_glm_model_qwen3_5_moe.py}"
(
set -e
test -f "$GOLDEN_CAPTURE/manifest.json"
test -f "$CPRO_EXPORT_SCRIPT"
test ! -e "$BATCH"
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" export \
  --capture "$GOLDEN_CAPTURE" \
  --source-root "$TARGET_SOURCE" \
  --converter-script "$CPRO_EXPORT_SCRIPT" \
  --points rope --phases prefill decode --rope-precision fp32 \
  --output "$BATCH"
python - "$BATCH/manifest.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
cases = r["cases"]
expected = {(p, n) for p in ("prefill", "decode")
            for n in ("golden_q_rotated", "golden_k_rotated")}
if (r["status"] != "EXPORTED" or len(cases) != 4
        or {(c["phase"], c["output_name"]) for c in cases} != expected
        or any(c["kind"] != "rope" or c["status"] != "EXPORTED" for c in cases)):
    raise SystemExit("需要完整四张 RoPE 图；请检查捕获选择及各 case 日志。")
print("AIR: EXPORTED 4/4 RoPE")
PY
)
```

每份 `models/<case>/export.json` 应有：

- `candidate.kind=ROPE_ROTATION_FP32`，`candidate_graph_check.status=PASSED`。
- `rope_contract.rotary_dim`、原输出 dtype 和系数生成函数的来源信息。
- `rope_contract.host_check.outputs`：原 CPro helper／FP32 候选与原 golden 的 CPU 对照，仅供诊断。

AIR 检查逐条核对乘加输入的生产端／消费端及输出均为 FP32，同时核对最终输出 dtype；不证明 OMC 内部未再降精度。失败时查看 `$BATCH/<case>.log`，不要绕过检查。跨机器复制**整个 BATCH 目录**，含 schema、输入、golden、清单，不能只拷 export.air。

## 3. 编译 OMC

在此前成功的 custom_pypto／DDK 环境执行。以下默认 DDK 是此前实测配置，TBE_DIAGNOSIS 沿用已经完成的诊断目录。

```bash
export CPRO_DDK_ROOT="${CPRO_DDK_ROOT:-/home/l00886453/ddk_0806/image/hiai_ddk/ddk_internal}"
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

摘要在 `$OMC_BATCH/compile.json`，单项日志在 `models/<case>/omc/compile.log`。若部分失败，成功项仍可上板，但不能视为四图完整验证。上板在另一主机时复制整个 BATCH_ROOT，保持 air／omc 相对位置。

## 4. 上板与回收

沿用此前跑 Cache/QLinear 的 runtime.json 和 runner，本轮无需重建运行器。在原 HDC 主机设置实际 DDK 路径及原 BIP／SN。

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

该入口虽名为 draft，也支持 Target。每图预热 3 次、采样 10 次，每次恢复捕获输入。启动后回收**同一个请求**：

```bash
python "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --recover "$EXECUTION" --output "$RECOVERY"
```

若返回 PENDING，稍后重复回收，不重复 apply。数值异常可能使回收返回非零；有 native 输出后仍可继续比较。

## 5. 比较整输出及旋转分区

```bash
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" compare \
  --fixture "$EXECUTION/fixture" --native "$RECOVERY/native" \
  --output "$COMPARISON"

python - "$COMPARISON/report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
for row in r["cases"]:
    m = row.get("invocations", [{}])[0]
    p = m.get("rope_partitions", {})
    print(json.dumps({
        "case": row["case"], "status": row["status"],
        "whole_relative_l2": m.get("relative_l2"),
        "rotary_dim": p.get("rotary_dim"),
        "rotated": p.get("rotated"), "unrotated": p.get("unrotated"),
        "dtype_preserved": row.get("source_dtype_preserved"),
        "stable": row.get("stable"), "error": row.get("error")
    }, ensure_ascii=False))
print("coverage_complete:", r["coverage_complete"])
PY
```

回传 summary.csv 和 report.json。整输出指标保持原定义；新增 `rope_partitions.rotated`／`unrotated` 只按捕获的 rotary_dim 切分最后一维，不删 token、不改变 padding／布局、不放宽门槛。RoPE 尚无批准的误差阈值，默认仍记录 METRICS_RECORDED，不能套用量化 out_proj 的 2%。

若未旋转部分已有明显误差，先查输入／输出绑定和编译后的类型、布局；若误差集中于旋转部分，再对照本次 AIR／OMC 的乘加与转换。候选改善也不能归功于本图中不存在的 Sin／Cos 修改。

## 暂后的 Cache 工作

Decode Cache 仍有四项失败；恢复 Cache 工作时只复测 Decode，已通过的 Prefill 不重编。当前不因 RoPE 修改启动 Cache。此前证据与修复方向见[精度状态](target-accuracy-status.md#decode-cache-原因分析)。

## 先读已有报告，无需重编或上板

在保留本次数值的回收主机执行，COMPARISON 沿用已有实际目录。默认指向本页 Cache 固定测试目录；若本批仍在 qlinear-cache-candidates 下，设置为该批 comparison 目录。命令只读，不改变数值或产物。

```bash
export REPO_DIR="$(git rev-parse --show-toplevel)"
export AI_RUN_DIR="${AI_RUN_DIR:-${REPO_DIR}-run}"
export COMPARISON="${COMPARISON:-$AI_RUN_DIR/target/tests/cache-candidates/comparison}"

python - "$COMPARISON/report.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"未找到 {path}；请把 COMPARISON 设置为产生本批表格的 comparison 目录。")
report = json.loads(path.read_text())
pairs = {}
for row in report["cases"]:
    if row.get("phase") != "decode" or not row.get("point", "").startswith("cache_"):
        continue
    invocations = row.get("invocations", [])
    m = invocations[0] if invocations else {}
    hashes = row.get("output_sha256", [])
    first_hash = hashes[0] if hashes else None
    print(json.dumps({
        "case": row["case"], "status": row["status"],
        "mismatched_preserved_elements": m.get("mismatched_preserved_elements"),
        "mismatched_updated_elements": m.get("mismatched_updated_elements"),
        "max_error_coordinate": m.get("max_error_coordinate"),
        "actual_at_max": m.get("actual_at_max"),
        "golden_at_max": m.get("golden_at_max"),
        "output_sha256_0": first_hash, "error": row.get("error"),
    }, ensure_ascii=False))
    point = row["point"]
    if "_index_" in point:
        base, mode = point.rsplit("_index_", 1)
        pairs.setdefault((row["layer"], row["call"], base), {})[mode] = first_hash
for key, pair in sorted(pairs.items()):
    a, b = pair.get("float"), pair.get("integer")
    print(key, "float_integer_same_bytes:",
          a == b if a and b else "UNKNOWN_MISSING_HASH")
PY
```

已有 V 片段说明历史未保留；哈希相同才进一步支持两种索引生成了逐字节相同输出。分区计数定义为 golden 与输入未变化／变化的位置，不完全等同物理写入范围。下一步修改应依据实际 CacheUpdate 的独立输出／原位契约，保留整缓存逐位门槛；不要通过预填输出或忽略历史区来制造通过。
