# Target Cache 候选修复：AIR → OMC → 上板 → 精度 CSV

当前仅保留**全部 Cache 与状态明确为 FAILED 的项**继续定位。上一轮 46 项中的非 Cache 项没有 FAILED；L0 out_proj 两项虽超过 1%，状态仍为 METRICS_RECORDED，按用户确认不再复跑。因此本页现在导出 **8 个 Cache 图**：K/V × Prefill/Decode × 浮点/整数。原 38 个 QLinear 从本轮构建与上板列表移除；以下 42 项数值仅为历史结果，不代表新增运行。

2026-11-12 上一轮 OMC／数值回传：**42 COMPILED／4 FAILED**。38 个 QLinear 和 Prefill／Decode 的 4 个 Cache 浮点控制图均已编译并回传数值；4 个 Cache 整数图编译失败。QLinear 有 **36/38 relative-L2 <1%**，仅 L0 out_proj 的 Prefill／Decode 为 1.3766%／1.7625%；Cache 浮点控制组 Prefill K/V 零误差，Decode 仍为 99.81%／99.69%。[原始候选 CSV](../specs/target-qlinear-cache-candidate-metrics-20261112.csv)保留 METRICS_RECORDED／PASSED／FAILED。

乘减求余后的最新 AIR 为 **42 EXPORTED／4 FAILED**：38 个 QLinear 与 4 个浮点 Cache 均已导出，四个整数 Cache 在 `candidate_graph_check` 被拦下。回传 pbtxt 显示 `Mul` 两输入 INT32，输出描述 FP32，之后 `Cast_1` 转回 INT32；不能把这项描述冲突当成设备已经进行浮点计算的证明。此前 Decode V 的 OMG 日志已通过 Gather／FloorDiv，拒绝 INT32 FloorMod，因此仍保留 `offset = position[0] - block * block_size`。新适配器仅在整数 Cache 导出时将这一 INT32×32 操作直接生成 GE Const／Mul，并先补齐乘法输出元数据；不再经过会对未完成元数据补 Cast 的转换分支。类型检查继续生效，实际新 AIR／OMC／数值待回传。

AIR／OMC 单项失败记录后继续；上板自动选择全部 COMPILED 项。已有 146 项指标仍是修改前结果，见[精度表](target-accuracy-status.md)。

**旧 Cache 小图直接输入 INT32 块号／偏移，没有 `update()` 的浮点计算。** 本轮扩展为“position／block table → 索引计算 → CacheUpdate”。旧 capture 未保存高层 position：仅在同一次 Attention 页表能唯一反推位置时重用，并标记为重建输入；缺失／歧义直接报错。浮点控制组先核对原 CPro 表达式，整数组使用 INT32 `//`、乘减求余和 INT32 页表。两者共用原 cache、updates、golden，CSV 分别标记 `_index_float`／`_index_integer`。

Cache 重建保留捕获的索引数量：`[T]` 仍为 `[T]`，多行 updates 配 `[1]` 时仍只传 `[1]`，不自动广播或生成逐行位置。`candidate.index_contract` 记录更新行数、原始／回放索引 shape；整数公式须逐值恢复捕获索引。该回放不解释单索引在内核中的写入跨度，也不恢复未捕获的高层完整 position。

整数候选保持外部 position 为 INT64，在已检查的 `0..INT32_MAX` 范围内先 Cast 到 INT32，再执行 `position[0]`、`//`、乘减求余和页表 Gather。仅在除法后 `.to(int32)` 无法修复更早的 Gather 数据类型。AIR 检查记录每个索引 Gather 的数据输入类型及 Mul 两端的 INT32 契约，发现 INT64 Gather、FloorMod 或浮点描述仍会拒绝；不限制轴常量／索引参数必须同为 INT32。浮点控制组保持原表达式。实际 DDK 编译和缓存精度仍需验证。

短上下文的主机计算中两套公式相同，因此不能仅凭整数改写认定旧 Decode 故障已解决。仍需检查整缓存，尤其未写历史区；本轮不验证 RefData alias 或连续状态反馈。

| 阶段 | 执行位置 | 输入 → 输出 |
|---|---|---|
| AIR | 原来可导出 CPro 算子的 TorchAir 环境；golden 在权重服务器 | capture＋原 Target 源码／converter → `air/` |
| OMC | 35 服务器的 custom_pypto／所选 DDK | 整个 `air/` → `omc/` |
| 上板／回收 | 原来已配置 HDC 的主机；若就是 35，无需搬运 | runtime.json＋AIR／OMC → fixture＋native 输出 |
| 比较 | 回收主机的模型 Python 环境 | fixture＋native → report.json／summary.csv |

## 1. 各主机初始化

进入**当前主机**的仓库，激活此前成功的 Python／vendor 环境后执行。35 上已知仓库为 `/home/w00949577/qwen3.5-35B-dflash`，默认运行根目录即其旁边的 `qwen3.5-35B-dflash-run`。权重服务器使用自己的实际目录。

```bash
export REPO_DIR="$(git rev-parse --show-toplevel)"
git -C "$REPO_DIR" pull --ff-only origin main
export AI_RUN_DIR="${AI_RUN_DIR:-${REPO_DIR}-run}"
export GOLDEN_CAPTURE="${GOLDEN_CAPTURE:-${CAPTURE:-$AI_RUN_DIR/target-layer-accuracy/golden}}"
export BATCH_ROOT="$AI_RUN_DIR/target/tests/cache-candidates"
export BATCH="$BATCH_ROOT/air"
export OMC_BATCH="$BATCH_ROOT/omc"
export COMPARISON="$BATCH_ROOT/comparison"
export RUNTIME_CONFIG="${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}"
export EXECUTION="$AI_RUN_DIR/fused/target-cache-execution"
export RECOVERY="$AI_RUN_DIR/fused/target-cache-recovery"
export PYTHONDONTWRITEBYTECODE=1
```

跨机器传输：AIR 导出后复制**整个 `BATCH` 目录**到 35 上同一角色路径；包括清单、schema、输入、golden 和权重 sidecar。若上板在另一主机，再把整个 `BATCH_ROOT` 传过去，保持 `air/` 与 `omc/` 相对位置；重新执行本节设置该主机路径。运行阶段不再需要原 checkpoint／capture。

本次使用固定测试目录；AIR、OMC 和 comparison 输出必须不存在。复测前先完成原请求回收、提取下述失败原因，再处理该测试的旧临时产物并复用原目录。编译失败不覆盖公共可运行模型。

新修复需要从第 2 节重新生成全部 8 个 Cache 图，再运行第 3 节全部 OMC 编译；直接重编旧 AIR 不会修复其中的 Mul／Cast 描述冲突。确认 Prefill 阻塞时，在**原采集／AIR 主机**读取 capture 元数据。上面的 `GOLDEN_CAPTURE` 优先沿用已有值或 `CAPTURE`，否则使用本项目原采集命令的默认目录；若实际采集在别处，设置其真实目录。只搬运了 AIR 的 35 主机通常没有 capture，无需为编译复制整套 capture。

```bash
python - "$GOLDEN_CAPTURE/manifest.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"未找到 {path}；请在原采集/AIR 主机设置 GOLDEN_CAPTURE 后运行。")
capture = json.loads(path.read_text())
if capture.get("status") != "CAPTURED":
    raise SystemExit(f"不是已完成的 capture: {path}")
print("capture:", path)
found = 0
for region in capture["regions"]:
    if region["id"] in ("l03_c00_cache_k", "l03_c00_cache_v"):
        found += 1
        print(region["id"])
        print(json.dumps([{k: t[k] for k in ("name", "shape", "dtype")}
                          for t in region["inputs"]], indent=2))
if found != 2:
    raise SystemExit("此 capture 未包含预期的两个 Prefill Cache 边界。")
PY
```

## 2. 全部 Cache 模块重新导出

`GOLDEN_CAPTURE` 沿用已成功采集的目录，`manifest.json` 为 `CAPTURED`；Cache 回放需要同一次 Attention 的页表。`TARGET_SOURCE` 是原 Target 源码根目录，包含 models 和下述 converter。Cache 边界没有权重，本轮不需要 QLinear 的权重 YAML。两个目录使用原采集主机上的真实路径。

用现有 `--kinds cache` 筛选，配合 `--cache-indexing both` 同时导出浮点控制和整数候选；所有阶段只处理这份 8 项清单。

```bash
: "${GOLDEN_CAPTURE:?请设置已有 CAPTURED 捕获目录}"
: "${TARGET_SOURCE:?请设置原 Target 源码根目录}"
export CPRO_EXPORT_SCRIPT="${CPRO_EXPORT_SCRIPT:-$TARGET_SOURCE/export_glm_model_qwen3_5_moe.py}"

(
set -e
test -f "$GOLDEN_CAPTURE/manifest.json"
test -f "$CPRO_EXPORT_SCRIPT"
test ! -e "$BATCH"
export_rc=0
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" export \
  --capture "$GOLDEN_CAPTURE" \
  --source-root "$TARGET_SOURCE" \
  --converter-script "$CPRO_EXPORT_SCRIPT" \
  --kinds cache --cache-indexing both \
  --output "$BATCH" || export_rc=$?

python - "$BATCH/manifest.json" "$GOLDEN_CAPTURE/manifest.json" "$export_rc" <<'PY'
import json, sys
from collections import Counter
r, capture = [json.load(open(p)) for p in sys.argv[1:3]]
counts = Counter(c["status"] for c in r["cases"])
expected = sum(len(c["outputs"]) * 2
               for c in capture["regions"] if c["kind"] == "cache")
print("AIR:", r["status"], dict(counts), "expected:", expected, "exit:", sys.argv[3])
if (int(sys.argv[3]) not in (0, 1) or r["status"] not in ("EXPORTED", "PARTIAL")
        or expected != 8 or len(r["cases"]) != expected
        or any(c["kind"] != "cache" for c in r["cases"]) or not counts["EXPORTED"]):
    raise SystemExit("AIR 未完成全批遍历或无成功项；先检查导出日志。")
PY
)
```

只有摘要检查成功才继续。PARTIAL 的退出码 1 在此被显式接收，仍核对完成态及全量 case 数；不会把中断当作完成。每个失败 case 的日志为 `$BATCH/<case>.log`；新版 manifest 优先保留 `export.json` 的实际异常，并单列 worker 错误。每份 `models/<case>/export.json` 的 `candidate` 记录改动及输入来源，整数 Cache 的 `ge_adapter` 记录所选 converter、适配次数与恢复状态；整数 Cache 应为 `calls=1`、`restored=true`，`multiply.dtype=int32`。`candidate_graph_check.cache_indexing` 应保留空 `floating_edges`，并在 `integer_multiply` 记录 INT32 输出与两条输入边。这些检查针对源 AIR 端口，不代替 OMC 实际存储检查；golden 保持原样。

## 3. 本轮全部 OMC 编译（35 服务器）

下面默认值来自本次已使用的 DDK；如已更换 DDK，设置其真实根目录和版本。使用 `isolated-kirinx90`、内嵌权重和原批次的 `use_origin_format=true`。本套 DDK 需传入此前已完成的 `TBE_DIAGNOSIS`：整理后的默认位置为 `common/tbe-diagnosis`，实际仍在原目录时保留该变量的真实值。工具只在本次编译目录准备适配，不改共享 DDK／Python 安装，OMG 目标仍为 `kirinx90`。仅 export 该变量不够，命令必须传 `--tbe-diagnosis`。

```bash
export CPRO_DDK_ROOT="${CPRO_DDK_ROOT:-/home/l00886453/ddk_0806/image/hiai_ddk/ddk_internal}"
export CPRO_DDK_ID="${CPRO_DDK_ID:-100.520.020.010}"
export TBE_DIAGNOSIS="${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}"

(
set -e
test -x "$CPRO_DDK_ROOT/tools/tools_omg/master/omg"
test -f "$TBE_DIAGNOSIS/report.json"
test -f "$BATCH/manifest.json"
test ! -e "$OMC_BATCH"
compile_rc=0
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" compile \
  --batch "$BATCH" \
  --output "$OMC_BATCH" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --ddk-id "$CPRO_DDK_ID" \
  --compiler-profile isolated-kirinx90 \
  --tbe-diagnosis "$TBE_DIAGNOSIS" \
  --use-origin-format || compile_rc=$?

python - "$OMC_BATCH/compile.json" "$compile_rc" <<'PY'
import json, sys
from collections import Counter
r = json.load(open(sys.argv[1]))
counts = Counter(c["status"] for c in r["cases"].values())
print("OMC:", r["status"], dict(counts), "exit:", sys.argv[2])
if (int(sys.argv[2]) not in (0, 1) or r["status"] not in ("COMPILED", "PARTIAL")
        or r["coverage"] != "ALL_MANIFEST_CASES"
        or set(r["cases"]) != set(r["selected_cases"]) or not counts["COMPILED"]):
    raise SystemExit("OMC 未完成全批遍历或无成功项；先检查 compile.json。")
print("可继续执行第 4 节上板命令，自动选择全部 COMPILED 项；FAILED / NOT_EXPORTED 保留。")
PY
)
```

单项日志：`$OMC_BATCH/models/<case>/omc/compile.log`；worker 适配结果在同目录 `report.json` 的 `tbe_compatibility.audit`。批次新增 `tbe_compatibility_requested` 仅表示请求了适配，不表示 worker 已执行适配或编译通过。失败需各自日志确认，不能仅凭退出码归因。编译部分失败不阻塞成功项上板；末行只是提示，执行下节命令才会启动设备。

## 4. 批量上板

在原 HDC 主机加载已有私有配置；没有配置文件时保留原终端中的 HDC／BIP／SN。BIP 是 HDC 服务端 IP:端口，SN 是设备序列号，均沿用此前成功值。若加载配置改变了 REPO_DIR／AI_RUN_DIR，重新执行第 1 节。

```bash
if test -f "${QWEN35_ENV_CONFIG:-$REPO_DIR/../qwen35-cpro.env}"; then
  source "$REPO_DIR/targets/cpro/scripts/env.sh"
fi
export HDC="${HDC:-hdc}"
: "${BIP:?沿用原 HDC 服务端 IP:端口}" "${SN:?沿用原设备序列号}"
: "${CPRO_DDK_ROOT:?上板主机实际 DDK 根目录}"
test -f "$RUNTIME_CONFIG"
```

沿用 runtime.json 与其中的 native_directory。**旧 runner 若仍是 16 MiB，先执行下面一次条件更新**；此前已成功更新为 32 MiB 且支持 INT16／输入复位时会直接复用。构建使用此前成功的 OHOS Native SDK／HIAI 环境，只重建运行器，不重编模型：

```bash
python - "$REPO_DIR" "$RUNTIME_CONFIG" <<'PY'
import json, subprocess, sys
from pathlib import Path
repo, config = map(Path, sys.argv[1:])
config = config.resolve()
cfg = json.loads(config.read_text())
native = (config.parent / cfg.get("native_directory", "native")).resolve()
built = json.loads((native / "result.json").read_text())
def ready(r):
    return (r.get("status") == "LINKED"
            and r.get("tensor_probe_limits", {}).get("max_io_bytes", 0) >= 32 << 20
            and "int16" in r.get("tensor_probe_dtypes", [])
            and "copy-each-call" in r.get("tensor_probe_input_reset", []))
if not ready(built):
    previous = json.loads((native / "command.json").read_text())
    cmd = [sys.executable, str(repo / "targets/cpro/scripts/build_hiai.py"),
           "--output", str(native), "--replace"]
    api = previous.get("compatible_sdk_version_requested")
    if api is not None:
        cmd += ["--compatible-sdk-version", str(api)]
    subprocess.run(cmd, check=True)
    built = json.loads((native / "result.json").read_text())
if not ready(built):
    raise SystemExit("runner 尚未具备本批所需能力。")
print("runner:", native, built["tensor_probe_limits"])
PY
```

上面检查成功后，启动一次：

```bash
python "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --compiled "$OMC_BATCH" \
  --config "$RUNTIME_CONFIG" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$EXECUTION" \
  --apply
```

不加 `--cases`，自动运行全部 COMPILED 项；每图预热 3 次、采样 10 次，每次恢复捕获输入。文件名虽然带 draft，入口已支持 Target。返回 `VECTOR_VALIDATION_STARTED` 表示已启动，随后回收同一个请求；中断或 PENDING 时也不重复 `--apply`。若某张图**板端运行**失败，当前运行器会停止后续图，已完成输出仍可回收；这与允许编译部分失败不同。

## 5. 回收与全批精度比较

```bash
python "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --recover "$EXECUTION" --output "$RECOVERY"
```

若返回 `VECTOR_VALIDATION_PENDING`，稍后**重复同一回收命令**，不重新启动。这里的 recovery 位于支持安全复用的 `fused/` 固定槽位。回收可能因数值异常返回非零，但只要已经取得 native 输出，仍执行下面的完整 CSV 比较：

```bash
(
set -e
test -d "$RECOVERY/native"
test ! -e "$COMPARISON"
compare_rc=0
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" compare \
  --fixture "$EXECUTION/fixture" \
  --native "$RECOVERY/native" \
  --output "$COMPARISON" || compare_rc=$?

python - "$COMPARISON/report.json" "$compare_rc" <<'PY'
import json, sys
from collections import Counter
r = json.load(open(sys.argv[1]))
print("comparison:", r["status"], "exit:", sys.argv[2])
print("cases:", dict(Counter(c["status"] for c in r["cases"])))
print("uncompiled:", len(r["uncompiled_cases"]))
print("coverage_complete:", r["coverage_complete"])
if int(sys.argv[2]) not in (0, 1):
    raise SystemExit("比较程序异常退出，请检查报告。")
PY
)
```

默认浮点记录 `METRICS_RECORDED`，不自动判为通过；已有冻结容差时才同时加 `--atol`／`--rtol`。CacheUpdate 对整缓存逐位比较。编译缺项保存在 report.json 的 uncompiled_cases，CSV 包含已准备执行的 case；未执行／输出缺失会报 FAILED。CSV 的 relative_l2 是比值，**0.01 = 1%**；详细 JSON 记录全部 10 次调用，CSV 取首次调用指标及总体重复稳定性。

最终只需查看／回传：

| 文件 | 内容 |
|---|---|
| `$OMC_BATCH/compile.json` | 全部编译状态 |
| `$RECOVERY/result.json` | 执行／回收是否完成 |
| `$COMPARISON/summary.csv` | 按层、阶段、模块的误差表 |
| `$COMPARISON/report.json` | 全部调用指标、dtype、稳定性与覆盖缺项 |

判读：Cache 比较同边界浮点／整数两行，并以整缓存逐位相同为通过条件。只有整数版改善、浮点版仍异常，才能支持索引路径相关；若两版都异常，继续定位写入／保留区与实际绑定。

主机测试覆盖索引公式、Mul 输出元数据、原源码匹配与 Cache 批次筛选；未执行真实 TorchAir／OMG／设备任务。
