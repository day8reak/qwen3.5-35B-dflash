# Target 单模块候选修复：AIR → OMC → 上板 → 精度 CSV

本轮只测试用户提出的 **QLinear FP16 scale 适配、CacheUpdate 整数索引**。覆盖已有 L0–L3 capture 的全部对应模块及 Prefill／Decode：38 个 QLinear＋4 个 Cache 边界各导出浮点／整数版本，共 **46 个候选／对照小图**。保持原模型、已安装 converter 文件、共享 DDK 和 golden 不变。

2026-11-12 回传：**AIR 42 FAILED／4 EXPORTED；OMC 42 NOT_EXPORTED／2 COMPILED／2 FAILED**。38 个 QLinear 和 4 个 Prefill Cache 在 AIR 阶段失败；Decode Cache 浮点版 K/V 编译通过，整数版 K/V 在 OMG 失败。仅一个 QLinear 已提供具体 traceback，其前端拒绝 FP16 per-token scale；其余失败原因须按各自日志确认。当前修正版已通过主机回归，尚未回传修正版 AIR／OMC／数值。

QLinear 实际转换 scale 常量为 FP16，保留 `[K/256,N]` 方向。**前端 per-token 保持原 FP32 `[M,1]`，在 GE converter 边界才执行 Reshape＋FP16 Cast。** 适配器仅在当前导出进程生效，继续调用已安装 converter，结束后恢复注册；不绕过 FakeTensor 检查。导出后核对 V4444 四个输入端口和实际 Cast 连线，保存 scale 转换及 packed W4 信息。FP16 舍入计入候选误差，golden 不重算。主机回归使用 CPU FakeTensor 与模拟 GE 接口，不代表真实 TorchAir／OMG 已通过。

AIR／OMC 单项失败记录后继续；上板自动选择全部 COMPILED 项。已有 146 项指标仍是修改前结果，见[精度表](target-accuracy-status.md)。

**旧 Cache 小图直接输入 INT32 块号／偏移，没有 `update()` 的浮点计算。** 本轮扩展为“position／block table → 索引计算 → CacheUpdate”。旧 capture 未保存高层 position：仅在同一次 Attention 页表能唯一反推位置时重用，并标记为重建输入；缺失／歧义直接报错。浮点控制组先核对原 CPro 表达式，整数组使用 `//`、`%` 和 INT32 页表。两者共用原 cache、updates、golden，CSV 分别标记 `_index_float`／`_index_integer`。

短上下文的主机计算中两套公式相同，因此不能仅凭整数改写认定旧 Decode 故障已解决。仍需检查整缓存，尤其未写历史区；本轮不验证 RefData alias 或连续状态反馈。

| 阶段 | 执行位置 | 输入 → 输出 |
|---|---|---|
| AIR | 原来可导出 CPro 算子的 TorchAir 环境；golden 在权重服务器 | capture＋原 Target 源码／converter＋原权重 YAML → `air/` |
| OMC | 35 服务器的 custom_pypto／所选 DDK | 整个 `air/` → `omc/` |
| 上板／回收 | 原来已配置 HDC 的主机；若就是 35，无需搬运 | runtime.json＋AIR／OMC → fixture＋native 输出 |
| 比较 | 回收主机的模型 Python 环境 | fixture＋native → report.json／summary.csv |

## 1. 各主机初始化

进入**当前主机**的仓库，激活此前成功的 Python／vendor 环境后执行。35 上已知仓库为 `/home/w00949577/qwen3.5-35B-dflash`，默认运行根目录即其旁边的 `qwen3.5-35B-dflash-run`。权重服务器使用自己的实际目录。

```bash
export REPO_DIR="$(git rev-parse --show-toplevel)"
git -C "$REPO_DIR" pull --ff-only origin main
export AI_RUN_DIR="${AI_RUN_DIR:-${REPO_DIR}-run}"
export BATCH_ROOT="$AI_RUN_DIR/target/tests/qlinear-cache-candidates"
export BATCH="$BATCH_ROOT/air"
export OMC_BATCH="$BATCH_ROOT/omc"
export COMPARISON="$BATCH_ROOT/comparison"
export RUNTIME_CONFIG="${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}"
export EXECUTION="$AI_RUN_DIR/fused/target-qlinear-cache-execution"
export RECOVERY="$AI_RUN_DIR/fused/target-qlinear-cache-recovery"
export PYTHONDONTWRITEBYTECODE=1
```

跨机器传输：AIR 导出后复制**整个 `BATCH` 目录**到 35 上同一角色路径；包括清单、schema、输入、golden 和权重 sidecar。若上板在另一主机，再把整个 `BATCH_ROOT` 传过去，保持 `air/` 与 `omc/` 相对位置；重新执行本节设置该主机路径。运行阶段不再需要原 checkpoint／capture。

本次使用固定测试目录；AIR、OMC 和 comparison 输出必须不存在。复测前先完成原请求回收、提取下述失败原因，再处理该测试的旧临时产物并复用原目录。编译失败不覆盖公共可运行模型。

本轮 Cache 尚缺具体错误。在 AIR 主机读取 Prefill 失败 receipt，在 35 主机读取 Decode 整数版 OMG 日志；不要只重复打印 manifest 的 worker 退出码：

```bash
python - "$BATCH" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for mode in ("float", "integer"):
    case = "l03_c00_cache_k_index_" + mode + "_o0"
    path = root / "models" / case / "export.json"
    print(case, json.loads(path.read_text()).get("error") if path.exists()
          else "receipt 缺失，查看 " + str(root / (case + ".log")))
PY
tail -n 120 "$OMC_BATCH/models/l03_c01_cache_k_index_integer_o0/omc/compile.log"
```

## 2. 两类模块全部重新导出

`GOLDEN_CAPTURE` 沿用已成功采集的目录：其中 `manifest.json` 为 `CAPTURED`，并有 regions／weights。`TARGET_SOURCE` 是原 Target 源码根目录，包含 models 和下述 converter。`TARGET_CONFIG` 是原 Target 导出使用的 YAML，其中 `model_file` 指向 HF 权重目录、`quanted_pth` 指向 W4 权重目录；相对路径按 `TARGET_SOURCE` 解析。这三个变量使用权重服务器上的真实值，不能填写示意目录。

默认导出行为仍沿原 CPro 源码。本轮显式开启下面两个候选开关；其他模块不参与此次导出。已有权重会按原替换工具重建并与 capture 核对，然后才做候选 scale 转换。

```bash
: "${GOLDEN_CAPTURE:?请设置已有 CAPTURED 捕获目录}"
: "${TARGET_SOURCE:?请设置原 Target 源码根目录}"
: "${TARGET_CONFIG:?请设置原 Target 的权重 YAML 路径}"
export CPRO_EXPORT_SCRIPT="${CPRO_EXPORT_SCRIPT:-$TARGET_SOURCE/export_glm_model_qwen3_5_moe.py}"

(
set -e
test -f "$GOLDEN_CAPTURE/manifest.json"
test -f "$CPRO_EXPORT_SCRIPT"
test -f "$TARGET_CONFIG"
test ! -e "$BATCH"
export_rc=0
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" export \
  --capture "$GOLDEN_CAPTURE" \
  --source-root "$TARGET_SOURCE" \
  --converter-script "$CPRO_EXPORT_SCRIPT" \
  --target-config "$TARGET_CONFIG" \
  --kinds qlinear cache \
  --qlinear-scales fp16 --cache-indexing both \
  --output "$BATCH" || export_rc=$?

python - "$BATCH/manifest.json" "$GOLDEN_CAPTURE/manifest.json" "$export_rc" <<'PY'
import json, sys
from collections import Counter
r, capture = [json.load(open(p)) for p in sys.argv[1:3]]
counts = Counter(c["status"] for c in r["cases"])
expected = sum(len(c["outputs"]) * (2 if c["kind"] == "cache" else 1)
               for c in capture["regions"] if c["kind"] in ("qlinear", "cache"))
print("AIR:", r["status"], dict(counts), "expected:", expected, "exit:", sys.argv[3])
if (int(sys.argv[3]) not in (0, 1) or r["status"] not in ("EXPORTED", "PARTIAL")
        or len(r["cases"]) != expected or not counts["EXPORTED"]):
    raise SystemExit("AIR 未完成全批遍历或无成功项；先检查导出日志。")
PY
)
```

只有摘要检查成功才继续。PARTIAL 的退出码 1 在此被显式接收，仍核对完成态及全量 case 数；不会把中断当作完成。每个失败 case 的日志为 `$BATCH/<case>.log`；新版 manifest 优先保留 `export.json` 的实际异常，并单列 worker 错误。每份 `models/<case>/export.json` 的 `candidate` 记录改动及输入来源，QLinear 的 `ge_adapter` 记录所选 converter、适配次数与恢复状态；`candidate_graph_check` 检查源 AIR 端口。这些不代替 OMC 实际存储检查。原权重与 capture 不一致会明确失败，golden 保持原样。

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

判读：QLinear 比较 38 项相同 golden 的 relative-L2；Cache 比较同边界浮点／整数两行，并以整缓存逐位相同为通过条件。只有整数版改善、浮点版仍异常，才能支持索引路径相关；若两版都异常，继续定位写入／保留区与实际绑定。

主机测试覆盖候选 dtype、权重字节不变、索引公式、原源码匹配和批次遍历；未执行真实 TorchAir／OMG／设备任务。
