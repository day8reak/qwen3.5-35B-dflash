# Target 分模块全批：AIR → OMC → 上板 → 精度 CSV

本页按现有完成态 capture 导出**全部边界、全部输出**，不加模块筛选。当前 capture 为 L0–L3、Prefill T256／Decode T1，共 158 个 case；尚不包含 MoE 内部、LM head 或全部 40 层。每个小图独立使用捕获输入，不能据此验证整图误差传播或连续状态反馈。

AIR／OMC 单项失败会记录后继续；只要有成功 OMC 就可上板。ConvState 失败可以保留，最终覆盖仍显示不完整。最新结果和高危区域见[精度表](target-accuracy-status.md)。

| 阶段 | 执行位置 | 输入 → 输出 |
|---|---|---|
| AIR | 原来可导出 CPro 算子的 TorchAir 环境；golden 在权重服务器 | 已完成 capture＋原 Target 源码／converter → `air/` |
| OMC | 35 服务器的 custom_pypto／所选 DDK | 整个 `air/` → `omc/` |
| 上板／回收 | 原来已配置 HDC 的主机；若就是 35，无需搬运 | runtime.json＋AIR／OMC → fixture＋native 输出 |
| 比较 | 回收主机的模型 Python 环境 | fixture＋native → report.json／summary.csv |

## 1. 各主机初始化

进入**当前主机**的仓库，激活此前成功的 Python／vendor 环境后执行。35 上已知仓库为 `/home/w00949577/qwen3.5-35B-dflash`，默认运行根目录即其旁边的 `qwen3.5-35B-dflash-run`。权重服务器使用自己的实际目录。

```bash
export REPO_DIR="$(git rev-parse --show-toplevel)"
git -C "$REPO_DIR" pull --ff-only origin main
export AI_RUN_DIR="${AI_RUN_DIR:-${REPO_DIR}-run}"
export BATCH_ROOT="$AI_RUN_DIR/target/tests/all-boundaries"
export BATCH="$BATCH_ROOT/air"
export OMC_BATCH="$BATCH_ROOT/omc"
export COMPARISON="$BATCH_ROOT/comparison"
export RUNTIME_CONFIG="${RUNTIME_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}"
export EXECUTION="$AI_RUN_DIR/fused/target-boundary-execution"
export RECOVERY="$AI_RUN_DIR/fused/target-boundary-recovery"
export PYTHONDONTWRITEBYTECODE=1
```

跨机器传输：AIR 导出后复制**整个 `BATCH` 目录**到 35 上同一角色路径；包括清单、schema、输入、golden 和权重 sidecar。若上板在另一主机，再把整个 `BATCH_ROOT` 传过去，保持 `air/` 与 `omc/` 相对位置；重新执行本节设置该主机路径。运行阶段不再需要原 checkpoint／capture。

本次使用固定测试目录；AIR、OMC 和 comparison 输出必须不存在。复测前先完成原请求回收，再处理旧测试产物。编译失败不覆盖公共可运行模型。

## 2. 全部 AIR 导出

`GOLDEN_CAPTURE` 沿用已成功采集的目录：其中 `manifest.json` 为 `CAPTURED`，并有 regions／weights；它不是原始权重、AIR 或 OMC 目录。`TARGET_SOURCE` 是原 Target 源码根目录，包含 models 和下述 converter。两个实际路径尚未在会话提供，下面要求保留／填写已有真实值，不猜路径。

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
  --output "$BATCH" || export_rc=$?

python - "$BATCH/manifest.json" "$GOLDEN_CAPTURE/manifest.json" "$export_rc" <<'PY'
import json, sys
from collections import Counter
r, capture = [json.load(open(p)) for p in sys.argv[1:3]]
counts = Counter(c["status"] for c in r["cases"])
expected = sum(len(c["outputs"]) for c in capture["regions"])
print("AIR:", r["status"], dict(counts), "expected:", expected, "exit:", sys.argv[3])
if (int(sys.argv[3]) not in (0, 1) or r["status"] not in ("EXPORTED", "PARTIAL")
        or len(r["cases"]) != expected or not counts["EXPORTED"]):
    raise SystemExit("AIR 未完成全批遍历或无成功项；先检查导出日志。")
PY
)
```

只有摘要检查成功才继续。PARTIAL 的退出码 1 在此被显式接收，仍核对完成态及全量 case 数；不会把中断当作完成。每个失败 case 的日志为 `$BATCH/<case>.log`。当前入口已包含 Norm gamma 预计算和 Conv 显式四维导出修正。

## 3. 全部 OMC 编译（35 服务器）

下面默认值来自本次已使用的 DDK；如已更换 DDK，设置其真实根目录和版本。使用 `isolated-kirinx90`、内嵌权重和原批次的 `use_origin_format=true`。工具会构造编译子进程环境，实际命令保存在各 case 中。

```bash
export CPRO_DDK_ROOT="${CPRO_DDK_ROOT:-/home/l00886453/ddk_0806/image/hiai_ddk/ddk_internal}"
export CPRO_DDK_ID="${CPRO_DDK_ID:-100.520.020.010}"

(
set -e
test -x "$CPRO_DDK_ROOT/tools/tools_omg/master/omg"
test -f "$BATCH/manifest.json"
test ! -e "$OMC_BATCH"
compile_rc=0
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" compile \
  --batch "$BATCH" \
  --output "$OMC_BATCH" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --ddk-id "$CPRO_DDK_ID" \
  --compiler-profile isolated-kirinx90 \
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
print("下一步自动运行全部 COMPILED 项；FAILED / NOT_EXPORTED 保留。")
PY
)
```

单项日志：`$OMC_BATCH/models/<case>/omc/compile.log`。当前已知 ConvState 的格式开关对照仍失败，本批允许它继续失败；不改写报告，也不改用仅诊断的 `sourced-wrapper-default`。

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

命令参数及语法已在主机核对；本页更新没有重新执行服务器导出、OMG 或设备任务。
