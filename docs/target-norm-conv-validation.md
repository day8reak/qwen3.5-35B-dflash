# 新增 26 个 Norm／Conv OMC：上板、回收与比较

用户已确认普通 Norm 20 项、Conv1d 6 项全部编译成功。本页只执行这些新模型，数值结果仍待回传。沿用之前 120 项的 runtime.json 和 HDC 配置，native runner 需按下文更新一次。

用户回传的 `selected boundary exceeds existing small-probe IO budget` 发生在主机 prepare，设备尚未执行。旧主机脚本和 native runner 均限制**单个模型的输入＋输出总计 16 MiB**。按 Prefill Conv 的 FP32 输入 `[1,8192,259]`、输出 `[1,8192,256]` 计算，总计 **16,875,520 字节，超出 96 KiB**，符合这次触发条件；旧错误未记录具体 case，不能据此确认其名字。现将两端限额同步设为 **32 MiB**，并在上传前检查已构建 runner 的实际声明。每方向最多 8 个张量、rank 上限 4 均保持。

## 1. 在原上板主机恢复变量

在之前成功运行 120 项的主机执行；若该主机就是 35，则在 35 操作。进入本机 GitHub 仓库，使用此前已跑通的 Python／DDK 环境：

```bash
export REPO_DIR="$(git rev-parse --show-toplevel)"
git -C "$REPO_DIR" pull --ff-only origin main

# 有既存私有配置时，恢复上次成功运行使用的变量。
if test -f "${QWEN35_ENV_CONFIG:-$REPO_DIR/../qwen35-cpro.env}"; then
  source "$REPO_DIR/targets/cpro/scripts/env.sh"
fi

export AI_RUN_DIR="${AI_RUN_DIR:-${REPO_DIR}-run}"
export CPRO_DDK_ROOT="${CPRO_DDK_ROOT:-${DDK_PATH:-}}"
export HDC="${HDC:-hdc}"

export RECHECK="$AI_RUN_DIR/target/tests/environment-recheck-001"
export RUNTIME_CONFIG="$AI_RUN_DIR/runtime/runtime.json"
export VALIDATION_ROOT="$RECHECK/validation-norm-conv-002"

: "${AI_RUN_DIR:?}" "${CPRO_DDK_ROOT:?}" "${HDC:?}" "${BIP:?}" "${SN:?}"
test -f "$RUNTIME_CONFIG"
```

`env.sh` 默认读取仓库旁的 `qwen35-cpro.env`，也支持已有的 `QWEN35_ENV_CONFIG`。若此前用其他配置文件，可先执行 `source "$REPO_DIR/targets/cpro/scripts/env.sh" /该配置的实际路径`。没有私有配置时，上方保留本终端已有的 `AI_RUN_DIR`、DDK、HDC、`BIP`、`SN`；BIP／SN 未提供就停止检查，使用上次成功的实际值，不猜设备地址。`RUNTIME_CONFIG` 若不在示例位置，改为之前成功的那个 runtime.json。

`VALIDATION_ROOT` 是本次新的验证目录；已有同名运行时继续递增编号。只更新并重建 runner；OMC、输入、golden、shape 和 dtype 不变，不重新采集或导出／编译模型。

需要跨主机搬运时，将新 AIR batch 和 OMC batch 一起搬运并保持相对位置。OMC 的 `compile.json.source_batch` 指向 AIR，AIR 内含本次输入和 golden。运行阶段不再需要原始 `GOLDEN_CAPTURE` 或完整 checkpoint。

### 更新这份 runtime 配置实际使用的 runner

在上板主机使用上次构建成功的 `OHOS_NATIVE_SDK`、`HIAI_INCLUDE_DIR`、`HIAI_LIBRARIES`，或同一份 `QWEN35_RESOURCES`。若 SDK 只在原构建主机，则在那里重建后搬运完整 native 目录。已有设备任务先完成回收；`--replace` 用于重建已归属的 native 目录，构建器会检查活动任务。本次 prepare 失败且 `device_execution=NOT_RUN`、`lock_state=NO_KNOWN_OWNED_LOCK` 的请求无需回收或清锁。

```bash
python - "$REPO_DIR" "$RUNTIME_CONFIG" <<'PY'
import json, subprocess, sys
from pathlib import Path
repo, config = (Path(p).resolve() for p in sys.argv[1:])
cfg = json.loads(config.read_text())
native = (config.parent / cfg.get('native_directory', 'native')).resolve()
previous = json.loads((native / 'command.json').read_text())
cmd = [sys.executable, str(repo / 'targets/cpro/scripts/build_hiai.py'),
       '--output', str(native), '--replace']
# 保留上次显式选择的 SDK API，避免重建时改变兼容版本。
api = previous.get('compatible_sdk_version_requested')
if api is not None:
    cmd += ['--compatible-sdk-version', str(api)]
subprocess.run(cmd, check=True)
built = json.loads((native / 'result.json').read_text())
assert built['status'] == 'LINKED'
assert built['tensor_probe_limits']['max_io_bytes'] >= 32 << 20
print('runner 已更新:', native)
print('tensor_probe_limits:', built['tensor_probe_limits'])
PY
```

只有此步成功才继续下方执行。不能只改 `result.json` 声明；真正的 C++ 二进制也必须更新。主机上传前会校验该二进制哈希，并按 case 检查字节数是否超过其限额。

## 2. 选择实际编译批次

下面两种组织方式**选择一种**。`OMC_NORM_CONV` 填含 `compile.json` 的批次目录，不是单个 `model.omc`，也不是旧的 120 项批次。

**若沿用文档中的 4＋22 分批流程，先运行 4 项：**

```bash
export OMC_NORM_CONV="$RECHECK/omc-norm-conv-probe-002"
export VALIDATION_TAG="probe4"
export EXPECTED_CASES=4
```

这批执行、回收、比较完成后，再用以下变量重新执行第 3～6 节，运行剩余 22 项：

```bash
export OMC_NORM_CONV="$RECHECK/omc-norm-conv-rest-002"
export VALIDATION_TAG="rest22"
export EXPECTED_CASES=22
```

两个目录名来自此前的编译命令，若实际改过编号则按实际路径设置。逐批启动、回收，再启动下一批，避免板端任务重叠。

**若 26 项编译在同一个批次，使用这一组变量：**

```bash
export OMC_NORM_CONV="/填写本机包含26项成功结果的OMC批次目录"
export VALIDATION_TAG="all26"
export EXPECTED_CASES=26
```

单批目录的实际名字尚未回传，所以这里明确保留待填路径。使用此方式时不再运行上面的 probe4／rest22。

## 3. 检查所选批次并设置输出

这个检查只读本地清单，防止误选旧批次。检查通过后再上板：

```bash
python - "$OMC_NORM_CONV" "$EXPECTED_CASES" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
root = Path(sys.argv[1]).resolve()
r = json.loads((root / 'compile.json').read_text())
m = json.loads((root / 'manifest.json').read_text())
by_name = {c['id']: c for c in m['cases']}
selected = [name for name, row in r['cases'].items() if row['status'] == 'COMPILED']
assert len(selected) == int(sys.argv[2]), f'成功项数与预期不同: {len(selected)}'
assert all(by_name[n]['kind'] in ('norm', 'conv') for n in selected), '混入了其他模块，请检查批次'
source = (root / r['source_batch']).resolve()
assert (source / 'manifest.json').is_file(), f'缺少关联 AIR 批次: {source}'
print('OMC batch:', root)
print('AIR batch:', source)
print('本次将运行:', len(selected), dict(Counter(by_name[n]['kind'] for n in selected)))
PY

export EXECUTION="$VALIDATION_ROOT/$VALIDATION_TAG-execution"
export RECOVERY="$VALIDATION_ROOT/$VALIDATION_TAG-recovery-001"
export COMPARISON="$VALIDATION_ROOT/$VALIDATION_TAG-comparison-001"
```

真正执行前，运行器还会核验模型／夹具哈希和编译后的输入输出接口。已有输出目录时先确认是否属于尚在执行的请求；新实验使用新的编号，已有请求走下面的回收步骤。

## 4. 启动当前批次一次

```bash
python "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --compiled "$OMC_NORM_CONV" \
  --config "$RUNTIME_CONFIG" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$EXECUTION" \
  --apply
```

入口自动选择这个批次的 COMPILED 项；4 项批次中的其他 NOT_SELECTED 项不会执行。每个模型 3 次预热、10 次采样，每次调用前恢复捕获输入。启动命令返回不一定代表板端已完成，下一步回收同一个请求。

同一 `EXECUTION` 只启动一次。终端中断或状态不明时，可先只读检查本地记录：

```bash
python "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" --status "$EXECUTION"
```

## 5. 回收当前请求

```bash
python "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --recover "$EXECUTION" \
  --output "$RECOVERY"
```

若返回 `VECTOR_VALIDATION_PENDING`，保持 `EXECUTION` 不变，稍后将回收目录改为新编号再执行同一回收命令：

```bash
export RECOVERY="$VALIDATION_ROOT/$VALIDATION_TAG-recovery-002"
python "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --recover "$EXECUTION" --output "$RECOVERY"
```

后续再次回收继续增加编号。PENDING 时继续等这次请求，不再次 `--apply`，也不启动第二批。执行完成且输出已取回后，`RECOVERY/native/` 才可用于下面的比较。若回收报告失败，先检查其 error 和执行完成情况；已回收的部分输出仍可分析，但不能算全批完成。

## 6. 批量比较当前批次

```bash
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" compare \
  --fixture "$EXECUTION/fixture" \
  --native "$RECOVERY/native" \
  --output "$COMPARISON"

cat "$COMPARISON/summary.csv"
```

`RECOVERY` 指向最后一次成功取回输出的目录。保持与之前 120 项相同的报告方式：没有已冻结的浮点门槛时记录 `METRICS_RECORDED`，不凭编译／执行成功记 PASSED。比较保留误差、非有限值、dtype 是否保持及十次输出稳定性；若已有明确的浮点验收门槛，才在 compare 中显式加 `--atol` 和 `--rtol`。

回传每批的 `summary.csv`；异常时附 `RECOVERY/result.json` 的状态和 error。4＋22 方式会得到两份 CSV，共 26 项；单批方式得到一份。原执行 fixture、回收 native 和详细 `report.json` 保留，后续离线分析可以直接复用。

本页命令已核对现有 CLI 并检查 Shell／Python 片段语法；主机测试覆盖 FP32 Prefill Conv 的字节预算、旧 runner 拒绝、实际 C++ 计划解析及 32 MiB 边界。本地主机未执行 OHOS 交叉构建或设备验证。完成后的实际精度状态更新到 [Target 状态表](target-accuracy-status.md)。
