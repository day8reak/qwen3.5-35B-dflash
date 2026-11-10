# 修正算子环境后：批量生成 OMC、上板与比较

更新：2026-11-10。本页保留**现有 Target 边界小图批次的完整复测命令**。旧 `environment-recheck-001` 有 120 项编译成功并回传数值摘要；用户随后确认重导的 Norm／Conv 26 项全部编译成功，累计 **146/158 成功，剩余 ConvState 12 项**。新增项中 22 项数值已回传，另 4 项待回传；累计 142 项有数值摘要，当前结论见[状态表](target-accuracy-status.md)。已执行的批次和输出直接保留，不重复本页整批流程。

原 38 项的 `inspect-compile --failed` 及 Norm／Conv 重导编译均已完成。[ConvState 成功整图对照](target-compile-triage.md#成功整图提供的对照)已回传，已确认专用 builder 和 executor 内核路径；两个既有小图的格式选项对照均已失败，参考整图读取遇到 schema 冲突，检查器已补充 AIR producer 描述核对，下一步只做[参考图重读](target-compile-triage.md#下一步读取构建策略与图上下文)，不重跑整批。已有数值回传的 142 项无需重跑；其余 4 项优先回传已有 CSV，需要补测时使用实际新 OMC 批次的执行／回收／比较流程。

**新增 26 项的专用执行步骤见 [Norm／Conv 上板与比较](target-norm-conv-validation.md)**，包含变量初始化、单批 26 项／4＋22 分批选择，以及运行、回收和 CSV 比较命令。

机器分工：DDK／OMG 在 **35 服务器**；310P golden 仍来自此前的**权重服务器**。TorchAir 导出使用已修正且能导出 CPro 算子的环境；上板命令在原来已配置好 HDC 的主机执行。比较使用本批自带的独立输入和 golden，不需要等待四层整图 dump 的自动映射。

这是已有边界批次的复测命令。[四层整图 dump](target-accuracy-status.md#3-主线四层整图一次-dump离线批量定位)仍是后续观察真实误差传播的方法；[compare_target_dump.py](target-dump-compare.md)用于那条整图路线。本页比较工具是 target_layer_probe.py compare，不混用两套输入清单。

## 本轮已编译：从 prepare 失败处继续

**历史恢复步骤：本批后续已有 120 项数值回传，当前无需再执行这一节。** 下方仅保留准备阶段失败时的处理方法；已启动的请求用原 execution 回收，不能重复启动。

此前回传为 `compiled input shape/dtype changed: arg6_1`、`stage=prepare`、`device_execution=NOT_RUN`。旧检查器漏掉 `INT16/DT_INT16` 别名，合法 INT16 描述会被解析为未知 dtype 并误报不一致。`ff23e33` 补齐映射，保留严格的 shape/dtype 校验；发生真实差异时显示具体 case、expected/observed 和 schema 枚举名。42 项主机测试通过，包括误报复现、INT16 夹具原字节保留及真实不匹配拒绝。

沿用当前 OMC_BATCH、RUNTIME_CONFIG、CPRO_DDK_ROOT 和 HDC 设置，更新代码后使用新的 execution 目录：

```bash
: "${REPO_DIR:?}" "${RECHECK:?}" "${OMC_BATCH:?}" "${RUNTIME_CONFIG:?}" "${CPRO_DDK_ROOT:?}"
git -C "$REPO_DIR" pull --ff-only origin main

export EXECUTION="$RECHECK/execution-002"
export RECOVERY="$RECHECK/recovery-002"
export COMPARISON="$RECHECK/comparison-002"

python "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --compiled "$OMC_BATCH" \
  --config "$RUNTIME_CONFIG" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$EXECUTION" \
  --apply
```

使用尚不存在的编号；保留旧 prepare 失败目录。此次不用重导 AIR、重编 OMC 或重建已有 INT16 runner，也不需要 recover 那个未启动的请求。新请求启动后按下方第 4／5 节回收、比较；若再次在 prepare 失败，只回传新的完整 error 字段即可看到真实接口差异。默认仍检查并运行全部 120 个 COMPILED 项，38 个未编译项保留为覆盖缺失。

下面第 0～2 节供重新生成批次时使用；本轮无需重复执行。

## 0. 更新代码并设置本轮目录

在各执行主机沿用其真实 REPO_DIR、AI_RUN_DIR 和已经修正的环境。以下变量使用**当前主机**上的路径；跨机器迁移后重新设置，不能沿用另一台机器的绝对路径。已有同名复测目录时换新的编号。

```bash
: "${REPO_DIR:?}" "${AI_RUN_DIR:?}"
git -C "$REPO_DIR" pull --ff-only origin main

export RECHECK="$AI_RUN_DIR/target/tests/environment-recheck-001"
export BATCH="$RECHECK/air"
export OMC_BATCH="$RECHECK/omc"
export EXECUTION="$RECHECK/execution"
export RECOVERY="$RECHECK/recovery-001"
export COMPARISON="$RECHECK/comparison-001"
export RUNTIME_CONFIG="$AI_RUN_DIR/runtime/runtime.json"
```

保留原已跑通的 runtime 配置、native runner 和 HDC 设置。若 runtime.json 不在上述位置，修改 RUNTIME_CONFIG。若更换了 DDK 版本，runtime 配套 assets 中的 DDK 身份也必须与本批实际编译版本一致，不能为了绕过检查随意改版本标签。

## 1. 按需要重新导出 AIR

**错误算子若参与过 AIR 导出，本轮先重导。** 复用原权重服务器上可信的 capture，保持权重、输入、golden 和量化合同；不用为重编 OMC 重新采集已经确认正确的 310P 数据。

在已修正的 TorchAir 导出环境中，设置 GOLDEN_CAPTURE 为已有完成态 capture 目录，TARGET_SOURCE 为原 Target 代码目录，CPRO_EXPORT_SCRIPT 为正确的原 CPro 导出脚本：

```bash
: "${GOLDEN_CAPTURE:?}" "${TARGET_SOURCE:?}" "${CPRO_EXPORT_SCRIPT:?}"
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" export \
  --capture "$GOLDEN_CAPTURE" \
  --source-root "$TARGET_SOURCE" \
  --converter-script "$CPRO_EXPORT_SCRIPT" \
  --output "$BATCH"
```

不加 --points/--kinds，因此覆盖捕获清单中的全部可导出边界。导出入口只导入原脚本的 converter 定义，每个输出独立成图，不调用原四层／40 层整图导出 main。失败项和日志保留在 manifest.json 和 batch 根目录的各 case 日志中；PARTIAL 不代表全部失败。

将 **整个 BATCH 目录**迁移到 35 服务器，包括 manifest.json、models/ 下的 export.air、schema、fixture 和 weight sidecar；在 35 上把 BATCH 设为迁移后的路径。

如果已确认错误仅影响 OMG 实现选择、现有 AIR 是正确算子导出的，则跳过重导，改用已有目录，例如：

```bash
export BATCH="$AI_RUN_DIR/target/air-batch"
```

仅在确认该目录就是正确批次时使用此例。旧四层 AIR 与本页含 manifest.json 的小图 batch 不是同一入口。

## 2. 在 35 服务器批量编译 OMC

CPRO_DDK_ROOT、CPRO_DDK_ID 必须指向修正后的实际 DDK。激活已经修正的 DDK／算子环境，再执行下方命令；不把历史 147 服务器的安装路径直接套到 35。

```bash
: "${CPRO_DDK_ROOT:?}" "${CPRO_DDK_ID:?}" "${BATCH:?}" "${OMC_BATCH:?}"
COMPILE_OPTIONS=()
# 仅当当前 DDK 仍需要且诊断与当前包/库匹配时，取消下一行注释：
# COMPILE_OPTIONS+=(--tbe-diagnosis "${TBE_DIAGNOSIS:?}")
# 若本次已确认的编译配置要求保留原格式，取消下一行注释：
# COMPILE_OPTIONS+=(--use-origin-format)

python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" compile \
  --batch "$BATCH" \
  --output "$OMC_BATCH" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --ddk-id "$CPRO_DDK_ID" \
  --compiler-profile isolated-kirinx90 \
  "${COMPILE_OPTIONS[@]}"
```

该入口构造 isolated-kirinx90 编译环境；实际选定的路径和命令保存在各 case 的 omc/omg-environment.json、compile-command.json、report.json 和 compile.log 中。修正外层 shell 不自动证明子进程选中了正确算子，复测时以这些实际记录为准。不要改用 sourced-wrapper-default：它是目标尚未确认的入口诊断，板端 fixture 准备会拒绝。

小图使用内嵌权重 OMC。默认编译全部 EXPORTED 项；未导出项保留 NOT_EXPORTED，单图失败不会阻止后面的编译。PARTIAL 的 CLI 退出码为 1；逐节执行本页命令，检查下列摘要后再进入上板阶段：

```bash
python - "$OMC_BATCH/compile.json" <<'PY'
import json, sys
from collections import Counter
with open(sys.argv[1]) as f:
    report = json.load(f)
counts = Counter(row["status"] for row in report["cases"].values())
print(json.dumps({"status": report["status"], "cases": dict(counts)}, ensure_ascii=False, indent=2))
if report["status"] not in ("COMPILED", "PARTIAL") or not counts["COMPILED"]:
    raise SystemExit("本批没有可上板的完成态编译结果；检查各 case 的 compile.log。")
PY
```

有 COMPILED 项即可运行这些成功图；未编译项继续显示为覆盖缺失。不要删除失败项或改写 compile.json。

## 3. 批量上板

在已配置 HDC 的主机沿用 SN、BIP、HDC 和 RUNTIME_CONFIG。若需把结果从 35 搬到另一主机，**AIR batch 与 OMC batch 一起迁移并保持相对位置**；OMC 的 compile.json 通过 source_batch 引用 AIR 中的输入和 golden。相同主机不需要再复制。

runner 已是此前支持 INT16 和每次调用恢复输入的版本时，直接复用。仅在工具明确提示旧 runner 缺少能力时，待其余任务结束，用原构建资源更新 runtime.json 中 native_directory 指向的目录。若它就是标准 runtime/native：

```bash
python "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$AI_RUN_DIR/runtime/native" --replace
```

**上面是条件步骤，不需要每次重建 runner。** 随后启动本批所有已编译图：

```bash
python "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --compiled "$OMC_BATCH" \
  --config "$RUNTIME_CONFIG" \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$EXECUTION" \
  --apply
```

虽然历史文件名带 draft，该入口已支持 Target 边界清单。不加 --cases，会选择本批全部 COMPILED 项。每图 3 次预热、10 次采样，每次调用前恢复原输入，执行前按实际编译接口绑定槽位和 dtype。板端逐图运行，失败会停止后续图；成功执行过的输出仍可回收。

同一 EXECUTION 只启动一次。如果终端中断或结果 PENDING，接下来回收这次请求，不再次 --apply。

## 4. 回收这次运行

```bash
python "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --recover "$EXECUTION" \
  --output "$RECOVERY"
```

回收检查同一台设备、同一请求及 fixture 哈希，不重新执行模型。若仍为 VECTOR_VALIDATION_PENDING，保留原 EXECUTION，稍后使用新的 recovery-002 目录再次回收：

```bash
export RECOVERY="$RECHECK/recovery-002"
python "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --recover "$EXECUTION" \
  --output "$RECOVERY"
```

RECOVERY 指向最后一次实际取得输出的目录。只在取得 native/ 后进入比较；不要将 PENDING 当作数值失败，也不要为了重新启动删除锁或旧请求。

## 5. 批量数值比较，生成 CSV / JSON

```bash
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" compare \
  --fixture "$EXECUTION/fixture" \
  --native "$RECOVERY/native" \
  --output "$COMPARISON"
```

若已有冻结的浮点容差，可在同一命令中显式加 --atol 和 --rtol；本页不代填新阈值。默认浮点 METRICS_RECORDED，记录每次误差、非有限值、实际输出 dtype 和重复稳定性；CacheUpdate 对整缓存逐位比较。编译成功、运行成功和精度通过分别判断。

查看／回传这些文件即可，不需要粘贴所有日志：

| 文件 | 用途 |
|---|---|
| $OMC_BATCH/compile.json | 全批编译结果、DDK 身份及未编译项 |
| $RECOVERY/result.json | 请求完成状态、执行／回收和初步比较结果 |
| $COMPARISON/report.json | 详细数值、重复稳定性、缺失项和首个失败边界 |
| $COMPARISON/summary.csv | 按层、调用和模块浏览数值结果 |

保留原 AIR batch 和 execution/fixture；后续更改报告展示或补充离线分析无需重新编译、上板。若要再次运行或重新比较，使用 execution-002／comparison-002 等新目录。本页命令已核对实际 CLI 和 Shell 语法；真实环境修正效果以用户新一轮回传为准。
