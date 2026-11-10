# Target OMC：Norm／Conv 已通过，剩余 ConvState 12 项

更新：2026-11-10。用户确认重导后的 **Norm 20 项、Conv1d 6 项全部编译成功**。累计 **146/158 项已编译，剩余 ConvState 12 项**。新增成功依据用户回传确认，具体 OMC 哈希尚未提供；随后收到 Norm 18 项、Conv1d 4 项的小误差且稳定的数值摘要，另 4 项待回传，详见[状态表](target-accuracy-status.md)。当前按用户安排集中解决 ConvState 编译；原批次报告保持原样。

**成功四层图已确认 ConvState 专用 graph builder、executor 内核及 OMC 保存成功。** 取消 `use_origin_format` 的两项小图对照仍失败。最新参考图读取已完成：原图有 RefData 状态输入、卷积消费者和 hybrid 构建路径；两边 `_executor_type=1`，端口 dtype／layout 相同。结果及下一步见[图上下文对照](#已完成读取构建策略与图上下文)。Norm／Conv 重导编译和本次 22 项数值回传已完成；缺少的 4 项优先补回已有 CSV。编译成功不计作精度通过。

以下保留此前完整 38 项检查摘要的归因及处理结果：

| 组别 | 本批证据 | 当前判断与处理 |
|---|---|---|
| 普通 Norm：20 项 | 旧 AIR 含 `Add → RmsNorm`，无导出修正记录；`add_kernel.cpp` 报类型不支持，随后 `Add ProduceTiling failed`、`inputTensorPtr is NULL` | gamma 预计算修正后重导，用户确认 20 项均编译成功；18 项数值已回传、误差较小且稳定，另 2 项待回传 |
| Conv1d：6 项 | 旧图在 `Convolution` Verify 报 `Input[0] and input[1] must have the same dim num`，无导出修正记录／结构检查 | 显式四维路径重导后，用户确认 6 项均编译成功；4 项数值已回传、误差较小且稳定，另 2 项待回传 |
| ConvState：12 项 | 小图在 `generated_lib` 的 builder／format config 处失败；成功整图同样有候选支持拒绝，但后续调用专用 graph builder 并保存 executor 内核 | 实现存在，整图可构建；省略 `use_origin_format` 的两项对照仍失败。停止该参数重试，检查构建策略和图上下文 |

Conv 的类型计数中出现 `Unsqueeze` 不足以证明修正已生效：它可能只处理权重，必须检查 Conv2D 的两条输入边。ConvState 摘要中的 `substore: !` 没有提供有效 store 名称，不能据此认定整个 DDK 缺少实现。

三组都出现了 `libai_npucore_ascendc_kernel.so not valid`，但这条消息不能替代上述各组的实际终止原因；先与同批成功 GDN 的日志对照。`IR InferShape ... Success` 和 `PassManager::Run begin` 即使标成 E 级也不是失败证据。

已成功的 146 个 OMC、已有 142 项数值回传对应的上板输出和全部 golden 保留。累计覆盖不通过手动合并／改写旧 `compile.json` 实现。

## 原 Target 给出的基线

本次实际阅读了此前保留的 master Target 源码。`models/modeling_qwen3_5_moe_hiai_cpro.py` 的 SHA256 为 `bd716718818b6b152ece5a2e345a5b674f091f53ba82059c86cd37060511f07d`，`export_glm_model_qwen3_5_moe.py` 为 `7ef1710a301b88a8f9f8201f95b97b62d22ac80f43a1c5bfffe710e1b39d21f6`。这是保留版本，尚不能证明与 35 服务器当前源码相同；本批 `export.json.source_files` 可用于核对真正参与导出的文件。私有源码不复制到仓库。

| 失败组 | 原 Target 实际调用 | 小图需要核对的差异 |
|---|---|---|
| 普通 Norm，20 项 | `npu_rms_norm(hidden_states, weight + 1.0, eps)[0]`，随后转回输入 dtype | 原图调用和权重语义保持一致。已有小图修正是在导出前按捕获权重 dtype 计算 gamma，避免在编译时计算 Add；检查本次 AIR 是否真的包含此修正 |
| Conv1d，6 项 | `F.conv1d(padded, weight, groups=conv_dim)`；`padded` 由 ConvState 的第二输出直接产生 | 独立小图的 padded 是捕获的三维 Data。旧 AIR 的 Conv2D 输入三维、filter 四维；已有修正用显式升维／降维表达相同卷积，需检查新的 AIR 结构 |
| ConvState，12 项 | `linear_attention_convstate_update(mixed_qkv, conv_state, linear_update_tensor)` 返回两项；第一项接 `conv_state.copy_`，第二项接卷积 | 单输出探针只接出选定输出，状态是普通 Data。先核对当前实现注册和格式支持，再判断是否需要保留原双输出、消费者及状态写回的连续子图 |

普通 Norm 的实现以实际 forward 为准。原导出脚本虽然注册了 `adn_rms_norm → AdnRmsNorm`，普通 Norm 调用的是 `npu_rms_norm`；不能据此直接换算子。已成功的 gated Norm 使用 FP32 输入和直接 gamma，也不能替代普通 Norm 的验证。

原 ConvState fake 实现返回 `[1,8192,4]` 和 `[1,8192,T+3]`，converter 输出端口为 `y1/y2`。历史失败日志中，DDK 将第二输出推成 `[1,8192,1,T+3]`。这使“原图直接连接卷积”和“小图以三维 Data 为卷积输入”成为值得核对的差异，但本批摘要仍未证明该 shape 差异就是这 12 项的失败原因。也不能只修改 golden 的 shape 来消除差异。

## 原成功命令应如何借鉴

用户提供的四层 T64 成功命令使用 `source set_ascendc_env.sh`，随后调用外层 `tools/tools_omg/omg`，参数为 `--framework 1 --limit_partition_size=false --save_weights_as_external_data=true`。40 层命令还带有原图输入名、RefData shape，以及 `--dynamic_dims="64,64,64,64,64;1,1,1,1,1"`。

| 对照项 | 核对方法 |
|---|---|
| wrapper 与直接 master/omg | wrapper 除选择二进制外还配置 TBE／AscendCPP 库、`CCE_LIB_DIR`、`KIRIN_CONF_JSON_PATH` 和工具路径。与失败任务保留的 `omg-environment.json` 比较，不能仅凭二进制路径不同认定根因 |
| 编译选项 | 当前小图显式使用 `--target=omc --platform=kirinx90`、内嵌权重和 `use_origin_format=true`；原成功命令省略目标／平台并使用外置权重。先以实际日志确认目标与实现选择，不再轮流盲试开关 |
| shape 和状态 | 原示例为 T64，40 层命令含 T64/T1 档位及 RefData；本批历史 Prefill 是静态 T256。不能把原图输入名／dynamic_dims 直接复制到已经固化 reshape 的小图 |
| 导出与转换阶段 | 原脚本的 `torchair.dynamo_export(dynamic=False, ...)` 与 OMG 的动态档位属于不同阶段；分别检查 AIR 和编译命令 |
| 上板入口 | 用户提供的是 `run_model_prefill.sh -m qwen3_5_35B/qwen3_5_35B.omc`。本地没有该脚本正文，不能据此猜测具体输入顺序或 ABI；本次编译定位尚不需要上板 |

35 是当前 DDK／OMG 服务器；310P golden 仍来自此前权重服务器。147 和历史命令中的安装路径只作为来源记录。

## 已完成的只读检查入口

本轮已经完成并回传，无需再次执行。下方保留给其他新批次使用。

沿用 35 上本批的 `REPO_DIR`、`RECHECK`、`OMC_BATCH` 和修正后的 `CPRO_DDK_ROOT`：

```bash
: "${REPO_DIR:?}" "${RECHECK:?}" "${OMC_BATCH:?}" "${CPRO_DDK_ROOT:?}"
git -C "$REPO_DIR" pull --ff-only origin main
export COMPILE_INSPECT="$RECHECK/compile-inspect-001"

python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" inspect-compile \
  --batch "$OMC_BATCH" --failed \
  --ddk-root "$CPRO_DDK_ROOT" --compiler-entry \
  --output "$COMPILE_INSPECT"

cat "$COMPILE_INSPECT/summary.txt"
```

输出目录若已存在，换一个编号。`--failed` 自动选择本批全部 38 个 FAILED case，只读取其现有证据；整个检查不执行导出、OMG 或设备命令。`--compiler-entry` 只读取入口脚本和环境记录，不 source 脚本。

`summary.txt` 列出每个失败边界的导出修正标记、AIR 算子类型、退出阶段与关键错误摘要；错误检索覆盖日志前部，避免只看末尾退出信息。完整 `report.json` 保留源码身份、图描述、环境、错误上下文和 IR 清单。日志扫描有 64 MiB 上限；摘要是定位线索，不自动认定根因。检查结果为 PARTIAL 时先看缺失字段，它不表示又进行了一次失败编译。

## 只重导 Norm／Conv，先编译四项

**本节为已完成流程的复现命令。本轮 26 项已全部编译成功，无需再次执行。** 旧失败批次 `manifest_sha256=9f433d0092c93f5c156a5c6ab59d6d603dcef338b8132dc28b1e0f77fcf1b1e2`、`source_batch=../../../air-batch`，与之前 AIR 批次相同。更新 Python 代码再执行 `compile` 不会更新既有 AIR。以下使用仓库中已实现的导出修正，不换 Norm 算子、不改捕获 dtype 或 golden。

本轮导出在**权重服务器**执行，Python 使用该机已经修正的 TorchAir／自定义算子环境。先进入这台机器上的 GitHub 仓库根目录，再设置下列变量。`TARGET_SOURCE` 和 `GOLDEN_CAPTURE` 的两个占位路径必须替换成本机实际目录；当前对话尚未提供它们在权重服务器上的确切位置。35 的路径不能直接用于这里。

```bash
export REPO_DIR="$(git rev-parse --show-toplevel)"
export AI_RUN_DIR="${REPO_DIR}-run"
export TARGET_SOURCE="/填写权重服务器上的原Target源码根目录"
export CPRO_EXPORT_SCRIPT="$TARGET_SOURCE/export_glm_model_qwen3_5_moe.py"
export GOLDEN_CAPTURE="/填写权重服务器上本批成功的capture目录"
export AIR_NORM_CONV="$AI_RUN_DIR/target/air-norm-conv-002"
```

| 变量 | 在权重服务器上指向什么 |
|---|---|
| `REPO_DIR` | GitHub 工程，含 `targets/cpro/tools/target_layer_probe.py` |
| `AI_RUN_DIR` | 本机仓库外的生成目录；上述设置选择仓库的同级 `-run` 目录，已有专用输出根目录也可沿用 |
| `TARGET_SOURCE` | 原 Target 工程，含 `models/modeling_qwen3_5_moe_hiai_cpro.py`、`utils.py` 和导出脚本 |
| `CPRO_EXPORT_SCRIPT` | 原 CPro converter 脚本的完整路径；文件不在源码根目录时按实际位置设置 |
| `GOLDEN_CAPTURE` | 已完成的原始捕获目录，包含 `manifest.json`、输入／输出和捕获权重；其状态必须为 `CAPTURED`，不是 HF checkpoint、AIR batch 或 OMC batch |
| `AIR_NORM_CONV` | 本次新 AIR 目录，必须位于 `AI_RUN_DIR` 下且尚不存在 |

旧 `inference_target_probe.py` 命令的示例 capture 是 `$AI_RUN_DIR/target-layer-accuracy/golden`；`inference_target_steps.py` 则在其输出目录的 `capture/` 子目录，例如 `$AI_RUN_DIR/target-layer-accuracy/direct-001/capture`。这只是入口约定，实际应选本批已经成功的目录。设置新 `AI_RUN_DIR` 不会移动已有 capture。

导出前可用下面的只读检查确认路径。此步骤不重新采集 310P golden：

```bash
python - <<'PY'
import json, os
from pathlib import Path
repo = Path(os.environ['REPO_DIR'])
source = Path(os.environ['TARGET_SOURCE'])
for p in (repo / 'targets/cpro/tools/target_layer_probe.py',
          source / 'models/modeling_qwen3_5_moe_hiai_cpro.py',
          source / 'utils.py', Path(os.environ['CPRO_EXPORT_SCRIPT'])):
    assert p.is_file(), f'路径不存在: {p}'
capture = Path(os.environ['GOLDEN_CAPTURE'])
m = json.loads((capture / 'manifest.json').read_text())
assert m.get('artifact_kind') == 'qwen35-target-cpro-boundary-probe-v1' and m.get('status') == 'CAPTURED', '必须使用完成态捕获目录'
out = Path(os.environ['AIR_NORM_CONV']).resolve()
assert Path(os.environ['AI_RUN_DIR']).resolve() in out.parents and not out.exists(), 'AIR 输出须为本机 AI_RUN_DIR 下的新目录'
print('capture:', capture, 'status:', m['status'], 'regions:', len(m.get('regions', [])))
print('新 AIR 输出:', out)
PY
```

`CPRO_DDK_ROOT`、`CPRO_DDK_ID`、`RECHECK` 和 OMC 输出变量在后面的 **35 编译步骤**设置；不会由上述变量自动同步到另一台机器。

```bash
: "${REPO_DIR:?}" "${AI_RUN_DIR:?}" "${GOLDEN_CAPTURE:?}" "${TARGET_SOURCE:?}" "${CPRO_EXPORT_SCRIPT:?}"
git -C "$REPO_DIR" pull --ff-only origin main

python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" export \
  --capture "$GOLDEN_CAPTURE" \
  --source-root "$TARGET_SOURCE" --converter-script "$CPRO_EXPORT_SCRIPT" \
  --kinds norm conv --output "$AIR_NORM_CONV"
```

新目录已存在时换编号。导出后先确认本批 26 项均带有修正和源图检查结果，避免再次拿旧 AIR 重编：

```bash
python - "$AIR_NORM_CONV" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads((root / 'manifest.json').read_text())
expected = {'norm': 'FOLD_ZERO_CENTERED_GAMMA', 'conv': 'EXPLICIT_CONV2D_SINGLETON_HEIGHT'}
assert manifest['status'] == 'EXPORTED', '导出未全部完成，先检查 manifest.json 中失败项'
assert Counter(c['kind'] for c in manifest['cases']) == {'norm': 20, 'conv': 6}, '不是本批 26 项'
for case in manifest['cases']:
    receipt = json.loads((root / case['export']['path']).read_text())
    kinds = {a['kind'] for a in receipt.get('export_adjustments', [])}
    check = receipt.get('export_graph_check') or {}
    assert expected[case['kind']] in kinds and check.get('status') == 'PASSED', case['id']
    print(case['id'], expected[case['kind']], json.dumps(check, ensure_ascii=False))
print('26 项源图修正已记录；尚未证明 OMG 编译或数值通过。')
PY
```

检查通过后，将整个 `AIR_NORM_CONV`（含 manifest、models、fixture、weight）迁移到 35；将变量设为 35 上的实际目录。沿用本批已修正 DDK 和编译环境，先测试 hidden Norm、Q Norm、Prefill Conv、Decode Conv：

```bash
: "${REPO_DIR:?}" "${RECHECK:?}" "${AIR_NORM_CONV:?}" "${CPRO_DDK_ROOT:?}" "${CPRO_DDK_ID:?}"
export OMC_NORM_CONV_PROBE="$RECHECK/omc-norm-conv-probe-002"
NORM_CONV_OPTIONS=(--use-origin-format)
# 仅当 environment-recheck-001 的实际编译也使用了同一份、仍有效的诊断时添加：
# NORM_CONV_OPTIONS+=(--tbe-diagnosis "${TBE_DIAGNOSIS:?}")

python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" compile \
  --batch "$AIR_NORM_CONV" --output "$OMC_NORM_CONV_PROBE" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --compiler-profile isolated-kirinx90 "${NORM_CONV_OPTIONS[@]}" \
  --cases l00_c00_input_layernorm_o0 l03_c00_self_attn_q_norm_o0 \
          l00_c00_conv1d_o0 l00_c01_conv1d_o0

python - "$OMC_NORM_CONV_PROBE/compile.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
for name in r['selected_cases']:
    row = r['cases'][name]
    print(name, row['status'], row.get('error', ''))
PY
```

先回传这四项结果。若失败，读取对应新 `compile.log`；已经去掉 Add／通过源图 rank 检查后出现的新失败需要另行判断。全部通过后，才编译剩余 22 项，保留四个成功模型：

```bash
(
set -e
python - "$OMC_NORM_CONV_PROBE/compile.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
assert len(r['selected_cases']) == 4
assert all(r['cases'][n]['status'] == 'COMPILED' for n in r['selected_cases']), '代表项尚未全部通过'
PY
mapfile -t NORM_CONV_REST < <(python - "$OMC_NORM_CONV_PROBE/compile.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
for name, row in r['cases'].items():
    if row['status'] == 'NOT_SELECTED':
        print(name)
PY
)
test "${#NORM_CONV_REST[@]}" -eq 22
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" compile \
  --batch "$AIR_NORM_CONV" --output "$RECHECK/omc-norm-conv-rest-002" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --compiler-profile isolated-kirinx90 "${NORM_CONV_OPTIONS[@]}" \
  --cases "${NORM_CONV_REST[@]}"
)
```

用户已确认上述范围共 26 项编译成功。[新增 26 项执行文档](target-norm-conv-validation.md)提供两种批次方式的完整命令，不手动拼接或改写旧 `compile.json`。其中 22 项数值已回传且稳定，另 4 项待回传；无需重复已完成的 22 项。

## ConvState：只取两份代表日志

**两份失败代表日志和成功整图的关键日志均已回传，无需重复提取。** 成功日志为 35 上的 `/home/w00949577/qwen3.5-35B-dflash-run/target/omg.log`，PID 130048；日志记录输出相对路径 `qwen3_5_35B/qwen3_5_35B.omc`，其绝对路径和哈希尚未核验。本次原图为 Prefill T64，小图为 Prefill T256／Decode T1。用户确认使用同套 DDK；编译器／库的逐文件身份仍以各次记录为准。

Prefill `l00_c00_convstate_update_o0`（PID 131252）与 Decode `l00_c01_convstate_update_o0`（PID 136256）出现相同失败链路：

| 原日志行 | 已观察到的事实 | 含义 |
|---|---|---|
| 61 | `convertFunc not exist` | 后续已进入该算子的专用 Verify／Infer；不能单凭这条认定 AIR 解析失败 |
| 301–310、327–328 | `LinearAttentionConvstateUpdateVerify exit, ret=0`，`Infer exit success` | 算子校验与 shape 推导成功；不是此次 Conv1d 曾出现的 rank Verify 错误 |
| 333–341 | `custom_ascendc_lib`、`elementary_lib`、`ascendc_lib`、`fe_lib` 支持检查均未通过；TBE store 无对应记录 | 这些是候选检查结果；成功整图也有相同拒绝，不能用它们判断最终实现不可用 |
| 347–348 | `force supported by NPU`，随后分区 NPU:1 | 强制分区发生在实现检查未通过之后，不是支持成功证据 |
| 372、399–405 | `generated_lib` 的 OpBuilder 不支持此 OP；format config 获取失败，NPU 子图构建退出 | 这是小图最终失败点；结合成功整图，早期候选拒绝本身不能解释为何走到此处 |

日志第 308 行还确认了 DDK 推导的输出：y1 均为 `[1,8192,4]`；Prefill y2 为 `[1,8192,1,259]`，Decode y2 为 `[1,8192,1,4]`。成功整图同样将 y2 推为四维 `[1,8192,1,67]`，后接四维 Conv2D。四维 y2 本身不构成这次失败的解释；两张失败代表图导出的都是 y1。

### 成功整图提供的对照

| 成功日志位置 | 已观察事实 | 可以得出的结论 |
|---|---|---|
| 418、908、1398 | 同样有 `convertFunc not exist` | 该提示不能单独解释失败 |
| 5250–5266 等 | ConvState Verify／Infer 成功；y2 四维，Conv2D 输入和 filter 均四维 | 原图的这段 shape 路径可继续构建 |
| 11718、13091–13113、13892–13895 | 同样有各候选 store 不支持、强制 NPU 分区 | 之前的支持拒绝不是整图／小图分歧的充分证据 |
| 14130 | `LLMLinearAttentionConvStateFusionPass DefinePatterns` | 只证明定义了规则，不证明命中或完成融合 |
| 15124–15127、18315–18318、21435–21438 | 三次进入 `linear_attention_convstate_update_graph_builder.cc GetOutputDesc` | 成功图调用了专用 graph builder；失败小图停在 `generated_lib` builder／format config 路径 |
| 31417 | `executor_linear_attention_convstate_update, kernelSize = 748` | 此次成功编译保存了 ConvState executor 内核；不能认定 DDK 完全缺少实现 |
| 日志末尾 | `SaveCompiledModelToFile SUCCESS`、`OMG generate offline model success` | 整图 OMC 编译／保存成功，尚不是上板或精度证据 |

当前结论收敛为**构建路径差异**。尚未确定触发条件是编译参数、输入 dtype／format、状态／消费者边界还是其他环境差异；也未证明必须融合后才能编译。小图显式添加的 `use_origin_format=true` 已做省略对照，两阶段仍在相同 builder／format config 处失败；本批证据排除“仅省略该参数即可修复”，不证明它在所有图上均无影响。

已重新核对保留的原 Target 源码及当前小图入口：

| 检查项 | 原 Target 与当前小图 | 当前判断 |
|---|---|---|
| 调用与输入顺序 | 同为 `linear_attention_convstate_update(mixed_qkv, conv_state, linear_update_tensor)`，converter 映射 `x1/x2/x3`，无额外属性 | 未发现调用顺序或算子名差异 |
| 辅助输入 | 原导出传 `zeros([128], int32)`；当前捕获亦按此构造 | 不能把全零辅助张量直接认定为本次格式匹配失败原因 |
| fake 输出 | 保留的原导出声明 y1 `[1,8192,4]`，y2 `[1,8192,T+3]`，dtype 均随 x1；小图加载用户指定的原 converter 脚本 | 服务器实际脚本版本待其 export 记录核对；源 IR 的空中间 shape 本身不是已知 rank 错误 |
| 输出消费者与状态 | 原图同时消费 y1/y2：y2 接 Conv，y1 接 `conv_state.copy_`；当前小图仅导出选定输出，状态为普通 Data | 存在图边界差异，尚无证据证明它导致 format config 失败 |

<a id="下一步两个小图的格式选项对照"></a>

### 已完成：两个小图的格式选项对照

`convstate-default-format-001` 两项均返回 1：Prefill PID 146807，Decode PID 146848。两份日志均在 L373 报 `generated_lib` OpBuilder 不支持 ConvState，L400 报 format config 失败，L414 报 OMG 失败；结尾还有 `OptImplStrategy Optimize Fail, strategyType:0`。用户回传命令确认未再传 `use_origin_format`。不再重跑本节。

成功日志曾出现 `pre_graph_save_hybrid_impl.cc` 和专用 builder，失败日志曾出现 `pre_graph_save_single_impl_strategy.cc`，最新日志明确记录 `strategyType:0`。这提示应核对构建策略的分流条件；尚未读取枚举／选择逻辑，不能仅凭数字 0 断言其含义，也不能直接认定状态写回或融合是根因。

<details>
<summary>保留已完成的参数对照命令，仅供复现</summary>

在 **35 服务器本批已修正的 Python／DDK shell** 执行。复用 L0 Prefill／Decode 的 o0 AIR，从旧报告读取源批次、DDK 和 TBE 诊断路径；只省略 `--use_origin_format=true`，仍指定 `kirinx90`、`isolated-kirinx90` 和内嵌权重。省略参数表示采用编译器默认值，不等于已经确认默认值为 false。不重新 source 另一套环境，不重新捕获 golden／导出 AIR。

```bash
export REPO_DIR="/home/w00949577/qwen3.5-35B-dflash"
export AI_RUN_DIR="/home/w00949577/qwen3.5-35B-dflash-run"
export CONVSTATE_BASE="$AI_RUN_DIR/target/tests/environment-recheck-001/omc"
export CONVSTATE_PROBE="$AI_RUN_DIR/target/tests/convstate-default-format-001"

python - <<'PY'
import hashlib, json, os, subprocess, sys
from pathlib import Path
base = Path(os.environ['CONVSTATE_BASE'])
out = Path(os.environ['CONVSTATE_PROBE'])
names = ['l00_c00_convstate_update_o0', 'l00_c01_convstate_update_o0']
read = lambda p: json.loads(p.read_text())
batch = read(base / 'compile.json')
source = (base / batch['source_batch']).resolve()
old = [read(base / 'models' / n / 'omc/report.json') for n in names]
assert not out.exists(), '输出目录已存在，请换 CONVSTATE_PROBE 编号'
assert batch['compiler_profile'] == 'isolated-kirinx90'
for name, row in zip(names, old):
    assert row['status'] == 'FAILED', name
    assert row['compiler_profile'] == 'isolated-kirinx90', name
    assert row['requested_options'] == {'save_weights_as_external_data': 'false',
                                        'use_origin_format': 'true'}, name
    air = source / 'models' / name / 'export.air'
    assert hashlib.sha256(air.read_bytes()).hexdigest() == row['air']['sha256'], 'AIR 已变化'
    assert Path(row['omg_path']).is_file(), '旧 OMG 路径已失效'
    with Path(row['omg_path']).open('rb') as f:
        h = hashlib.sha256()
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    assert h.hexdigest() == row['omg_sha256'], 'OMG 已变化，不能作为仅参数变化的对照'
settings = [(r['ddk_root'], r['ddk_id'], r.get('tbe_compatibility', {}).get('diagnosis')) for r in old]
assert settings[0] == settings[1], '两个旧 case 的编译环境配置不同'
ddk, ddk_id, diagnosis = settings[0]
cmd = [sys.executable, str(Path(os.environ['REPO_DIR']) / 'targets/cpro/tools/target_layer_probe.py'),
       'compile', '--batch', str(source), '--output', str(out),
       '--ddk-root', ddk, '--ddk-id', ddk_id, '--compiler-profile', 'isolated-kirinx90',
       '--cases', *names]
if diagnosis:
    cmd += ['--tbe-diagnosis', diagnosis]
log = out.with_suffix('.cli.log')
log.parent.mkdir(parents=True, exist_ok=True)
print('仅编译:', ', '.join(names), '\nCLI 日志:', log, flush=True)
with log.open('x') as f:
    done = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
if not (out / 'compile.json').is_file():
    raise SystemExit(f'未生成报告，请查看 {log}，退出码 {done.returncode}')
result = read(out / 'compile.json')
print(json.dumps({n: result['cases'].get(n, {'status': 'NOT_REPORTED'}) for n in names},
                 ensure_ascii=False, indent=2))
raise SystemExit(done.returncode)
PY
```

该对照已由用户执行并回传失败。命令核对 AIR／OMG 哈希并复用旧 TBE 设置；实际加载库及继承环境仍须用新旧 `omg-environment.json` 核对，不能仅靠 DDK 路径相同断言环境完全相同。若旧报告的参数不是上述组合，命令停止，避免把多项变化误当作单因素实验。

运行结束后，无论成功或失败，读取新构建路径与结尾；不再回传全部 158 项的 `NOT_SELECTED`：

```bash
for CONVSTATE_CASE in l00_c00_convstate_update_o0 l00_c01_convstate_update_o0; do
  CONVSTATE_LOG="$CONVSTATE_PROBE/models/$CONVSTATE_CASE/omc/compile.log"
  grep -n -E 'linear_attention_convstate_update_graph_builder|executor_linear_attention_convstate_update|LLMLinearAttentionConvStateFusionPass|OpBuilderFactory::CreateOpBuilder|Get format configs failed|OMG generate offline model' "$CONVSTATE_LOG"
  tail -n 20 "$CONVSTATE_LOG"
done
```

</details>

<a id="下一步读取构建策略与图上下文"></a>

### 已完成：读取构建策略与图上下文

用户已回传 `convstate-route-inspect-002`，状态为 `INSPECTED`，未重新编译或执行设备。参考为成功工作目录的 `target/dump_graph`；两个失败小图仍为 `convstate-default-format-001` 的 L0 Prefill／Decode o0。

读取工具先前按所选 DDK 的 HIAI schema 解码时失败；本次用保留的 AIR producer GE schema 取得唯一兼容解释。HIAI schema 仍在 AttrDef 字段 15 报 wire 类型冲突。报告明确标注 `STRUCTURAL_COMPATIBILITY_ONLY; REFERENCE_PRODUCER_UNVERIFIED`：这解决了本次读取，不能反推出历史 producer 身份，也不证明此前 dump 损坏。

| 项目 | 成功四层整图 | 失败小图 | 当前结论 |
|---|---|---|---|
| 图属性 | `_executor_type=1` | `_executor_type=1` | 该值没有提供差异，不直接改枚举 |
| 保留图的 model 属性 | memory／weight／stream／event 均 0 | 同样均 0 | 这些已展示属性没有提供差异 |
| ConvState 输入 dtype／layout | FP16／FP16／INT32，ND | 相同 | 仅源描述一致，不等于已核对编译物理存储 |
| x1 序列长度 | 日志为 T64；源中间 shape 为空 | T256／T1 | 长度仍是需要控制的变量 |
| x2／x3 shape | `[1,8192,4]`／`[128]` | 相同 | 已展示维度一致 |
| 输出源描述 | y1／y2 shape 均为空 | 同样为空 | 两边后续 Infer 均成功，不能仅凭空描述归因 |
| 输入生产者 | `Reshape_8`、`RefData_0`、`arg14_1` | 三个普通 Data | 状态输入语义和上游上下文不同 |
| 输出消费者 | Conv2D 与 NetOutput；原源码有 `conv_state.copy_` | 所选 y1 直接 NetOutput | 需要核对双输出存活与状态回写；不凭邻接表认定存在 Assign |
| 构建日志 | `pre_graph_save_hybrid_impl`，专用 ConvState builder／executor | single 实现路径，`strategyType:0`，format config 失败 | 构建路径确有不同；具体选择条件仍未证实 |

参考图有三个 ConvState，对应 RefData_0／2／4、Reshape_8／43／78、Conv2D／1／2。详细端口与邻居在 `report.json`；源算子的属性仍是输入／输出名称与索引映射。`DefinePatterns` 只证明融合规则定义，不能据此宣称命中融合。

**下一步最小对照：先恢复原图状态语义，再逐项增加消费者。** 保持同一 DDK、输入数据／dtype、编译参数和所比较图的 T 不变：

1. 现有普通 Data、单输出图作为已失败基线。
2. 普通输入，保留 ConvState 的两个输出，检查是否仅输出存活关系改变了构建路径。
3. 在第 2 项上加入原源码的 `conv_state.copy_(new_state)`，由原导出流程生成状态引用关系，再核对 RefData 和输出绑定；不手工将 Data 字符串改名。
4. 若仍失败，再加入原图卷积消费者；只在匹配的 T 下比较。必要时单独做 T64 对照，不把 T64 成功推广到 T256／T1。

以上是待实现／待执行的编译诊断，不是新的编译成功结果。状态引用或双输出会改变接口，不能直接套当前普通 Data、单输出的上板验证合同。达到专用 builder／executor 并生成 OMC 后，仍须核对两个输出及状态回写的数值。当前 12 项仍编译阻塞；已完成的格式参数试验和参考图读取不再重复。
