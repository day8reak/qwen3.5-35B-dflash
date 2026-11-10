# 剩余 38 项 OMC：对照原 Target 定位编译失败

更新：2026-11-10。用户已回传 `inspect-compile --failed` 的完整 38 项摘要，三类失败各自一致。以下是本批实际观察，未重新执行 DDK／上板；编译计数仍为 120 成功、38 失败。

| 组别 | 本批证据 | 当前判断与处理 |
|---|---|---|
| 普通 Norm：20 项 | 全部仍有 `Add → RmsNorm`，无导出修正记录；`add_kernel.cpp` 报类型不支持，随后 `Add ProduceTiling failed`、`inputTensorPtr is NULL` | gamma 预计算修正没有进入本批 AIR。先重导这 20 项，不能把 Add 的失败归为 RmsNorm 不支持 |
| Conv1d：6 项 | 全部在 `Convolution` Verify 报 `Input[0] and input[1] must have the same dim num`，无导出修正记录／结构检查 | 确认本批仍存在卷积两输入 rank 不一致。用现有显式四维导出路径重导这 6 项，检查真实连接和 shape |
| ConvState：12 项 | 两阶段、两个输出均报 `Get format configs failed ... LinearAttentionConvstateUpdate, substore: !` | 已确定失败环节是 format config 获取；尚未确定是注册、实现选择、支持条件还是图边界造成。暂不重导／重编这一组 |

Conv 的类型计数中出现 `Unsqueeze` 不足以证明修正已生效：它可能只处理权重，必须检查 Conv2D 的两条输入边。ConvState 摘要中的 `substore: !` 没有提供有效 store 名称，不能据此认定整个 DDK 缺少实现。

三组都出现了 `libai_npucore_ascendc_kernel.so not valid`，但这条消息不能替代上述各组的实际终止原因；先与同批成功 GDN 的日志对照。`IR InferShape ... Success` 和 `PassManager::Run begin` 即使标成 E 级也不是失败证据。

**下一步：复用现有 capture，只重导 Norm／Conv 共 26 项，先编译 4 个代表项；ConvState 只提取两份代表日志。** 命令见本页后半部分。已成功的 120 个 OMC、上板输出和 golden 保留。

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

本轮 `manifest_sha256=9f433d0092c93f5c156a5c6ab59d6d603dcef338b8132dc28b1e0f77fcf1b1e2`、`source_batch=../../../air-batch`，与之前 AIR 批次相同。更新 Python 代码再执行 `compile` 不会更新既有 AIR。以下使用仓库中已实现的导出修正，不换 Norm 算子、不改捕获 dtype 或 golden。

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

后续上板分别以两个新 OMC 批次为输入即可，不手动拼接或改写旧 `compile.json`。这一步尚未执行，不能将 26 项记为编译成功。

## ConvState：只取两份代表日志

Prefill／Decode、o0／o1 目前都在同一 format config 环节失败。先取第 0 层两阶段的 o0；完整日志可看见候选实现拒绝和 store 选择，而当前摘要只保留了终止点。

```bash
rg -n -C 2 'LinearAttentionConvstateUpdate|CheckSupport|substore|subStore|format config' \
  "$OMC_BATCH/models/l00_c00_convstate_update_o0/omc/compile.log" \
  "$OMC_BATCH/models/l00_c01_convstate_update_o0/omc/compile.log"

rg -n 'GetKernelbinAddr|libai_npucore_ascendc_kernel' \
  "$OMC_BATCH/models/l00_c00_convstate_update_o0/omc/compile.log" \
  "$OMC_BATCH/models/l00_c00_gdn_o0/omc/compile.log"
```

第二条若没有匹配，`rg` 退出码 1 只表示没找到文字。若成功 GDN 也有同条库消息，就不能将其直接当作全部失败的共同根因；即使只有 ConvState 有，也需要继续核对该算子实际使用的 store／库。先依据日志定位到具体注册或支持条件，再读相应实现，保持共享 DDK 不变。只有实现选择已核对且证据指向独立图边界时，才尝试原 Target 的双输出＋卷积／状态写回连续子图。

只读工具此前已通过 18 项主机检查；本次只更新用户回传证据和执行命令。数值异常与已完成项继续见[状态表](target-accuracy-status.md)。
