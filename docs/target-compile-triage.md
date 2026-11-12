# Target 编译：本轮 90 成功／68 失败

2026-11-12 用户回传 `target/tests/all-boundaries/omc/compile.json`：158 个输出图全部遍历，90 COMPILED、68 FAILED；Prefill／Decode 各 45 成功、34 失败。DDK `100.520.020.010`，`isolated-kirinx90`，`use_origin_format=true`。随后 GDN 日志明确报出 `RuntimeError: te_update_version() return error.`；补回兼容参数后的编译结果尚未回传。

| 类别 | Prefill 失败 | Decode 失败 | 与旧批相比 |
|---|---:|---:|---|
| QLinear／QuantBatchMatmulV4444 链路 | 19 | 19 | 38 项此前可编译 |
| GDN／ChunkGatedDeltaRule，两输出 | 6 | 6 | 12 项此前可编译 |
| CacheUpdate K/V | 2 | 2 | 4 项此前可编译 |
| AdnFusedInferAttention | 1 | 1 | 2 项此前可编译 |
| LinearAttentionConvstateUpdate，两输出 | 6 | 6 | 12 项此前也失败；本轮错误是否相同待查 |

90 个成功项是 Norm、Conv1d、SiLU、decay/beta、repeat Q/K、门控 Norm、回写值转换、两次残差、RoPE、attention layout/gate。可以继续[批量上板](target-batch-rerun.md#4-批量上板)，但它们不覆盖本轮失败的高危算子。

## 已定位：GDN 的 TE worker 平台更新失败

本次 GDN 已找到 custom 注册、匹配输出格式并得到 tiling；FP32→FP16 的 CastT prebuild 也成功。真正退出链为 `_prerun → _update_options → te_update_version → RuntimeError → worker died`。日志显示初始化返回 `Ascend310B1`，随后该任务更新请求为 `kirinx90`，与项目此前完成的名称兼容诊断吻合。不能仅凭内部 SOC 名或缺失 schedule JSON 把 OMG 目标改成 Ascend310B1。

**全批操作页遗漏了原来已使用的 `--tbe-diagnosis`，现已补回。** 它复用已有诊断，在运行目录复制并适配 DDK TE wheel 的底层 update 参数，保留真实 worker 审计，OMG 仍编译 kirinx90。仅设置环境变量不会自动启用此参数；也不能从 OMG 命令行没有这个字符串推断未启用，因为它是 Python 编译入口参数。

QLinear／CacheUpdate／Attention 是否同因尚未有对应日志；先用相同 AIR 做代表项重试。源码复查提交 `889f497` 没有改变 OMG 编译入口或 CacheUpdate 计算表达式。旧数值和源码／CPU 检查均不代表新 OMC 精度通过。

## 当前执行：复用 AIR 重试五项

在 35 服务器原 custom_pypto／DDK 终端执行。`TBE_DIAGNOSIS` 使用此前成功的完整诊断目录，整理后的默认位置为 `common/tbe-diagnosis`；若尚未搬迁，先设置为原真实目录。无需重采集、重导 AIR、重跑诊断矩阵或修改安装目录。此次输出是独立的五项编译对照，原 90 个成功 OMC 保留。

```bash
export REPO_DIR="/home/w00949577/qwen3.5-35B-dflash"
export AI_RUN_DIR="/home/w00949577/qwen3.5-35B-dflash-run"
export BATCH_ROOT="$AI_RUN_DIR/target/tests/all-boundaries"
export OMC_BATCH="$BATCH_ROOT/omc"
export CPRO_DDK_ROOT="${CPRO_DDK_ROOT:-/home/l00886453/ddk_0806/image/hiai_ddk/ddk_internal}"
export CPRO_DDK_ID="${CPRO_DDK_ID:-100.520.020.010}"
export TBE_DIAGNOSIS="${TBE_DIAGNOSIS:-$AI_RUN_DIR/common/tbe-diagnosis}"
export TBE_PROBE="$BATCH_ROOT/omc-tbe-check"
git -C "$REPO_DIR" pull --ff-only origin main

(
set -e
test -f "$TBE_DIAGNOSIS/report.json"
test ! -e "$TBE_PROBE"
compile_rc=0
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" compile \
  --batch "$BATCH_ROOT/air" --output "$TBE_PROBE" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --compiler-profile isolated-kirinx90 --use-origin-format \
  --tbe-diagnosis "$TBE_DIAGNOSIS" \
  --cases l03_c01_self_attn_k_proj_o0 \
          l00_c00_gdn_o0 l00_c01_gdn_o0 \
          l03_c01_cache_k_o0 l03_c00_fused_attention_o0 || compile_rc=$?

python - "$TBE_PROBE" "$compile_rc" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
r = json.loads((root / 'compile.json').read_text())
print('batch:', r['status'], 'exit:', sys.argv[2])
for name in r['selected_cases']:
    case = r['cases'][name]
    path = root / 'models' / name / 'omc/report.json'
    d = json.loads(path.read_text()) if path.is_file() else {}
    audit = (d.get('tbe_compatibility') or {}).get('audit', {}).get('status', 'NOT_RECORDED')
    print(name, case['status'], 'stage:', d.get('stage'), 'TBE audit:', audit)
    if case['status'] != 'COMPILED':
        print(case.get('error', 'see compile.log'))
PY
)
```

回传最后五行状态与 TBE audit。`APPLIED` 表示记录到适配；还需 `COMPILED` 才是编译通过。`NOT_OBSERVED` 不证明适配已生效，需检查 worker 事件。其余 153 项在这个对照报告中是 NOT_SELECTED，并非新失败，也不会被当作已覆盖。五项通过后再恢复其余失败项；本页不预报 56 项全部恢复或 ConvState 已修复。

若重试仍失败，使用下列只读命令提取新日志，不再次编译。输出目录须不存在：

```bash
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" inspect-compile \
  --batch "$TBE_PROBE" --failed --ddk-root "$CPRO_DDK_ROOT" --compiler-entry \
  --output "$BATCH_ROOT/compile-inspect"
cat "$BATCH_ROOT/compile-inspect/summary.txt"
```

完整日志为 `$TBE_PROBE/models/<case>/omc/compile.log`。检查报告同时保留源码／权重绑定、内部描述与 Cast 链、实际编译环境；当前 shell 环境另记，不冒充历史进程环境。源 AIR 描述不是编译后存储证明。

## 旧批 ConvState：成功整图提供的对照

用户成功的四层 T64 日志已显示 ConvState 专用 graph builder、`executor_linear_attention_convstate_update` 内核及 `OMG generate offline model success`。因此不能把小图失败解释为整套 DDK 没有该实现。

小图在 shape infer 成功后，进入 single 构建路径，最终因 `Get format configs failed ... substore: !` 退出。L0 Prefill／Decode 取消 `use_origin_format` 的对照仍然失败。成功整图也有 `convertFunc not exist` 和候选 store 不支持日志，单凭这些消息不能定根因。

## 旧批已读取的构建策略与图上下文

`convstate-route-inspect-002` 已完成只读检查，未新编译或执行设备。

| 项目 | 成功四层整图 | 失败小图 |
|---|---|---|
| 序列长度 | T64 | T256／T1 |
| 图属性 | `_executor_type=1` | 相同 |
| 源输入 dtype／layout | FP16、FP16、INT32；ND | 相同 |
| 状态／控制 shape | `[1,8192,4]`／`[128]` | 相同 |
| 输入生产者 | Reshape、RefData、Data | 三个普通 Data |
| 输出消费者 | 卷积与 NetOutput；源码含状态 copy_ | 单个选定输出接 NetOutput |
| 构建路径 | hybrid；专用 builder／executor | single；format config 失败 |

参考 IR 使用保留的 GE schema 得到唯一结构兼容解释；所选 DDK 的 HIAI schema 不兼容。该结果不证明历史 producer 身份。源输出 shape 两边均为空、后续 infer 均成功，不以空描述认定问题；没有从邻接表推断融合或 Assign。

下一步在相同长度和编译配置下，分别控制 RefData 状态语义、双输出存活、卷积消费者，寻找进入成功构建路径的最小条件。目前尚未证实是哪一项触发分流；12 项没有新编译成功，也没有精度结果。

原 38 项排障和已完成重导命令查阅 [Git 历史](https://github.com/day8reak/qwen3.5-35B-dflash/blob/459060e/docs/target-compile-triage.md)。
