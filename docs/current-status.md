# 当前阶段测试命令

当前优先增量Draft：六层12份KV、17输入／14输出，候选档位15／7／3／1。Target精度独立调试，暂不开发批量verify。每个Draft阶段通过完整板端流程检查接通，不以整体接受率作为Draft独立精度结论。

## 整理为公共产物

已有公共目录可直接执行下面命令。脚本自动选择唯一的增量AIR，去掉三类模型的`current/`和Draft档位目录，更新运行配置；Prefill、Decode的权重统一为`weight/`，同步改写AIR内部引用。只同盘移动，不复制大权重或重编OMC。执行前确认板端空闲；中断后重复同一命令。

```bash
git -C "$REPO_DIR" pull --ff-only

python3 "$REPO_DIR/targets/cpro/scripts/organize_run_directory.py" \
  --root "$AI_RUN_DIR" --clean-completed --confirm-idle --apply
```

尚未迁入公共目录时，加`--draft-air /实际路径/增量AIR目录`。存在多个AIR时也需明确选择。未选定的成功构建和未知文件不会自动删除。

```text
$AI_RUN_DIR/
  prefill/air/{export.air,weight/}   prefill/omc/   prefill/tests/
  decode/air/{export.air,weight/}    decode/omc/    decode/tests/
  draft/air/{draft.air,manifest.json,draft/}
  draft/omc/{model.omc,manifest.json,...}
  draft/source/prepared/          # 后续构建所需权重
  draft/tests/compile/            # 固定编译临时目录，成功发布后删除
  draft/results/profiling/        # 已采集CSV
  common/embedding/  common/tokenizer/
  runtime/                       # runtime.json、runtime-assets.json、native/
  fused/                         # 整体流程run/result固定目录
```

档位保存在manifest中，不另建目录。`draft/air/draft/`保存生成AIR所需的图配方和序列化代码，属于产物内容。已有S64模型暂在`draft/omc/model.omc`；新模型编译、精度策略和ABI检查成功后发布，编译失败不替换它。首次新模型板端闭环成功回收前，保留唯一临时回退`.omc-previous/`；成功回收后自动删除。编译检查不代表板端数值精度通过。

## 重新编译K15

上次AIR已生成，编译因空间不足停止，尚无可用的增量OMC。清理空间后直接复用AIR，不需要重新导出或复制原权重。以下使用整理后的目录，编译产物固定放在`draft/omc`。当前预检至少需要6,685,841,408字节（约6.23GiB）；编译临时文件还可能占用额外空间。

```bash
git -C "$REPO_DIR" pull --ff-only

export MAX_DRAFT_TOKENS=15
export INCREMENTAL_AIR="$AI_RUN_DIR/draft/air"
export INCREMENTAL_COMPILED="$AI_RUN_DIR/draft/omc"

df -h "$AI_RUN_DIR"

PYTHONDONTWRITEBYTECODE=1 python3 "$REPO_DIR/targets/cpro/scripts/compile_incremental_draft.py" \
  --air-export "$INCREMENTAL_AIR" \
  --replace-failed --output "$INCREMENTAL_COMPILED"
```

失败信息保存在`draft/tests/compile/manifest.json`，用同一条命令加`--replace-failed`重试，不创建版本目录。成功后只保留公共OMC和必要元数据，临时编译目录删除。

编译成功后检查公共产物：

```bash
python3 - <<'PYCODE'
import json, os
from pathlib import Path
p = Path(os.environ['INCREMENTAL_COMPILED']) / 'manifest.json'
d = json.loads(p.read_text())
keys = ('status', 'stage', 'context_rows', 'cache_capacity', 'max_draft_tokens',
        'model_compilation', 'compiled_precision', 'compiled_abi', 'error')
print(json.dumps({k: d.get(k) for k in keys}, ensure_ascii=False, indent=2))
PYCODE
```

需要`DRAFT_INCREMENTAL_OMC_COMPILED_ABI_BOUND`、`compiled_precision=PASSED`和`compiled_abi=PASSED`才继续。运行入口从同目录manifest读取当前档位、模型缓存标识和新ABI，不再套用旧S64 ABI。成功后测试直接复用公共产物。

## 更新共享运行器

本次有KV运行器改动，编译一次放到公共目录。后续测试复用此目录，不在测试目录重复构建；以后修改C++运行器时再执行此命令。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$AI_RUN_DIR/runtime/native" --replace
```

## 新KV Draft完整文字闭环

前提：已导入`$AI_RUN_DIR/runtime/runtime.json`并完成目录整理，以上编译状态成功。保留原Prefill、Decode和词表，只替换本次Draft。首次接通仍限制prompt＋生成≤64；Draft容量192，按16行分块补上下文。

```bash
export RUN_DIR="$AI_RUN_DIR/fused/run"
export RECOVERY_DIR="$AI_RUN_DIR/fused/result"
export MAX_DRAFT_TOKENS=15
export INCREMENTAL_COMPILED="$AI_RUN_DIR/draft/omc"

python3 "$REPO_DIR/targets/cpro/scripts/run_board_serial_dflash.py" \
  --config "$AI_RUN_DIR/runtime/runtime.json" \
  --incremental-draft "$INCREMENTAL_COMPILED" \
  --max-draft-tokens "$MAX_DRAFT_TOKENS" \
  --text '你好，请介绍一下自己。' \
  --max-new-tokens 8 --max-proposals-per-round 2 \
  --apply --detach --output "$RUN_DIR"
```

等待完成后回收同一请求。修复后重复同一组命令即可复测：已确认退出的旧测试会自动清除并同名重建，运行中的请求保留供回收。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_board_serial_dflash.py" \
  --recover "$RUN_DIR" --output "$RECOVERY_DIR"
```

通过条件：Prefill一次、Decode与Draft各加载一次、至少两轮提案、12路KV持续推进、返回文字／tokens和时延。接受率可以为0，不代表Draft精度或加速通过。`catchup_calls`只补上下文，`proposal_calls`生成候选；最新anchor和未接受候选不写入持久KV。

新版运行器将12路KV保留在SDK输入张量中，只回填新增上下文行，跳过hidden回读和每轮KV输出清零；无需为此重编OMC。回收结果的`draft_kv.state_storage`应为`SDK_INPUT_TENSORS_DELTA_WRITEBACK`，`cache_commit_bytes = cached_context_rows × 24576`，`cache_host_staging_bytes`与`hidden_readback_bytes`均为0。`cache_commit_us`单独记录回填墙钟，不包含在`native_api_latency.draft`中。这里减少的是运行器拷贝，图内仍输出完整KV；板端精度和时延待本次测试。

## 查看已有profiling CSV

旧S64 CSV已回收，不需要重新采集或自动解析。整理后原始CSV保存在`draft/results/profiling/`。

```bash
find "$AI_RUN_DIR/draft/results/profiling" -type f -name '*.csv' -print
```

该次最大记录：入口投影Convolution 282.430ms、logits MatMul 19.912ms、CPU Top1 13.058ms。属于独立工具进程的S64诊断，不能与文字闭环时延直接相加或据此承诺50ms。
