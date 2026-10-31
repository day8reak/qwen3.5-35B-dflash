# 当前阶段测试命令

增量Draft的C16／缓存192／K15 AIR已在DDK主机导出成功：17输入、14输出、六层12份KV缓存，不复制原权重。新增17／14接口的常驻运行适配：12路KV保存在板端内存，按已提交上下文分块追加。主机模拟已覆盖四档候选；新OMC的实编结果与板端接通仍待回传。Target精度继续独立调试，新批量verify暂不开发。

## 采集当前S64 Draft算子耗时

复用`real-feature-draft-plan`的真实Prefill64特征和现有内嵌权重Draft；不重编模型或runner、不重新上传模型。启动前会核对旧输入和元数据的完整SHA256，在新输出目录内重新生成输入计划；允许内容相同的文件身份变化，不改旧报告。板端模型缓存须仍在。通过已观察到的`model_run_tool_internal --enable_item=1`，用V1执行一次，再由同目录`data_proc_tool_internal`转CSV。可与主机上的增量OMG编译并行；板上模型调用使用同一把排他锁。

```bash
git -C "$REPO_DIR" pull --ff-only

python3 "$REPO_DIR/targets/cpro/scripts/profile_draft.py" \
  --prepared "$AI_RUN_DIR/real-feature-draft-plan" \
  --tool-inspection "$AI_RUN_DIR/target-board-tool-inspection-writable" \
  --output "$AI_RUN_DIR/draft-s64-profile-v2" \
  --apply
```

返回`DRAFT_OPERATOR_PROFILE_STARTED`后，稍后回收同一次采集。断线也用下面命令，不重跑模型；若返回`PENDING`，等待后换一个新的回收目录再执行。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/profile_draft.py" \
  --recover "$AI_RUN_DIR/draft-s64-profile-v2" \
  --output "$AI_RUN_DIR/draft-s64-profile-recovery-v2"

find "$AI_RUN_DIR/draft-s64-profile-recovery-v2/prof_data" -type f -name '*.csv' -print
```

回收目录保留原始`prof_data/*_op.csv`、`*_model.csv`、工具日志、PID和退出码。脚本只回收CSV，不自动解析或汇总；完成后打印CSV目录及文件列表，可直接查看`total time(us)`列。厂商CSV可能含历史模型记录，需核对模型归属，因此状态为`DRAFT_PROFILE_CAPTURED_REQUIRES_ATTRIBUTION_REVIEW`。

本次`draft-s64-profile-recovery-v2`已回收到CSV，模型和厂商转换工具退出码均为0。旧脚本仅因模型CSV多出`model name`、`model id`列而报错，已有CSV可直接使用，无需重跑或重新回收。

用户回传的最大耗时记录为入口`draft.fc`卷积投影282.430ms，其次为logits MatMul 19.912ms、CPU Top1 13.058ms。下一步优先评估投影替代实现并继续增量KV接入；当前C16 AIR仍使用同一卷积表达，减少候选数不会消除这一投影。

这是新工具进程中的一次profiling诊断，输入是已有64行真实特征，不是18-token文字请求中的同一次Draft调用；也不是稳定时延分布或精度验收。算子耗时总和可能含重叠、嵌套执行；不能据此直接推算50ms。工具使用方法参考[官方示例](https://developer.huawei.com/consumer/jp/doc/hiai-Guides/examples-0000001053605660)，实际参数以本机internal工具帮助为准。

## 编译已经导出的K15

复用本次成功的`air-v1`目录，无需重新导出。脚本从原`full-draft/residual-fix`记录选择相同DDK，在新目录编译；原权重仅建立逐文件引用并检查前后完整性。至少预留两份Draft权重大小加1 GiB空间，供新编译权重、打包OMC与临时文件使用；编译器实际峰值可能更高。不要同时编译其他档位。

```bash
git -C "$REPO_DIR" pull --ff-only
export MAX_DRAFT_TOKENS=15
export INCREMENTAL_AIR="$AI_RUN_DIR/draft-kv-c16-cap192-k15-air-v1"
export INCREMENTAL_COMPILED="$AI_RUN_DIR/draft-kv-c16-cap192-k15-compiled-v2"

PYTHONDONTWRITEBYTECODE=1 python3 "$REPO_DIR/targets/cpro/scripts/compile_incremental_draft.py" \
  --air-export "$INCREMENTAL_AIR" \
  --output "$INCREMENTAL_COMPILED"
```

预期`DRAFT_INCREMENTAL_OMC_COMPILED_ABI_BOUND`，包含17输入、14输出。编译后检查FP32计算路径及KV形状／类型，再生成新`model/model.omc`。每30秒显示编译等待状态，完整日志保留在`compilation/compile.log`。失败时保留目录，不覆盖重试。

## 检查编译结果

编译成功后执行下面的新KV闭环。使用`--incremental-draft`绑定新目录，不能传给旧S64的`--draft-bundle`；运行时按编译后的实际输入槽位加载，检查SDK返回的形状和字节数。

```bash
python3 - <<'PYCODE'
import json, os
from pathlib import Path
p = Path(os.environ["INCREMENTAL_COMPILED"]) / "manifest.json"
d = json.loads(p.read_text())
keys = ("status", "stage", "context_rows", "cache_capacity", "max_draft_tokens",
        "model_compilation", "compiled_precision", "compiled_abi",
        "device_execution", "runtime_integration", "error")
print(json.dumps({k: d.get(k) for k in keys}, ensure_ascii=False, indent=2))
print("manifest:", p)
PYCODE
```

## 新KV Draft完整文字闭环

前提是上面的编译状态成功。若编译成功目录实际叫`compiled-v1`，把`INCREMENTAL_COMPILED`改为那个目录。第一次保通继续使用prompt＋生成≤64，先不扩展Target控制；Draft缓存容量为192。`max-draft-tokens`必须等于所编译档位，`max-proposals-per-round`仅限制本次串行验证数量。其他档位须分别导出并编译。

```bash
git -C "$REPO_DIR" pull --ff-only
export MAX_DRAFT_TOKENS=15
export INCREMENTAL_COMPILED="$AI_RUN_DIR/draft-kv-c16-cap192-k15-compiled-v2"
export TOKENIZER_DIR="${TOKENIZER_DIR:-$AI_RUN_DIR/tokenizer-offline/source}"
export RUN_DIR="$AI_RUN_DIR/dflash-kv-c16-k15-text8-run"
export RECOVERY_DIR="$AI_RUN_DIR/dflash-kv-c16-k15-text8-recovery"

python3 "$REPO_DIR/targets/cpro/scripts/run_board_serial_dflash.py" \
  --decode-job "$AI_RUN_DIR/target-pytorch-decode1-compile" \
  --runtime-probe "$AI_RUN_DIR/target-decode1-runtime-io-v2" \
  --state-lineage "$AI_RUN_DIR/target-decode1-state-lineage-v7" \
  --source-torchair-pb2 "$AI_RUN_DIR/target-decode1-row-response/ge_ir_by_protoc_3_19_pb2.py" \
  --incremental-draft "$INCREMENTAL_COMPILED" \
  --max-draft-tokens "$MAX_DRAFT_TOKENS" \
  --weight-table "$AI_RUN_DIR/original-embedding-tables/qwen3_5_moe_embdedding_weight.bin" \
  --scale-table "$AI_RUN_DIR/original-embedding-tables/qwen3_5_moe_embdedding_scale.bin" \
  --prefill-feedback "$AI_RUN_DIR/target-prefill64-decode1-feedback-plan-v2" \
  --tokenizer-dir "$TOKENIZER_DIR" --text '你好，请介绍一下自己。' \
  --max-new-tokens 8 --max-proposals-per-round 2 \
  --resident --draft-preread --apply --detach \
  --output "$RUN_DIR"
```

等待完成后回收同一次请求。新模型首次需要上传；原Prefill／Decode及权重缓存复用，只构建新版运行程序。中途不与服务器交换token或KV，不开启dump。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/run_board_serial_dflash.py" \
  --recover "$RUN_DIR" --output "$RECOVERY_DIR"
```

结果新增`draft_kv`：`cached_context_rows`为已经写入的上下文行数，`catchup_calls`为只补上下文的调用数，`proposal_calls`为生成候选的调用数。18-token prompt的首轮会执行16＋2两块，其中第一块不计提案轮次；后续只补新增已提交特征。`phase_latency.draft`统计所有原生调用，含追赶，不能直接当成稳定提案时延。最新anchor和未接受候选不写入持久KV。末轮提交的token不需要再给Draft提案，因此最终游标会落后于最终prefix。

本阶段通过标准：原Prefill一次、Decode与新Draft各加载一次、至少两轮提案、12路缓存持续推进、最终返回文字／tokens和时延。接受率可以为0；不代表Draft独立精度通过，也不宣称已加速。当前仍复制KV输入／输出，尚非零拷贝。
