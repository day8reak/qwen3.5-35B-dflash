# 当前阶段：复现完整K/V行重复

2026-11-05：用户回传确认，RoPE候选的isolated case16（start162、valid16）在cache176重复了177行。全部12路K/V都满足 `isolated[176] == isolated[177] == feedback[177]`，逐位一致；feedback的176、177行不同。保留features局部14、15行有16381/16384个元素不同，因此输入文件本身不是重复行。完整精度仍FAILED，性能工作暂缓。[完整结果](results.md)。

源码共享Gather索引应分别为14和15。连经过RoPE的K都重复，支持优先检查共享行选择及后续Gather／搬运路径，尚不能确定具体算子或设备索引字节。CPU保存的文件不能证明NPU当时读到了相同数据。其余15行的数值误差仍未通过原门槛。

本次新增一个明确开启的验证选项，复用同一OMC、夹具和参考：先原样执行38次isolated／feedback调用，再在同一次模型加载后追加case16的10对调用。每对先重写全部17路输入，再原样复用输入，两次均不提交输出KV；调用前后记录全部输入的CPU缓冲区SHA256并与保留文件核对。输入回读异常即停止追加部分，保留原回放和已完成的追加输出。

## DDK服务器执行

只重建native运行器，不编译模型。保留原 `output-init/result`、`rope-fp32/result` 以及它们引用的fixture和OMC。以下命令沿用已有SDK资源配置。

```bash
git -C "$REPO_DIR" pull --ff-only
export ROPE_DIR="$AI_RUN_DIR/draft/tests/top1-integration/rope-fp32"
export KV_REPEAT_DIR="$ROPE_DIR/kv-reproduction"

python3 "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$AI_RUN_DIR/runtime/native" --replace

python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --replay "$ROPE_DIR/result" --config "$AI_RUN_DIR/runtime/runtime.json" \
  --output-initialization untouched --kv-reproduction-case 16 \
  --output "$KV_REPEAT_DIR/run" --apply
```

启动后只恢复同一请求；返回PENDING时继续使用这条恢复命令，不重新执行上面的启动命令。

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --recover "$KV_REPEAT_DIR/run" --output "$KV_REPEAT_DIR/result"
```

恢复会自动比较原38次验收及追加20次输出。原验收与追加诊断分开报告；所有数值比较仍使用原golden、atol/rtol和未写区逐位保持门槛。追加部分的详细报告在 `result/kv-reproduction.json`。回传以下摘要即可，无需重新运行离线行匹配或打包文件：

```bash
python3 - "$KV_REPEAT_DIR/result/result.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    r = json.load(f)
print(json.dumps({k: r[k] for k in ('status', 'stage', 'error', 'model',
    'numerical_validation', 'lock_state', 'kv_reproduction') if k in r}, ensure_ascii=False))
PY
```

摘要包含每次重写／复用调用是否重复整行、是否与同策略前一次逐位一致、是否匹配本次原isolated／feedback新增KV，以及原数值门槛结果。CPU回读可能改变可见性和时序；20次稳定不能证明所有运行历史都正确，也不能证明内部Gather索引正确。没有新增输出清零、模型精度改写或验收放宽。
