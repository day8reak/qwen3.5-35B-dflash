# 当前阶段：StableArgMax 尾部边界验证

2026-11-05：Qwen `[1,15,248320]` 的七组上板验证全部通过，70次输出逐字节一致且重复稳定，保留单NPU子图。同步API总均值约 **1.232ms**，范围0.922–1.695ms；完整Draft尚未接入。详情见[结果](results.md)。

Qwen基线保留在 `draft/tests/stable-arg-max-ge-config`，原任务已完成、锁已释放，无需重跑。下一步只补 `[17,8193]`：同时覆盖16＋1行输出、8192＋1元素尾块和未对齐行起点。复用当前算子，不重建allops；新shape需要一份新AIR/OMC，四组输入共用。

## 用户现在执行什么

在已有DDK/HDC环境的服务器执行。`REPO_DIR`为仓库；`CANN_ROOT`为此前使用的GE SDK；其他路径沿用项目配置。`VECTOR_CONFIG`保留刚才成功的设备配置，未设置时默认公共runtime配置。

```bash
source "$REPO_DIR/targets/cpro/scripts/env.sh"
export OP_DIR="$REPO_DIR/targets/cpro/ops/draft-top1"
export VECTOR_CONFIG="${VECTOR_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}"

python3 "$OP_DIR/scripts/stable_arg_max_ge.py" \
  --shape 17 8193 --ge-root "$CANN_ROOT" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --tbe-diagnosis "$AI_RUN_DIR/common/tbe-diagnosis" \
  --output "$AI_RUN_DIR/draft/tests/stable-arg-max-tail/compiled"
```

返回 `COMPILED` 后启动四组边界输入，每组3次预热＋10次测量：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --compiled "$AI_RUN_DIR/draft/tests/stable-arg-max-tail/compiled" \
  --ddk-root "$CPRO_DDK_ROOT" --config "$VECTOR_CONFIG" \
  --output "$AI_RUN_DIR/draft/tests/stable-arg-max-tail/run" --apply
```

启动确认后回收同一次请求：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --recover "$AI_RUN_DIR/draft/tests/stable-arg-max-tail/run" \
  --output "$AI_RUN_DIR/draft/tests/stable-arg-max-tail/result"
```

`PENDING`时继续回收原请求，不重复启动。通过条件：四组first_last/equal/tile_ties/subnormal的40次输出全部INT32精确一致、重复稳定且无执行错误。这只补该shape的边界证据，其他未测shape和整模型不继承通过。
