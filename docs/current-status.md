# 当前阶段：StableArgMax 上板验证

2026-11-05：自定义 Top1 已在 CPro DDK `100.520.020.010` 编译通过。图层面只有一个 NPU 子图，无 CPU 或主图遗留计算；设备数值与时延尚未验证。输入 `FP16[1,15,248320]`，输出 `INT32[1,15]`，有限值、并列取最小索引。

当前产物为 `$AI_RUN_DIR/draft/tests/stable-arg-max-ge-config`，**直接复用，不再导出或编译**。本轮文档更新不修改算子和验证脚本。完整 Draft 尚未接入该算子，现有精度限制见[结果](results.md)。

## 用户现在执行什么

在已有 DDK/HDC 环境的服务器运行。`REPO_DIR` 指向模型仓库；加载项目配置后，`AI_RUN_DIR`、`CPRO_DDK_ROOT` 沿用原值。`VECTOR_CONFIG` 使用此前 Top1 验证成功的设备配置；未设置时默认公共 `runtime/runtime.json`，若此前使用别处配置则填其实际路径。

```bash
source "$REPO_DIR/targets/cpro/scripts/env.sh"
export VECTOR_CONFIG="${VECTOR_CONFIG:-$AI_RUN_DIR/runtime/runtime.json}"
test -r "$VECTOR_CONFIG"

python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --compiled "$AI_RUN_DIR/draft/tests/stable-arg-max-ge-config" \
  --ddk-root "$CPRO_DDK_ROOT" --config "$VECTOR_CONFIG" \
  --output "$AI_RUN_DIR/draft/tests/stable-arg-max-validation" --apply
```

默认执行全部七组输入，每组预热3次、实测10次。覆盖大索引、跨块并列、零与极值、subnormal、相邻FP16、随机和全部有限binary16模式。

启动确认后，回收同一次请求：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --recover "$AI_RUN_DIR/draft/tests/stable-arg-max-validation" \
  --output "$AI_RUN_DIR/draft/tests/stable-arg-max-validation-result"
```

若返回 `PENDING`，继续回收原执行目录，不重复 `--apply` 或删除锁。成功标准：七组全部完成、每组十次 INT32 逐字节一致且重复稳定。API墙钟与纯NPU时延分开解释；通过后再安排其他shape与整模型接入。

[算子构建与Torch调用](../targets/cpro/ops/draft-top1/README.md) · [当前结果](results.md)
