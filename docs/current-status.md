# 当前阶段：Draft 精度定位与 profiling

2026-11-05：StableArgMax单算子两种shape、110次精确稳定验证通过。完整Draft候选已执行并回收，38次同步API加权均值约66.55ms；数值仍失败。Top1保持isolated 82/86、feedback 83/86，但isolated首例hidden相对L2从旧候选1.45%升到65.11%，需单独定位。详情见[结果](results.md)。

复用 `draft/tests/top1-integration` 的模型、输入与快照。下面精度分析只读服务器文件，可以与一次板端profiling同时进行；不重新导出、编译或运行整套精度测试。profiling按用户要求在数值失败状态下用于热点诊断，不作为性能验收。

## 用户现在执行什么

DDK／HDC服务器先更新仓库；`REPO_DIR`、`AI_RUN_DIR`及设备环境沿用项目配置。保留本轮run/result、compiled以及板端模型缓存，完成分析前不清理。

```bash
source "$REPO_DIR/targets/cpro/scripts/env.sh"
export TOP1_DIR="$AI_RUN_DIR/draft/tests/top1-integration"

# 只读：比较已回收的首例isolated/feedback全部14路输出及逐行hidden。
python3 "$REPO_DIR/targets/cpro/tools/validate_incremental_draft.py" \
  --diagnose "$TOP1_DIR/result" > "$TOP1_DIR/diagnosis.json"

# 只读：核对同一OMC的CPU操作、NPU边界及模型身份；不调用OMG。
python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --inspect-boundaries "$TOP1_DIR/compiled" \
  --accuracy-result "$TOP1_DIR/result" --ddk-root "$CPRO_DDK_ROOT" \
  --output "$TOP1_DIR/boundaries.json"
```

`diagnosis.json` 的 `initial_call_comparison` 列出各输出是否逐位相同、hidden差异位置及两种模式相对golden的逐行误差。运行器在两种模式的第0例重设全部输入；文件检查不等于设备输入回读，也不能单凭差异断言kernel或buffer根因。原容差和失败结果保持。

另一个已加载相同环境的终端可直接启动profiling，无需等待上述离线分析：

```bash
export TOP1_DIR="$AI_RUN_DIR/draft/tests/top1-integration"

python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --accuracy "$TOP1_DIR/result" --allow-numerical-failure \
  --modelmanager-itf V1 --output "$TOP1_DIR/profile-run" --apply

python3 "$REPO_DIR/targets/cpro/tools/profile_draft.py" \
  --recover "$TOP1_DIR/profile-run" --output "$TOP1_DIR/profile-result"
```

默认使用原fixture的case 2（partial-context，15个proposal、参考KV），复用已部署OMC，只发输入。脚本核对所选工具help后执行一次 `enable_item=1`，随后转换并回收原始 `*_op.csv`／`*_model.csv`；出现PENDING继续回收同一请求。这是新建工具进程、无预热的一次诊断采集，与常驻API均值口径不同。CSV需按本次PID／模型归属筛选，算子耗时合计不当作关键路径。

回传diagnosis中的首例比较、边界摘要与profiling回收摘要即可；CSV可粘贴模型事件和主要耗时行，无需打包文件。重点查看StableArgMax、末尾MatMul、Conv、布局转换及CPU分段。新热点与精度问题分别判断，当前候选不晋升生产模型。首次构建方法见[完整流程](run-end-to-end.md)。
