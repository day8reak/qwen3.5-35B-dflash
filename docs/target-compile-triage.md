# Target 编译：剩余 ConvState 小图

累计 146/158 个边界输出图编译成功；Norm 20 项、Conv1d 6 项的导出问题已修正，剩余 12 项均为 LinearAttentionConvstateUpdate。最新全批命令见[批量验证](target-batch-rerun.md)，允许这些失败项保留，继续执行成功 OMC。

## 成功整图提供的对照

用户成功的四层 T64 日志已显示 ConvState 专用 graph builder、`executor_linear_attention_convstate_update` 内核及 `OMG generate offline model success`。因此不能把小图失败解释为整套 DDK 没有该实现。

小图在 shape infer 成功后，进入 single 构建路径，最终因 `Get format configs failed ... substore: !` 退出。L0 Prefill／Decode 取消 `use_origin_format` 的对照仍然失败。成功整图也有 `convertFunc not exist` 和候选 store 不支持日志，单凭这些消息不能定根因。

## 已完成读取构建策略与图上下文

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

## 新全批失败摘要

沿用[批量页](target-batch-rerun.md)的变量，完成编译后只读检查全部失败项；这不会再次导出、调用 OMG 或上板。输出目录须不存在。

```bash
python "$REPO_DIR/targets/cpro/tools/target_layer_probe.py" inspect-compile \
  --batch "$OMC_BATCH" --failed \
  --ddk-root "$CPRO_DDK_ROOT" \
  --output "$BATCH_ROOT/compile-inspect"
cat "$BATCH_ROOT/compile-inspect/summary.txt"
```

完整字段在同目录 report.json；逐项日志在 `$OMC_BATCH/models/<case>/omc/compile.log`。无需安装 rg。

原 38 项排障和已完成重导命令查阅 [Git 历史](https://github.com/day8reak/qwen3.5-35B-dflash/blob/459060e/docs/target-compile-triage.md)。
