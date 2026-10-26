# 一批验证剩余 CPU 候选

当前 target 已通过 ReshapeFusion，19个 row Mul 退出 CPU-only 候选列表，但主图仍因 NPUCL 分区数超过9回退。用户要求减少编译、上板和反馈往返，因此提供 `--suite remaining-cpu` 共50项，覆盖最新报告中的全部13类已识别 CPU-only 计算候选。用户已回传其中7项GDN结果，当前加 `--exclude-family gdn`，一次只跑其余43项，复用已有结果。

这是当前 tiny 图 shape、dtype 和上下文的对照批次，生产模型保持当前版本。它包含原形、可行的精确改写以及部分串联上下文；单项 OMG 或板端执行失败会记录并继续后续案例。它不重复历史 dense/reduction/row-Mul 等无关探针。前一版7项 GDN 探针保留在完整批次目录中；当前命令跳过这7项，避免重复编译和上板。

| 分组 | 当前候选 | 本批次检查 |
|---|---|---|
| GDN（7项，已有结果，本次跳过） | ConcatD、Sub | 普通/严格下三角原 Slice/Concat 与静态 Gather 复制；mask→Exp→mask；同形复制后一次 Sub |
| MoE 路由（16项） | Equal、bool Cast、整数 ReduceSum、Greater、bool BroadcastTo | 原整数/布尔控制、索引查表候选及路由串联上下文，保留选中但权重为零的语义 |
| 选择（15项） | TopKD、ArgMaxExt2 | tiny router TopK、全词表64项 ArgMax 的原形/布局对照及 tie 用例；部分 Softmax→TopK→归一化上下文 |
| 位置与注意力掩码（12项） | INT32 Cast、BatchMatMul、LogicalAnd、bool Reshape | RoPE K=1 外积的 BMM/二维 MatMul/Mul，位置转换，动态掩码转换/reshape/逻辑选择 |

这个表描述覆盖范围，不表示所有候选都能进入 NPU。纯复制/整数输出保持精确比较；FP16 算术沿用既有门槛，不放宽。使用查表的 attention mask 候选限二值输入；不能直接推广成任意非零整数掩码。ArgMax 首个最大值和 TopK 的 tie 行为分别检查，不能因唯一分数通过就推断 tie 已对齐。

## 已有批次断连后补跑

用户回传 `cpu-gaps-20261026-153933` / `cpu-gaps-board-20261026-153938`：43项中42项编译成功，23项完成加载、执行和数值比较。第24个已编译案例 `selection_argmax_rank2_ties` 的 hash-result 调用收到 `[Fail]Device not founded or connected`，批次状态为 `INFRASTRUCTURE_FAILED`。该项及后续共19项没有完成验证，不应判为算子执行失败。`moe_cast_bool_roundtrip` 是独立的编译失败，日志在 CastT v0 的 `InitKernelAttr` 报告缺少 TBE kernel name，未进入板端。

不用重新导出或编译。确认所选 HDC server 上原设备恢复连接后，复用原 `GE_CPU_GAPS_RUN`，从未完成项开始，在新目录补跑：

```bash
git -C "$REPO_DIR" pull --ff-only
"${HDC:-hdc}" -s "$BIP" list targets -v
export GE_CPU_GAPS_REMAINDER_RUN="$AI_RUN_DIR/cpu-gaps-remainder-$(date +%Y%m%d-%H%M%S)"
python3 "$REPO_DIR/targets/cpro/scripts/probe_hdc_runtime.py" \
  --probe-run "$GE_CPU_GAPS_RUN" --native-from "$OP_PROBE_RUN" \
  --refresh-runner --replay-all --start-at selection_argmax_rank2_ties \
  --output "$GE_CPU_GAPS_REMAINDER_RUN"
cat "$GE_CPU_GAPS_REMAINDER_RUN/runtime-comparison.txt"
```

`--start-at` 包含指定项，按原manifest顺序选择此项及其后的已编译案例，只刷新一次probe runner。它要求 `--replay-all`，起点必须是原批次中编译成功的案例；不会重新编译，也不会把旧通过结果搬进新报告。旧输出目录保留，重复使用会报 `FileExistsError`。新旧报告需要一起保留；如果已经开始全量回放，可让它完成，无需为了使用该选项再启动第二个回放。

这次结果还表明：MoE membership/count/active的Gather版本仍为编译CPU1；weights/gate/完整route的两类方案均为NPU1/CPU1。TopK rank2为CPU1，rank3/4为NPU2/CPU1，不能将多出的NPU分区当成TopK已经由NPU执行。RoPE BMM/二维MatMul及原形位置Cast显示NPU1/CPU0，但板端尚未验证。完整图仍未解除9分区上限，当前不晋级这些路由/选择替代方案。详见 `specs/remaining-cpu-probes-user-report.json`。

## 首次运行批次

在原服务器环境运行，继续复用 `CANN_ROOT`、`CPRO_DDK_ROOT`、`CPRO_DDK_ID`、`OP_PROBE_RUN`、`SN`、`BIP` 和原 SDK 配置：

```bash
git -C "$REPO_DIR" pull --ff-only
export GE_CPU_GAPS_RUN="$AI_RUN_DIR/cpu-gaps-$(date +%Y%m%d-%H%M%S)"
export GE_CPU_GAPS_BOARD_RUN="$AI_RUN_DIR/cpu-gaps-board-$(date +%Y%m%d-%H%M%S)"

python3 "$REPO_DIR/targets/cpro/scripts/probe_hdc_ops.py" \
  --suite remaining-cpu --exclude-family gdn --compile-only --output "$GE_CPU_GAPS_RUN" &&
python3 "$REPO_DIR/targets/cpro/scripts/probe_hdc_runtime.py" \
  --probe-run "$GE_CPU_GAPS_RUN" --native-from "$OP_PROBE_RUN" \
  --refresh-runner --replay-all --output "$GE_CPU_GAPS_BOARD_RUN"
```

这两步可以整段粘贴，无需逐项指定 `--case`；预计列出并编译43项。若将来需要完整50项，去掉 `--exclude-family gdn`。第一步独立编译每个案例，第二步仅刷新一次独立 probe runner，再依次执行所有编译成功的案例；未编译成功的案例被列为跳过。原 native donor 和历史结果不被修改。基础工具链/传输失败仍可能终止批次并保留记录，不能把未运行项算作通过。

完成后一次提供两份报告即可：

```bash
cat "$GE_CPU_GAPS_RUN/report.txt"
if [ -f "$GE_CPU_GAPS_BOARD_RUN/runtime-comparison.txt" ]; then
  cat "$GE_CPU_GAPS_BOARD_RUN/runtime-comparison.txt"
fi
```

可选查看完整案例清单（不启动编译或 HDC）：

```bash
python3 "$REPO_DIR/targets/cpro/scripts/probe_hdc_ops.py" \
  --suite remaining-cpu --exclude-family gdn --list-cases
```

保留当前 `GE_ROW_CHAIN_RUN` 和 inspect 报告。根据这批结果统一选择后续整图改写，再做一次完整图编译与双图执行验证。编译分区、Load、Process、数值比较及实际 NPU 运行位置分开记录。7个角色未确定的编译节点不被猜测成某个算子；全模型精度后置未豁免，DFlash 至少2倍提速尚未测量。


用户已回传的 `gdn-copy-20261026-150635` / `gdn-copy-board-20261026-150641`：四个Gather候选均为编译主图NPU1/CPU0，Load、Process和原数值比较通过。普通Slice/Concat控制也NPU1/CPU0且通过；严格Slice/Concat控制为NPU1/CPU1，Load失败、未执行；Exp串联控制为NPU1/CPU1且执行/数值通过。结果支持将Gather候选纳入后续整图改写，但不能据此认定整图分区限制已解除或实际NPU位置已证实。严格原形加载失败的底层原因尚未从这份报告确定，本轮不要求重跑这个旧控制。
