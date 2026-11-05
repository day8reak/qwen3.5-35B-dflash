# 当前结果

更新：2026-11-05。以下设备/编译结果来自用户回传，未在本地重新运行。当前操作见[上板验证命令](current-status.md)。完整 Draft 的数值门槛仍未通过，尚无配对端到端加速结论。

## 自定义 Top1

| 项目 | 当前结果 |
|---|---|
| 算子与接口 | StableArgMax，有限FP16 `[1,15,248320]` → INT32 `[1,15]`，并列取最小索引 |
| 目标 | kirinx90，CPro DDK 100.520.020.010 |
| 构建 | allops生成/安装包装文件、GE AIR→OMG编译通过 |
| 图层面归属 | NPUCL=1；CPU操作=0；主图遗留计算=0 |
| 设备数值与稳定性 | 七组全部通过；3次预热后每组10次，70次／1050个INT32结果精确一致、重复稳定 |
| 同步API墙钟 | 七组等权总均值1.232ms；全部调用范围0.922–1.695ms |
| 模型集成 | 未接入完整Draft；TopK和310P未实现 |

当前OMC：`draft/tests/stable-arg-max-ge-config/omc/compile-work/generated/model.omc`。
SHA256：`2e76eda684ae0a4ac201af1c1ba511ca6b84e864a698193cfd1cfdfc1de2598f`。

| 输入组 | 均值 ms | 最小–最大 ms |
|---|---:|---:|
| index_boundaries | 1.1119 | 1.083–1.136 |
| ties_across_boundaries | 1.1874 | 0.922–1.315 |
| zeros_equal_extremes | 1.2344 | 1.072–1.639 |
| signed_subnormal | 1.4330 | 1.260–1.695 |
| adjacent_representable | 1.1799 | 1.143–1.241 |
| random_seed350248320 | 1.2852 | 1.163–1.544 |
| finite_binary16_alphabet | 1.1911 | 1.097–1.434 |

本任务已完成回收，native/archive/各case退出码均为0，锁已释放。计时为同步API墙钟，调用间快照不计入，不是纯NPU内核时间；不能与旧CPU算子profiling直接相除，或从完整Draft时延中扣减。当前冻结Qwen配置作为正确性基线，下一步补17×8193尾部边界。

身份和证据范围见[算子验证记录](../targets/cpro/ops/draft-top1/docs/VALIDATION.md)。

## 完整 Draft 候选

C16／cap192／K15、host embedding候选：板端加载和执行已完成；isolated／feedback各19例仍全部未通过原严格数值门槛，有效Top1分别82/86、83/86。缓存保留检查通过，重复输出稳定。host查表准备不计入下表API。

| 同步API均值 | 上一双NPU轮 ms | 单NPU主体／host embedding ms |
|---|---:|---:|
| isolated catchup，11次 | 192.067 | 103.958 |
| isolated proposal，8次 | 191.469 | 103.998 |
| feedback catchup，11次 | 200.515 | 106.090 |
| feedback proposal，8次 | 189.147 | 104.994 |
| 38次加权均值 | 193.772 | 104.802 |

均值下降45.9%，但不是含host查表的端到端结果，也不是固定输入3预热＋10次的稳态验收。目标是常驻预热后proposal同步Process均值≤50ms，尚未达到。

该候选另一次profiling：model `inference` 为 **103.720ms**，算子duration合计73.656ms，其中末尾MatMul 19.960ms、CPU ArgMax 12.353ms、Conv 8.643ms。算子可能重叠，不能相加当关键路径或把差额直接归于搬运。load 160.281ms、unload 4.859ms另计。此模型尚未集成新Top1。

## 已有端侧闭环

旧接线模型已完成文字编码→Prefill→常驻Draft/Decode→文字解码。以下为同次旧闭环结果，不能与上述新候选的独立API测量混作一组。

| 项目 | 观察值 |
|---|---:|
| Prefill同步Run，1次 | 450.087ms |
| Decode同步Run，8次均值 | 109.617ms |
| Draft同步Process，8次均值（1补齐＋7proposal） | 608.155ms |
| Prefill加载 | 10.300s |
| 完整请求墙钟 | 32.569s |
| 接受／实际考虑候选 | 0/7，0% |
| 纠正／Bonus | 7/0 |
| 输出 | 8个token；文本为 `=3'!=%3%` |

流程接通不代表语言质量通过。Target精度仍在调试，普通Decode与DFlash的配对加速未测得。同步API、加载及完整请求计时存在不同范围，不能相加。

## 内置 Top1 路线的保留结论

21项形状/组合图已编译；后续9个OMC共61例、610次测量调用完成。原含NaN合同为45例通过、16例失败，输出重复稳定。四个小NPU ArgMax的非NaN场景通过，混合NaN优先级不符；完整组合仍有全量CPU归约。

| 原fixture | 同步API均值 ms | 范围／限制 |
|---|---:|---|
| 小NPU ArgMax四种shape | 0.4993／0.6210／0.6964／0.5957 | 局部归约，不能推断全词表收益 |
| CPU ArgMax `[7275,512]` | 7.4633 | 局部归约，并非最终完整Top1 |
| tiled／flat／transposed／485宽 | 19.8443／19.5646／40.0270／19.6726 | 完整词表混合图，均有两行NaN失败 |

后续TopK、INT32选择/合成、比较树及Bitcast路线未得到完整词表纯NPU候选，已停止扩张。NaN来自人工fixture，不证明真实Draft异常；新算子以有限FP16为独立合同，旧失败结果不改写。

逐次排障和旧测量保留于[整理前记录](https://github.com/day8reak/qwen3.5-35B-dflash/blob/e1b00b5/docs/results.md)；机器状态见[执行状态](../specs/dflash-execution-state.json)。
