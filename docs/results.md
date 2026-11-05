# 当前结果

更新：2026-11-05。以下设备/编译结果来自用户回传，未在本地重新运行。当前操作见[精度定位与profiling](current-status.md)。完整Draft数值门槛仍未通过，尚无配对端到端加速结论。

## 自定义 Top1

| 项目 | 当前结果 |
|---|---|
| 算子与接口 | StableArgMax，有限FP16 `[1,15,248320]` → INT32 `[1,15]`，并列取最小索引 |
| 目标 | kirinx90，CPro DDK 100.520.020.010 |
| 构建 | allops生成/安装包装文件、GE AIR→OMG编译通过 |
| 图层面归属 | NPUCL=1；CPU操作=0；主图遗留计算=0 |
| 设备数值与稳定性 | 七组全部通过；3次预热后每组10次，70次／1050个INT32结果精确一致、重复稳定 |
| 同步API墙钟 | 七组等权总均值1.232ms；全部调用范围0.922–1.695ms |
| 模型集成 | 完整Draft候选已执行回收，数值失败；单算子通过不代表整图精度通过；TopK和310P未实现 |

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

本任务已完成回收，native/archive/各case退出码均为0，锁已释放。计时为同步API墙钟，调用间快照不计入，不是纯NPU内核时间；不能与旧CPU算子profiling直接相除，或从完整Draft时延中扣减。Qwen配置已冻结为正确性基线。

补充尾部配置 `[17,8193]` → INT32 `[17]` 已通过：四组共40次／680个索引精确一致、重复稳定，NPUCL=1，所有退出码0且锁已释放。覆盖16＋1行输出、8192＋1元素尾块和未对齐行起点。

| 尾部输入组 | 均值 ms | 最小–最大 ms |
|---|---:|---:|
| first_last | 0.5562 | 0.448–0.677 |
| equal | 0.5037 | 0.434–0.886 |
| tile_ties | 0.4810 | 0.437–0.512 |
| subnormal | 0.5189 | 0.438–0.657 |

尾部API等权均值0.515ms；结果在 `draft/tests/stable-arg-max-tail/result`。两轮累计11组、110次／1730个索引通过，shape不同，不混算加速比。尾部回传摘要未包含OMC digest；shape按本阶段编译命令关联。

身份和证据范围见[算子验证记录](../targets/cpro/ops/draft-top1/docs/VALIDATION.md)。

## 完整 Draft 候选

C16／cap192／K15、host embedding候选：板端加载和执行已完成；isolated／feedback各19例仍全部未通过原严格数值门槛，有效Top1分别82/86、83/86。缓存保留检查通过，重复输出稳定。host查表准备不计入下表API。

| 同步API均值 | 原生Top1／host embedding ms | StableArgMax候选 ms |
|---|---:|---:|
| isolated catchup，11次 | 103.958 | 66.378 |
| isolated proposal，8次 | 103.998 | 67.224 |
| feedback catchup，11次 | 106.090 | 66.435 |
| feedback proposal，8次 | 104.994 | 66.288 |
| 38次加权均值 | 104.802 | 66.554 |

新候选报告均值下降约36.5%，但两轮不是配对稳态验收，也不含host查表／输入准备，不能据此给出端到端加速或把全部收益归因于算子本身。目标是常驻预热后proposal同步Process均值≤50ms，尚未达到。

新请求 `qwen35_draft_accuracy_279e74c27a2c470598f5d4488c3bf54f` 已完成、输出回收且锁释放。Top1七处差异及KV误差范围与上一候选相同：appended KV相对L2约0.44–0.83%，未写入区保持；feedback两组重复逐位稳定。isolated首例hidden相对L2 **65.11%**，feedback仍为1.45–2.61%，均未通过原容差。

首例快照进一步确认：isolated第0次与feedback第19次使用同一组文件输入，Top1及12路KV逐位相同；hidden的30720个元素中12928个不同。两者零值数为12931／3，差值恰等于全部不同元素数，因此所有差异都是isolated为零、feedback非零，其余17792个元素逐位相同。各行差异计数和首末位置呈32个FP16对齐，但尚未检查每段范围或证明cacheline根因。运行器会在分配后清零输出，现提供复用原夹具的显式 `untouched` 初始化实验；不跳过首轮，不改变模型或验收门槛。

后续边界报告已与这次精度测试绑定：OMC为1790943538字节，SHA256 `fadec11fab3e72c9fc2060a53fc57e0f6ccee136a87af1a064f7ece8f7b2f85f`。当前CPUCL=1、NPUCL=1：CPU只接收三个INT32控制量，解码操作为比较、逻辑、Gather、Select等，没有ArgMax；其6路输出逻辑大小合计410496字节。NPU有20路输入／14路输出。原末尾CPU全词表归约分区已消失，整图仍保留前置CPU控制计算。端口大小不是实测传输量，NPU内部指令精度及实际热点需其他证据。

StableArgMax整图profiling已回收：同一OMC、V1新进程、case 2（start16＋valid2，15个proposal），无预热一次调用。PID36349，model/parser退出码0，原始CSV两份，锁已释放。下表来自用户粘贴表和回收摘要，未在本地读取原CSV；不是常驻十次验收。

| profiling记录 | 本次 ms |
|---|---:|
| model load／inference／unload | 166.337／66.556／10.080 |
| 算子duration合计 | 61.960 |
| MatMul_30 | 19.965 |
| Conv2D | 8.753 |
| Transpose_1 | 5.284 |
| StableArgMax，npu_aicore | 0.415 |

Top1约占算子duration合计0.67%；前三项合计34.002ms、约54.88%。按源码结构，末尾MatMul应对应完整词表head，入口Conv及Transpose应对应特征投影与空间打包，具体编译节点映射仍需shape/边关系确认。后续优先分析这些区域，当前不继续优化Top1。

原生Top1旧profiling为model inference103.720ms／算子合计73.656ms，CPU ArgMax12.353ms；本次两个合计分别下降37.164ms／11.696ms。它们不是配对稳态测量，不能把全部差额归因于Top1。新表NPU start time集中在 `02:36:57.697844–.698040`，晚于model inference end `.695706`；时间域或事件含义未解释，不能据此计算CPU→NPU等待。算子合计不当作关键路径，66.556−61.960=4.596ms也不等于实测搬运或调度时间。

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
