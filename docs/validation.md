# 验证范围与保留结论

[首页](README.md) · [当前更新](current-status.md) · [项目结构](project-structure.md)

本页记录当前实现能证明什么。安装、编译、加载、执行、数值一致、实际执行位置和性能是不同检查；主机测试不作为设备通过记录。

## 融合探针首次真实编译回传

2026-10-27，用户回传DDK `100.520.020.010` 的5例结果：构图与AIR导出通过，OMC编译均失败，未上板。conv维推明确要求 `x2=[1,8192,4]`，与文档绑定的3不符；RMS当前注册表找不到 `AdnRmsNorm` 维推函数；GDN进入 `chunk_gated_delta_rule.py` 编译入口后子进程退出。GDN编译日志的末态是FP16，导出绑定为FP32，尚不能断定变化原因。`Ascend310B1` 内部日志及缺库提示也不能单独证明目标选错或崩溃根因。

本轮修复编译工作目录被删除的问题，新增只读 `inspect-fused-probes`，读取现有日志与匹配的安装接口文本；7项主机测试通过。没有修改文档绑定、生产缓存语义或共享DDK，也没有重新执行真实编译。下一步命令见[当前更新](current-status.md#用户现在执行什么)。

随后用户回传只读检查结果：两例GDN都在TBE `platform_info.py` 的 `te_update_version()` 抛出 `RuntimeError`，说明本次直接失败点是编译子进程的平台设置；尚不能判定参数、包版本或配置文件中的哪一项不匹配。路径位于Conda的 `site-packages` 不足以证明环境混用，`Ascend310B1` 与 `kirinx90` 的日志也不足以证明需要切换目标。安装目录确有GDN的Python编译入口及C++源码；仅凭通用 `DTYPE_MAP` 不能确认状态dtype支持范围。

针对首次检查遗漏调用帧的问题，检查器补充完整traceback、保存的OMG环境记录、报错指向的TBE平台函数源码、选定DDK配置目录及GDN状态读写片段。8项主机测试通过，覆盖调用帧保留、仅读源码、环境字段过滤及状态读写片段提取；独立代码审查通过。只读源文件，不执行TBE；这是诊断修正，尚未修复实际编译失败。

第二次检查回传进一步确认：安装版GDN的 `isGlobal`、`lsGlobal` 均为 `GlobalTensor<half>`，分别以 `reinterpret_cast<__gm__ half *>` 绑定 `initial_state` 和 `last_recurrent_state`。这为该源码的FP16状态ABI提供依据，与文档的FP32末态不同；不推广到其它DDK。新增 `bindings-from-document --gdn-state-source`，校验两处直接half指针后生成FP16末态绑定，并用组合来源文件绑定原文档和安装源码SHA256；默认文档绑定不变。

TBE源码显示：包装函数调用 `cce.te_update_version("kirinx90", "AiCore", "", "false")` 后丢弃非success返回值，抛出通用异常。新增 `check-tbe-platform` 在独立Python进程中读取该底层返回值，使用与OMG相同的环境构造，记录调用前后平台信息、模块哈希及加载库。仅导入初始化，不能宣称复现OMG工作进程；无论成功失败，都不能单独判断目标支持或编译通过。源码/接口修正及此轻量调用的实际结果仍待黄区执行。

本轮40项主机定向测试通过；命令行测试另验证源码覆盖、两例GDN的FP16输出描述和参考字节数，以及模拟底层返回成功/失败、字符串对象的退出状态。使用模拟TBE验证子进程流程，未调用真实DDK或设备。

用户随后回传：安装源码绑定生成成功，GDN C++ SHA256为 `01634d23da746856c1119a9743d7f6db5f66aef9560d745423556a9c3162125a`。轻量平台调用返回字符串 `error`；调用前后SOC为 `Ascend310`、core为unknown，未复现原OMG工作进程的 `Ascend310B1` 初始化状态。进程同时加载本DDK TBE目录的 `libtvm.so` 和OMG目录的 `libplatform.so`。这些是定位线索，尚不能证明该DDK不支持目标或两库不兼容。

随后增加当前DDK内的库搜索顺序与已安装初始化接口对照。最多四个独立子进程，均使用 `kirinx90`；不更改正式OMG环境，不自动采用任何对照结果。各进程的平台初始化、实际加载库和原始返回值分别记录；报告收集完成不等于编译环境通过。

对照工具17项主机测试通过，覆盖子进程隔离、同目标初始化、库顺序、缺失/不匹配接口、初始化失败和全部失败时的状态。命令行与文档Shell语法验证通过；使用模拟TBE。

用户回传的四组结果均失败：两种库顺序下，仅导入时SOC为 `Ascend310`，调用 `te_set_version("kirinx90")` 后为 `Ascend310B1`，随后底层更新再次收到 `kirinx90` 均返回 `error`。这说明单改库顺序不足以解决该次失败；初始化与更新对平台名的处理差异是下一项假设。`Ascend310B1` 是该DDK的实测返回值，不能据此把OMG目标改为该名字；用户分析文档的 `Ascend310P3` 仍只描述原工程平台。

新增 `check-tbe-platform --diagnose`，把包来源（当前环境/当前DDK的TE wheel副本）、库顺序（OMG/TBE优先）、更新参数（请求名/初始化后读取的SOC名/空字符串）组成最多12组、4路并行。每组都先初始化 `kirinx90`，独立进程、独立输出和超时；包副本仅解压到运行目录，不安装或改写DDK。记录实际导入位置、映射库、更新前后SOC与原始返回值；成功仅代表该次API调用成功，需后续真实OMG验证。此轮不修改生产转换配置或算子计算。

批量诊断工具24项主机测试通过，覆盖并发上限、别名与空参数、崩溃后继续、wheel来源隔离及不产生错误候选。真实CLI用模拟DDK完成12组及逐例进度检查；命令互斥和文档Shell语法通过，独立代码审查通过。

随后用户回传12组真实API结果：传 `kirinx90` 的4组均返回 `error`，传初始化后查询到的 `Ascend310B1` 或空字符串的8组均返回 `success`，SOC均保持。当前环境与DDK wheel所列7个模块哈希一致；改变OMG/TBE库顺序没有改变结果。这把本次平台API失败定位到名称处理差异，不支持继续把已列模块版本或库顺序当作主要原因，也不证明OMG完整编译已修复。

新增显式 `probe-fused-ops --tbe-diagnosis`：核对该诊断与选定DDK的来源，在运行目录复制TE wheel，只适配 `platform_info.te_update_version` 的底层平台名称参数。只有请求为 `kirinx90` 且当前SOC与已验证的初始化结果一致时才转换，其余参数保持；记录真实工作进程调用。OMG平台参数仍为 `kirinx90`，DDK及已安装Python包保持不变。`--cases gdn_prefill,gdn_decode` 只重试两张真实尺寸GDN图，conv/RMS另行处理。

兼容适配的28项环境测试与40项探针/绑定测试通过；最终AST守卫的4项定向复测通过。实际CLI以模拟GE/OMG工具联通两例GDN，独立Python工作进程均加载运行目录副本并留下 `APPLIED` 记录；参数仍为 `--platform=kirinx90`。这验证主机接线与审计，不是实际DDK编译通过。文档命令及独立审查通过，真实OMG结果待回传。

## 完整结构与独立融合调用

2026-10-27，已在主机实际生成完整40层 target、6层 DFlash：PyTorch 输出全尺寸 meta 模型及可执行的 decoder 层级 FX；GE 输出完整节点、连边、C++、recipe 和缓存 ABI，分别生成了非融合、融合 prefill 与融合 decode 结构。均未加载真实权重，也未生成真实 AIR。

新增独立 GE 探针只覆盖 `ChunkGatedDeltaRule`、`LinearAttentionConvstateUpdate`、`AdnRmsNorm`，共5例。每张图只有一个自定义节点，使用完整模型的算子尺寸；不依赖模型权重。

主机联调已贯通接口绑定、生成、打包、批量脚本、native runner、回收及比较，5例均通过接口校验；GE/OMG 和 HIAI 使用测试替身。CTest 5/5、真实 DDK 头文件编译通过。GDN 检查有效前缀及 FP32 末态，conv 的 `y1` 顺序未定义，只记录形状与非有限值；数值误差仅作诊断，没有设验收阈值。

此后的真实GE导出和目标编译回传见本页首节；融合算子板端调用尚未执行。命令见[融合探针](ge-guide.md#融合算子调用探针)，主机证据见 [specs/verification.json](../specs/verification.json) 的 `latest_full_graph_and_fused_probe`。

## 状态执行与融合接口接入

2026-10-27，本轮新增了可选 PyTorch 融合适配、GE 状态图独立编译及输入打包、正式 runner 的 `--stateful-step` 模式。

| 范围 | 主机验证结果 |
|---|---|
| PyTorch 融合选择、模型及缓存事务 | 108 项和27项 subtests 通过；其中一项测试夹具修正后单独复测 |
| GE 编译与来源、ABI 校验 | 133项回归及4项补充检查通过 |
| 状态输入打包与连续调用校验 | 35项通过 |
| C++ runner | CTest 5/5通过 |
| Python → native prefill → 状态回传 → native decode | 通过；past_length 为 `0→3→4`，conv、FP32 recurrent、K/V 四组状态字节与 SHA256 精确接续 |

跨模块验证只替换外部 GE/OMG 运行产物和 HIAI 实现为明确标记的测试替身，保留真实的图构造、产物校验、打包和 native 执行流程。它证明主机接口与状态传递接通，**不证明真实融合 kernel、OMC 编译或板端精度通过**。常驻状态管理及 native DFlash 调度仍未接通。当前环境也未取得可用的目标资源清单，真实设备联调待完成。

命令见 [GE 指南](ge-guide.md#状态图编译和输入打包)和[板端单步执行](board-guide.md#6-新状态图的单步执行)，证据定位见 [specs/verification.json](../specs/verification.json) 的 `latest_integration`。

## 本次代码整理的验证

2026-10-27，模型解释器下的定向回归：

| 范围 | 结果 |
|---|---|
| PyTorch 新模型、算子接口、缓存及验证事务 | 89 tests + 27 subtests 通过 |
| GE 融合绑定、状态图、生成器与原导出回归 | 154 tests 通过 |
| HIAI/HDC 生产路径与依赖回归 | 150 tests 通过 |
| C++ 主机 runner 构建与 CTest | 构建成功，5/5 通过 |

剩余1843项 Python 测试收集通过，无导入错误；88个本地文档链接、29段Shell和3段Python示例通过静态检查。PyTorch/GE tiny参考均为PASS，文档接口生成与非融合状态图源码生成也已实际通过。

移动代码的定义与入口保留情况另做 AST 对照；整理验证没有执行真实 GE 库、OMG、融合 kernel 或板端程序。新代码的 host/reference/fake 测试覆盖接口和行为，不证明黄区 kernel 已安装或新 ABI 已在真实设备接通。

需修改模型或接口时，可在[环境配置](environment-setup.md)后执行对应回归；仅使用项目不必重跑：

```bash
export TEST_RUN="$(mktemp -d "$AI_RUN_DIR/tests-XXXXXXXX")"
"$MODEL_PYTHON" -m pytest "$REPO_DIR/tests/test_fused_model.py" \
  "$REPO_DIR/tests/test_fused_session.py" "$REPO_DIR/tests/test_ge_fused_target.py" \
  -p no:cacheprovider --basetemp "$TEST_RUN/tmp" --junitxml "$TEST_RUN/results.xml"
```

## 用户回传的设备结果

旧 tiny 双图已回传加载与生成成功。普通/DFlash 在其测试中一致，但尚未与主机参考完全对齐。41个旧 GDN repair 用例全部编译/加载/运行，数值32通过、9失败；这些是用户粘贴报告，未在本次获取原始设备产物重新校验。

已收敛的结论：

- 所测 rank-3 padding/window 及 beta/decay 布局修复后，对照恢复；weighted value 和 scan 对照通过。
- 小幅度 Q/K 的 MatMul 归约丢值，原生 Sum 对照保留16/16个非零结果；正确 sum 输入下的加 epsilon/Rsqrt 对照通过。
- 原生归一化首层最终输出通过，max_abs=`0.000793457`、relative_L2=`0.005915196552830379`；分阶段 `normalized_key` 仍失败，max_abs=`0.00390625`。
- 部分 MatMul/BMM 边界用例仍丢小值。最终 BMM 对照虽然过原阈值，仍只有92/128个非零结果，relative_L2约3.81%；不能只看 PASSED，也不能据此把所有算子概括为同一种下溢机制。

这些结果结束了相应探针的反复定位，未改变原有数值阈值。它们不覆盖最终融合 GDN、完整模型、缓存连续生成或 DFlash 加速，执行位置也未独立确认。

## 后续验收

围绕实际采用的融合路径，集中验证算子接口、同权重目标输出、prefill/decode 状态连续性、部分接受回滚，以及普通/DFlash token 与停止原因一致性。最终性能比较必须匹配权重、量化、缓存策略、提示、生成长度和测量条件。完整模型质量与至少2倍加速目标均未验收。

## 历史归档

本次删去的探针源码、用例、旧文档和逐次验证 JSON，已与整理前的未提交更改一起归档到仓库外。归档有逐文件 SHA256 清单，包含原 HEAD `4a043641fb5872b5af24c39b86b4b86e70b77014`；已提交版本也可从 Git 历史找回。

归档文件为 `before-reorganization/source.tar.gz`，SHA256：

```text
4269cccd3b3a57f04a15e8b520fb81f248a77a441593fa8a3b2d4ba0651071c3
```

本地运行记录和归档定位见 [specs/verification.json](../specs/verification.json)。源码仓库只保留当前结论与必要生产回归。
