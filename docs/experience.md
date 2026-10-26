# 经验审计（2026-10-20）

本工程由 `deliver-model-end-to-end` 管理；下表是首次实现的历史记录。当前因量化精度要求更新，完整模型阶段停留在 contract。以下结论均限定证据范围；没有把未运行的 X90 行为写入通用技能。

| 观察/最小复现 | 证据与范围 | 决定和下一验证 |
|---|---|---|
| 4B 中已有 DFlash draft math 和 native greedy scheduler | `specs/reference-provenance.json` 保留原始文件 SHA256；旧工程记录有 CPU/mock 范围限制 | 复用源代码并审计 35B 配置；不继承 310P 或 4B 的板端/性能结论 |
| 用户把 NNRT 更正为 HIAI，明确 AIR→OMG→OMC | 用户本会话；蓝区头文件与 OMG help | 正式源码移除 NNRT；保留 HIAI 适配和 AIR pipeline，旧原型仅在本 run 历史日志中 |
| 去掉签名查询串的华为下载链接返回 403 | 早期失败记录、官方 preparations.md 中完整签名链接、官方 SHA256 | 下载脚本保留签名链接并核验哈希；作用域为本次两个公开 6.1.1.0 包 |
| 解压的 OMG binary 未带执行权限；vendor wrapper 包含在 DDK 树创建 glibc symlink 的逻辑 | 实际文件权限、wrapper 源码、真实 `--help` | 只修改 run 内解压副本；转换直接运行 binary 并显式设置隔离环境，不修改 shared profile |
| eager MoE 只遍历样例命中专家，直接 trace 会冻结 expert coverage | 上游专家 forward；跨 seed/input 路由测试 | ONNX 数学验证使用动态路由索引；AIR 基线使用固定形状全部专家 + 动态 mask。后者成本高，尚无性能晋升 |
| Transformers 输出采集进入 lock/ContextVar，Dynamo fullgraph 断图 | `log/dynamo-preflight.log`（实际失败）；`log/dynamo-explicit-loop.log`（修复后完整捕获） | 目标导出使用显式层循环，保留相同 mask/position/feature 顺序；与未包装 upstream 比较，完整权重仍待验证 |
| 文本 MRoPE 的原地 strided slice 在早期 tiny ONNX 导出失败 | 旧失败日志与 `tests/test_onnx.py` | 三轴文本位置相同，改为等价频率计算；不扩展到 multimodal 假设 |
| HIAI 公共兼容 API 有 ND dims、同步 Process、NPU 检查和路径加载，size buffer 为 uint32 | 蓝区真实头文件；OHOS ARM64 对象文件编译；fault double 测试 | 用路径加载大模型，检查 tensor byte overflow；不宣称 >4GB OMC 和 BF16 已被运行库支持 |
| HIAI 公开 API 不暴露模型 IO dtype | 真实 `GetModelIOTensorDim` 签名 | ABI dtype 由 AIR/OMC hash manifest 锁定，加载端检查 ND/shape/bytes；内部接口若不同先反馈编译或加载日志 |
| 双方目前没有 TorchAir；本机无完整权重和板机权限 | dependency preflight、用户回复 | 加入导出前依赖检查与 `export-tiny-air`；未用 fake payload 或 ONNX 冒充真实 AIR/OMC |

新增可执行检查：MoE 路由/显式导出循环跨输入与 BF16 测试；真实 Dynamo fullgraph 捕获；AIR 外部权重篡改与 OMG 失败清理；HIAI 初始化/兼容性/部分加载失败/张量分配/运行失败及单次推理测试。结果见 validation。

未修改通用 skill：这些观察与新芯片/模型的目标集成尚缺完整证据，当前沉淀为模型局部实现和回归检查，避免推广成未经验证的平台规律。首次实现没有委派 custom operator，也没有执行量化；恢复后的量化工作见下节。

拒绝的晋升：toy/ONNX/Dynamo PASS 不能替代 AIR；fake HIAI PASS 不能替代设备；对象文件编译不能替代内部库链接；缺少画像不能自动解释为 hardware-reference opt-out；未量测不能宣称 DFlash 加速。

剩余假设：内部 HIAI 与蓝区 API 是否一致、X90 BF16/GE 算子支持、FileConstant 大权重转换及加载、实际峰值内存、真机 token 一致性和收益。下一步由用户内网先补 TorchAir 并跑 tiny AIR→OMC，再提供日志用于有针对性的修改。

## 恢复后经验审计

当前 run：`20261020T092412Z-4528-622cba`。用户解除暂停，并明确允许权重与激活量化、其余 FP16。

| 观察/最小复现 | 证据与范围 | 处理及下一验证 |
|---|---|---|
| `.half()` 后上游仍显式执行 FP32 GDN、norm、router、RoPE | `tests/test_precision.py`，真实 FX tensor dtype metadata | 普通图执行 FP16 lowering 并重新捕获审计；这是数值改变，保留 upstream reference；全权重质量待验证 |
| 改特征 dtype 必须同步 HIAI 字节数和 artifact 身份 | ABI v2、真实头文件编译、fake HIAI 首末 FP16 元素复制检查 | 拒绝旧 FP32 bundle；整数 token/mask 仍为4字节 |
| MoE 专家是 Parameter bank，普通 Linear visitor 会漏量化 | 全模型 meta 统计、专家改写前后跨输入比较 | 暴露全部10,240专家/20,480矩阵；不得仅统计被样例命中的专家 |
| DOPT 公开插件可实际校准小 MoE，但 Cython QLinear 不能直接 Dynamo capture | `log/dopt-model-probe.log`、`log/dopt-fx-probe.log` | 主机参考以不改变 vendor math 的 opaque bridge 调用；AIR 只允许原生 packed quant op，不能回退导出模拟层 |
| A16 输入量化的 FP16 缩放可溢出，产生 NaN | `log/dopt-frozen-compare-v2.log`、首次完整测试中 quant reference 失败 | 输入量化器内部改为 FP32、输出回 FP16；用独立高精度饱和量化 oracle 和真实 DOPT 回归；板端仍待验证 |
| DOPT 暴露 get_air_model，但缺 NpuQLinear/torch_npu 扩展 | `log/dopt-linear-probe.log`、`out/preflight.json` | 导出前检查特定扩展，避免先分配35B再报错；没有宣称当前环境具备 AIR 量化能力 |
| 只比较 checkpoint SHA 不能区分 BF16、FP16 和不同校准 | `tests/test_cli.py` 的 precision、quantization、ABI 负例 | token/stop比较同时绑定源、校准、precision、ABI、OMC哈希；普通/DFlash一致不能证明量化质量 |
| OMG 短相对路径修复后实际负例进入 AIR parser | 原 run `log/omg-air-short-path-smoke.log` | 路径问题已定位；当前编排回归覆盖成功/失败清理；仍无真实有效AIR转换PASS |

本轮加强的是模型局部可执行检查。未修改通用技能：DOPT二进制、CPU模拟溢出和X90导出能力没有足够跨版本/板端证据，不推广成普遍规则。没有委派独立 NPU operator。

拒绝晋升：真实 DOPT tiny校准也不能证明35B质量；主机opaque桥接不能作为AIR；存在 get_air_model 不等于已支持X90整个模型；32GiB预算不能替代峰值测量。余下假设是内部量化op布局与API、全模型转换覆盖、数据校准质量、峰值内存、量化参考与设备数值一致性及实际性能。

## 已有 AscendC 环境复用（2026-10-21）

用户明确指出服务器已经 source AscendC 环境脚本，配置说明应省略重复的工具路径设置。指南现在复用 `DDK_PATH` 并仅补项目变量。源码审计同时发现 OMG 子进程覆盖了继承的库/Python 路径；已改为所选 DDK 优先、继承扩展随后，子进程回归覆盖真实扩展导入和路径优先级。证据见 `specs/environment-verification.json`。

范围限定为本项目的环境传递。没有把用户脚本成功解释成 AIR、量化扩展或板端运行成功；没有修改共享 DDK 或通用技能。后续在用户现有环境执行原预检/导出步骤，缺失项以实际结果为准。


## 直接 GE AIR 入口（2026-10-22）

本 run 为 `20261022T064712Z-4550-c5859a`。官方 CANN 9.1 文档明确 `Graph::SaveToFile` 保存 AIR；把 TorchAir 当作唯一入口会将 Python 后端依赖问题误当成 AIR 格式的必然要求。本项目新增主机 C++ 的最小构图/保存入口，先验证内部 OMG 的兼容性，保留完整模型原有导出入口。

独立审查覆盖图连接、FP16 常量、保存后读取、无设备初始化、资源优先级、文件身份与失败处理。官方头文件 compile-only 通过，主机编排测试通过；scope 与证据见 `specs/direct-air-verification.json`。已有端侧 DDK 的 Graph API 与服务器 GE 不同，不能把 `libgraph_host.so` 当作 `libgraph.so`。同版本 GE 的 Operator/Tensor 库拆分与 C++ ABI 也需按实际主机 SDK 匹配。

新增可执行检查为独立 smoke 的工具边界测试；显式 `--resources` 优先于环境里的 CANN 路径，避免选错 SDK。没有新增 NPU 自定义内核，没有修改共享 DDK、通用技能或模型计算。该路线的内部 OMG 和设备兼容性尚未被实际验证，经验仅记录在本模型；拒绝将仅有对象文件或 fake 转换的结果晋升为 AIR、OMC、完整模型或板端通过。

待验证假设：用户服务器真实 GE 动态库链接与序列化、内部 OMG 的 AIR/算子兼容、后续完整模型 GE 构图和量化映射、单图板端执行与双图数值门槛。本样例不证明 35B 内存可用或 DFlash 有性能收益。


## GE 查询接口与实际图状态（2026-10-22）

用户在 CANN 9.1 上实际运行旧样例，保存前检查报 `x failed`。官方 [GE 9.1 实现](https://github.com/hicann/ge/blob/0ef703178896cbeb2a66e7c751ac9626df7f7439/graph_metadef/graph/normal_graph/graph.cc) 表明 `FindOpByName`/`GetAllOpName` 针对 `AddOp` 注册表，不能用来判断 `SetInputs` 创建或 `LoadFromFile` 恢复的计算图是否含某节点。只补保存前 `AddOp` 会遗漏回读后的同一问题。

本次修复让保存前、回读后都检查实际 GNode；原来的数值与描述条件全部保留。经验审计包含保存/加载两条路径，以及 Tensor 属性访问是否再次依赖注册表；独立审查均通过，真实头文件 compile-only 通过。源码与证据位于 run `20261022T072928Z-5968-3afd6d`，摘要见 `specs/direct-air-lookup-verification.json`。

包装器曾只显示 CalledProcessError 命令，导致实际错误被留在日志中而用户反复排查依赖。现已直接展示阶段、日志路径和有界尾部，保留完整日志及异常链；17 项本地编排测试通过，包括具体用户报错与非空部分 AIR 仍须失败。新增输出冲突提示引导使用新目录，避免删除旧证据。

未修改通用技能：这是当前模型样例及特定 GE 接口的经验。没有自定义算子、近似计算或库替换。拒绝将程序启动、头文件编译或 mock 测试解释为 AIR/设备通过；没有把通用 GE 错误码猜成 NPU 执行失败或运行库 ABI 故障。剩余验证为修复版在用户真实 GE 环境的保存与回读、内部 OMG 和后续设备链路。


## 直接 GE 模型图与可扩展保存器（2026-10-22）

run：`20261022T074906Z-6507-ca8b2b`。Add AIR/OMC 通过来自用户报告，范围只到该单图。选择新增独立 GE 模型图、保留 PyTorch 路径，不把小图成功推广为完整模型转换通过。

数值审查发现 Python 浮点标量提前写入 FP16 Const 会增加一次舍入。使用 Power 的 scale/shift 属性保留原标量语义，浮点张量继续 FP16；目标 tiny 的严格比较恢复通过。GDN 保留原 chunk64 计算顺序，没有用递推式替换浮点累加顺序。NumPy 与 PyTorch 的内核差异由独立 PyTorch 原语执行器分离，未通过放宽阈值关闭问题。draft 的隐藏状态/logits 对原语参考使用默认 FP16 严格比较，Top1 对保留的 lowered SDPA 路径一致。

完整静态专家图会产生 23 万个 IR 节点；将每个常量和节点展开为 C++ 会使 tiny 编译已消耗数 GB RAM。改为有界 C++ 读取器加二进制构图描述与常量文件，完整 target 的读取器通过真实头文件编译。该文件格式只用于主机构图，不能改名或当作 AIR。检查覆盖损坏长度、类型、引用、端口、属性、常量，以及保存/回读的实际 GE 节点连接。

独立审查关闭 BOOL 算子类型限制、TopK 输出对应关系、实际 prepared 随机权重与参考不一致风险、权重内容身份及 recipe/blob 来源哈希缺口。tiny 使用 NumPy RNG，与旧 Torch tiny 不同，参考必须加载实际文件。完整权重通过 safetensors 的边界受检分块读取，FileConstant 按官方 schema 构建；没有把 FP16 `.bin` 标为整数压缩权重。

新增可执行检查位于 `test_ge_ir.py`、`test_ge_recipe.py`、`test_direct_ge_target.py`、`test_ge_draft.py`、`test_ge_export.py`、`test_ge_reference.py` 和 `test_ge_pair.py`。经验留在本模型，未修改通用技能，也未委派自定义 NPU 算子。拒绝将编译、配方解析、mock OMG 或 CPU 参考晋升为模型 AIR/设备成功。剩余假设是内部 OMG 对全部公开 GE 算子及 FileConstant 的支持、实际 kernel 数值、全量量化布局、内存峰值、完整权重质量及速度。


## GE 输入工厂与首错诊断（2026-10-22）

run：`20261022T084136Z-11227-7c2c63`。用户的 tiny 两个 Data 输入均报空原型，随后 SetInputs 构图失败；固定 GE 9.1 的 `BuildGraph` 查询顺序和工厂空返回路径解释了整条错误链。实例上的 InputRegister/OutputRegister 与算子工厂注册是不同机制。修复复用成功 Add 样例的现有 Data IR 工厂注册，没有新增 NPU 算子内核、改动计算或替换共享库。

只保留日志末尾会丢失最早的具体错误，GE 后续的错误码初始化日志尤其容易触发。现在同时收集有界首错和尾部，真实日志结构作为回归；完整日志始终保留。独立审查同时覆盖两种生成模式及无 GE 的配方解析模式，新增真实头文件编译和可启用的 GE Save/Load 检查。

经验保留在本模型及可执行检查中，未修改通用技能。拒绝将头文件编译、配方解析或 mock 转换晋升为真实 AIR 成功；当前主机缺少匹配 GE 库，运行测试明确跳过。下一步为用户服务器重试双图保存/回读与内部 OMG，后续算子或布局问题按新日志继续定位。


## 工厂原型不能用已赋形状的查询条件检查（2026-10-22）

run：`20261022T092815Z-12515-a03008`。Data 工厂注册修复正确，但新增预检查误将 `TryGetInputDesc` 当作端口存在性查询，给有效原型引入了新的阻断。固定 GE 9.1 源码证实该接口委托 `InputIsSet`，后者要求形状维度数大于零；失败并不意味着 Data 工厂不存在。修复移除该预检查，实际模型的描述和连接验证仍完整保留。

此前的头文件编译只能验证 API 签名，无法验证 API 对工厂原型的运行语义。此次审查沿调用链核对返回条件，并复核剩余前置条件；已有真实 GE Save/Load 回归保留，缺库时如实跳过。经验留在本模型，不推广为通用技能规则，不将主机源码审查或编译解释为实际 AIR 成功。下一验证为用户服务器重跑 export。


## AIR 成功后暴露的内部 OMG 广播约束（2026-10-22）

run：`20261022T093951Z-12973-586101`。用户报告 tiny target/draft AIR 已生成；失败阶段推进到内部 OMG。最初的退出码没有算子信息，补充日志后定位到 `v7/Select` 的维度验证与 `v23/Add` 的 CPUCL 类型检查。前面的 W/I 级插件加载失败不应凭字面 `failed` 当成根因。

`v7` 在源码为四维条件选择两个 FP16 标量。公开 SelectV2 的构图成功并不能证明内部 OMG 转成 Select 后支持相同广播；因此在生成 AIR 时显式广播条件和两个分支，所有输入与输出同形。范围覆盖 target 掩码、GDN、MoE 及 draft 注意力，保留 BOOL 条件和原有 INT32 布尔分支载体。使用真正 Select，避免乘法掩码污染 `-inf`/NaN；独立解码序列化 recipe 检查实际描述与载荷，再对边界数值逐位比较。接口类型仍依据[固定版本的官方 GE 定义](https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/python/torchair/_ge_concrete_graph/auto_generated_ge_raw_ops.py)。

`v23` 是 RMSNorm 的 `1 + weight`。生成图和 AIR 回读检查均声明 FP16，不能由该错误猜测某个 OMG pass 改成了 FP32。局部兼容改写将该运算表示为 Power 的 shift 属性，避开报告中失败的二元常量 Add；全部 63,488 个有限 FP16 位模式对原 Add 的主机比较一致。这不证明内部 Power kernel 的数值或转换已经通过。

OMG 包装器现在保留完整日志，并直接输出 graph、阶段、路径、首错与尾部；failure.json 记录退出码/超时及已完成图，失败不会生成成功的双图 manifest。摘要支持内部 `E/AI_FMK` 与 CPUCL 格式，并排除 W/I 级插件提示。GE/OMG 共用仅标准库的日志模块，避免将 NumPy/Torch 依赖带回 compile 路径。故障回归覆盖 target/draft 非零退出、超时、启动错误、空输出与缺输出及临时目录清理。

修复改变 AIR 内容，用户必须重做 export；旧 prepared 中的权重与配置仍可复用。未改 DDK、OMG 参数、模型 ABI 或通用技能，没有新增自定义 NPU 算子。未将用户 AIR 成功报告晋升为 OMC 或板端通过；full 35B/量化/性能状态仍待验证。证据见 `specs/direct-ge-omg-select-verification.json`，下一步在同一内网环境执行更新后的重试命令并检查实际 OMG 结果。


## Softplus 方言与 CPUCL 常量路径（2026-10-22）

run：`20261022T095753Z-14065-91325c`。用户的新进程 157178 日志明确显示内部 OMG 缺少 `SoftplusV2` 的推导函数；前一条 `v51/Add` 推导成功，不应再归咎于先前的 Select 或猜测环境缺库。官方 GE 文档提供原生 Softplus，蓝区 parser 包含其兼容映射相关字符串；后者仅支持选择候选，不能证明内部 X90 已支持整个图。

对于当前固定 FP16、beta1、threshold20，用单个原生 Softplus 避免手动 Exp/Add/Log 分解引入额外半精度舍入。全部有限 FP16 输入的理想函数阈值/非阈值版本舍入逐位一致；其他参数直接拒绝。该检查只证明阈值替换，不证明内核精度：审计确实发现当前 CPU Torch Softplus 与高精度数学 oracle 在一个接近舍入中点的输入上差一个 FP16 ULP，所以没有宣称不同内核全域 bitwise 一致。

同一日志的 CPUCL 报 32B 输入却要 64B，且 fp16Flag0；这符合半精度小常量进入单精度执行路径，但具体节点归属仍是推断。修正提前计算归一化偏移和 A_log 的小型静态一元算术，每个原节点仍转回 FP16；原始权重文件与来源哈希不动。转换只读单张量不超过 64KiB 的 FP16 文件，并限制累计缓存/读取到 8MiB。

不能无界地预计算所有常量子图：完整结构含大量 expert 切片和 reshape，按输出复制会膨胀到约 129GB。当前只允许 scalar_add/scalar_mul/exp/neg；完整结构候选 161 节点、约 346KB 为结构统计，未加载实际全模型权重。测试验证同一源文件不被改写、大张量路径根本不读取、预算和逐节点舍入，以及 tiny target 多种有效长度的输出不变。

日志摘要改为识别具体已观察到的 E 级 shape 回显和明确返回 SUCCESS 的空配置消息，而非忽略所有 E/[lsy]；原日志及尾部保留。未知 E 级错误、真正 shape 失败及 CPUCL 缓冲区错误继续报告。经验固化为模型局部实现和回归，未改通用 skill、DDK、OMG 参数或 PyTorch 路径；没有把主机回归晋升为 AIR/OMC/设备通过。下一验证是用户服务器复用 prepared 重导 AIR、重新 OMG，随后再做板端数值检查。

## 缺失前端推导不等于算子拼写错误（2026-10-22）

run：`20261022T101117Z-15205-4ffded`。内部 X90 的 Cumsum 缺少推导，但公开 GE 与 TorchAir 使用的规范名称就是 Cumsum；未将其猜改为 CumSum/CumsumD。完整结构的39种发射类型同时核查公开 schema、蓝区推导注册及实际转换表；GatherElements 等有正面证据，ArgMaxV2 仍保留内部待验证标记，不因旧蓝区缺少入口便改动 argmax 的 tie/NaN 行为。

FP16 扫描不能简单逐次 Add：每步半精度舍入不同于宽累加后输出半精度。前缀归约也不能无条件推广，正负抵消输入存在超过默认容差的反例。本次只对模型明确的非正 GDN 衰减域使用前缀 Slice/ReduceSum，长度限制256；整数位置扫描另用整数移位相加。主机门槛未放宽，并把序列化后扫描真正放回 tiny 双图输出比较。目标内核的归约累加精度仍须上板验证。

静态广播不能按每个专家复制：full S128 的重复扩展估算超过5GB，按源内容及目标形状复用后 target 仅约1.64MB。生成器保留1MiB/8MiB载荷上限，不读取大型权重；按字节复制而非乘法广播，保留非有限值及符号位。惰性常量还避免替换后原标量不再连接到图却仍被列入 Verify；整数长度1的别名也必须先物化源常量。相关反例已固化为测试。

193项本地回归和六张冻结描述图结构检查通过，4项真实 GE 可选检查跳过。仅用用户日志确定缺失推导根因，CPUCL 2B/4B失败节点仍未唯一定位。已观察到的 Reshape 链回显从首错摘要中过滤，未知 E 级消息和完整尾部保留。经验限于本模型，不改共享 DDK、通用技能或 PyTorch 路径；下一步验证内部 OMG，再比较板端输出，不能以本次静态审计宣称全算子或全模型已上板通过。

## 用转换前后字节证据收窄形状与环境错误（2026-10-22）

run：`20261022T103232Z-17196-b89598`。此前仅有 `1 0` 的回显不足以改变形状类型；本次 v974/v975 新证据为 `55 0 → -1 0`，与错误的最终大维度对应。两个32位字按 INT64 解释为4294967295，而不是-1。官方 Reshape 本来允许 INT32，所以只窄化该算子的 shape Const，并在 NumPy 转换前检查范围；覆盖普通 view、MatMul辅助展平/恢复、argmax keepdims。曾考虑的 Squeeze 方案未采用，因为新增日志证明相邻插维 Reshape 同样被改写，单点替换不能覆盖它。

Slice 类型错误被新上下文确认发生在 ConstantFolding 阶段，但具体节点仍未命名。只处理已确定的小型静态 FP16切片/reshape，按字节复制、不扩大浮点精度；每个源/结果64KiB、累计8MiB。tiny32个静态专家切片可以消除，完整专家大权重不能按这个方法无界展开。保留大源读取前的预算检查，累计收费包含缓存原始源及每个折叠输出。

TE的缺失Python函数说明所加载Python包与本地库接口不一致；本机蓝区对应包有该函数，但内部包未能检查。OMG子进程需自行优先选定DDK的包，不能仅优先lib64并沿用CANN的Python路径；路径剔除仅按显式根目录、解析符号链接、保留选定DDK及未知内部扩展，父shell不变。候选路径日志仍不能当成实际嵌入式Python加载结果。图转储有界保留并确保诊断失败不掩盖OMG原始错误；绝不通过给厂商模块补假函数隐藏不兼容。

最终测试覆盖255通过、4跳过；新增功能合并时两项测试假设需同步，生产代码未变，随后73项相关复验通过。源码、字节复现与实际内部运行证据分开记录；未把静态全结构核查或蓝区包检查升级为内部OMG/上板通过。经验保留本模型文档和可执行回归，未修改通用技能或共享DDK。下一步是新目录重导AIR、内部OMG，再核对真实HIAI输出。


## 编译 IPC 路径及首错优先（2026-10-22）

run：`20261022T104957Z-18903-e12366`。用户 PID157707 首先报 AF_UNIX 路径超长，随后 TE 的 multi_process_check 失败，最后才报 BatchMatMul 不支持。不能用末尾的算子支持结论掩盖前面的编译后端初始化失败。源码核对 v1115 的 FP16 五维矩阵形状、转置属性和 GE 端口均匹配；本轮保持模型和 AIR 不变，先修复编译临时路径。

完整 run 路径作为 TMPDIR 会让 CPython 的 pymp/listener 子路径超过 socket 限制。本机真实 Manager 重现该错误；父 PID 的 procfs 目录 FD 别名让后代无需继承 FD 即可访问短路径，实际载荷仍位于本次 run。独立审查还实测出仅使用 FD 编号的别名在描述符复用时可能让旧 worker 写入新目录；最终别名保留每次独有的随机目录后缀，重复同一故障注入后跨调用写入失败。已有超时处理仅结束直接 OMG 进程，完整进程树清理仍属既有生命周期限制，没有宣称此次修复结束了所有孤儿 worker。

最终45项定向检查通过，包括真实 fork/spawn/forkserver Manager 和 Listener/Client 往返、错误清理、并行目录隔离、旧别名失效及编译包装器集成。实际内部 DDK 仍需重试：procfs 访问必须可用，厂商若自行 realpath 还原长路径仍会失败；主机 IPC 成功不证明真实 TE 或 BatchMatMul kernel 已支持。经验留在本模型源码、测试和文档，未改共享 DDK、通用技能或设备门槛。


## TE 恢复后的批量矩阵乘兼容处理（2026-10-22）

run：`20261022T110926Z-19546-f28296`。用户 PID157847 补充日志明确 TE 与并行编译初始化成功，不能再用此前初始化故障解释最终 v1115 拒绝。v10 的三维常量子图可以在 CPUCL 编译，但不能由此推断动态 FP16 BMM 在 NPU 上可用。缺失预编译内核库后继续 JIT 的日志也不足以认定必须补库；Kirin 蓝区有 Ascend 内部标识映射，不能凭 Ascend310B1 字样切换目标。

蓝区实际 builder/vtable 检查证实 adj_x1 为真被拒绝；后续 Python 检查缺失，也没有直接五维限制证据。采用相同批次前缀压成三维、必要时显式左转置、结果恢复原形状的布局适配，并保留内部重试状态。禁止把仅 batch 乘积相同当作批次配对等价；所有新增 shape 继续用受检 INT32。逐批展开虽然数学可行，但完整模型卷积分支共有超过24万矩阵，会造成大量图节点，未采用。

六张实际发射描述图均保持矩阵乘数量，全部 BMM 为三维且 adj_x1=false，S128完整 target 的30处左转置被显式表达；未读取完整权重。新增30项主机回归覆盖序列化后执行、转置组合、NaN批次隔离和81个 tiny整图子图替换；193项既有回归通过，4项真实GE检查因资源缺失跳过。初次新增测试12项失败仅因用 bool 对象身份比较解码的0/1，改为值相等后30项全部通过，生产代码未因该测试修正改变。

经验保留本模型及可执行回归，不修改共享 DDK 或通用技能。当前模型的等前缀批次已覆盖，不能将此称作通用广播矩阵乘适配；没有宣称内部X90 kernel或板端精度/性能通过。下一步复用 prepared 重新导出 AIR，再在原内部 DDK 上复验。


## INT32 anchor Gather 与 dtype 验证（2026-10-22）

run：`20261022T112725Z-20655-b96dab`，基线 `bbcc505`。用户 PID158166 的流程已到 draft，明确拒绝 `v4/GatherElements` 的 INT32 数据输入。按编排执行顺序，可推断 target 已通过非空 OMC 和原输入哈希检查；没有收到该产物哈希、确切源版本或设备结果，不能晋升为独立验证的双图成功。

先前算子核查只确认 schema、注册和端口，遗漏了旧 IR 验证器的 dtype 限制，本次已更正兼容性表。蓝区实际验证器只允许 GatherElements 使用 FP16/FP32 数据。draft 的唯一整数实例改为 `[1,S]` INT32 token 向量化后执行 axis0 GatherV2，索引和输出仍为 INT32 `[1,1]`。非空右填充前缀保证索引范围；不引入浮点载体，不泛化为任意广播或负索引替换。导出器提前拒绝其余非 FP16 GatherElements。

蓝区 AIR 的 GatherV2→GatherV2D 转换、INT32 验证及 CPUCL 注册/DoGather<int> 路径均已核查。NPU 算子表未声明 INT32，因此该证据不能证明内部 X90 的设备分配。模型局部回归覆盖完整词表的大 ID、全部73种 S8/S65有效前缀长度、独立的 INT32 边界复制、两种序列化方式及原 tiny draft 全输出等价。最终235通过、4跳过，无未解决失败；跳过项需要实际 GE 资源。

六张 tiny/full 实际发射描述图均无 GatherElements，三个 target 图的全部发射记录哈希与基线一致。修改只影响 draft 的整数选择，旧 prepared 可复用，AIR 必须重导。内部 OMG、HIAI 设备和全模型量化/内存/性能仍待验证。证据见 `specs/direct-ge-integer-gather-verification.json`。经验保留本模型及可执行测试，不修改共享 DDK 或通用技能。


## INT32 Maximum 后端与有界位置截断（2026-10-22）

run：`20261022T114012Z-21463-7a8f45`，基线 `be88c95`。用户 PID158383 的 draft 编译已越过 Gather 验证，最终拒绝 `v12/Maximum`。该算子计算 INT32 上下文位置 `max(count-1,0)`。蓝区 IR 允许该数据类型，但 CPUCL 与 elementary NPU 的 Maximum 支持表不接受 INT32，说明仅检查前端注册和推导仍不足以保证后端支持。

审计进一步发现蓝区 CPUCL Less<int> 以 INT32 差值再转浮点判断，极值输入可能溢出。因此撤销初步的通用整数 max/min 改写，只在模型内用 `Less(x,0)` 和同形 Select 替换有界位置截断：`x∈[-1,S-2]`，`S<=INT32_MAX`，差值 `0-x∈[2-S,1]` 不溢出且转浮点保持符号。原生 FP16 极值、通用整数极值发射器与 PyTorch 路径均未改。蓝区 Minimum 本身允许 INT32，不能把它和 Maximum 的限制混为一谈；INT32 Select 只确认 CPUCL 支持，不承诺纯 NPU 放置。

新增10项位置测试覆盖74种前缀、S上界证明、构图前超限拒绝及完整tiny草稿输出/ABI/权重不变；最终既有回归235通过、4跳过，总计245通过。通用极值实验与一次中断的旧源码测试明确作废并保留在run，未作为最终验证。tiny/full共六张发射描述图的审计显示target全部记录与基线一致，draft只增加两个节点。没有实际内部OMG或设备执行。证据见 `specs/direct-ge-position-clamp-verification.json`。

经验保留在本模型和范围限定的可执行回归，未修改共享DDK或通用技能。旧prepared可复用，必须重导AIR；内部双图转换、上板、完整35B压缩量化、质量、内存与性能仍待验证。


## Tiny 编译成功后的 HIAI 交接（2026-10-22）

用户反馈最新tiny双图转换成功。保留用户报告与独立产物/设备验证的边界，未要求重做已通过的转换。实际成功的compile输出可能是用户持续使用的 `$GE_PAIR_RUN/omc`，不能仅依据上一条示例变量假定为 `omc-clamp-*`；新指南要求绑定实际 `--output` 并检查manifest及两张OMC哈希。

匹配参考必须读取同一prepared，不能重新运行另一随机权重路线的demo。将此前跨页路径映射整理为直接GE专用的服务器/板端命令；INT32辅助算子可能走CPUCL，编译成功不代表全图NPU执行。正式构建需要OHOS ARM64库，服务器AscendC库与头文件编译结果不能替代。文档命令语法和接口核对通过，未改通用技能；经验仅记录当前模型。后续HIAI链接、probe、真实推理与比较结果仍待用户执行。


## 用小图区分算子能力、编译分区和板端执行（2026-10-23）

用户的板端日志已从空 model buffer 失败推进到 HCL RestoreFromBuffer：CPUCL 兼容检查拒绝 FP16 MatMul，并报告某个 activation mode 与 Cast dtype 不匹配。另一份 OMG 日志明确出现 `input dimCnt 5 > 4`、Slice `not support realDimCnt 5`，以及 BatchMatMulV2 不在某个 TBE store 中。单个 `custom_ascendc_lib` 的拒绝不是整个 NPU 的拒绝；CPUCL 的候选检查错误也不自动证明最终 CPU 放置。

为此新增 `probe_hdc_ops.py`：47 个固定输入小图沿现有 GE C++ → AIR → OMG → HIAI 路线，比较窄 M/N MatMul、常量和动态右输入、卷积形态 BMM，以及 Neg/Add/Mul/Select/Slice 的 5D/4D 对照。每个案例独立进程、显式 model buffer、哈希校验上传和回传，按实际阶段记录编译/加载/执行/数值结果。默认只删除本次创建且所有权标记匹配的目录；基础设施失败保留现场。SDK/库复用现有构建记录，不增加 lib 参数。

编译器的成功分区转储与失败转储均按预算保留，并排除在模型上板 payload 之外。报告优先展示 dtype、维数和算子注册问题，保留常量折叠图与主图的分区范围，设备执行位置没有独立证据时保持 UNVERIFIED。完整 HiLog 仅保留本地，分享包只含对应 PID 的日志；环形缓冲区可能遗漏早期案例，PID 也可能复用。

主机验证包含确定性数据、发射、模拟工具失败、HDC 退出标记/文件哈希/目录所有权、报告阶段判定，以及 C++ fake HIAI 生命周期检查。OHOS 编译仅证明公开头文件下的编译契约，不代表内部配套库链接或真机成功。蓝区 FP16 MatMul 的独立编译对照可形成 NPU 分区，但没有用于断言用户内部 X90 工具链支持。证据见 `specs/operator-support-probe-verification.json`。未改共享 DDK、生产图的算子放置或数值阈值；下一步收取用户 `results.tar.gz` 后按真实支持差异选择改写。

## 含 NPU 分区的统一加载失败与 NPUCL 注册（2026-10-23）

run：`20261023T090425Z-28419-2bbca1`，基线 `6330fc0`。用户回传 `op-probe-20261023-165317` 报告：47 项中 45 项编译成功，35 项含 NPU 分区的案例全部 Load 失败，10 项 CPU-only 案例全部加载成功，其中 7 项数值通过，3 项停在 IO 描述检查。这一分组先指向共同的 NPU 加载路径，不能把每个 Load 错误都归为单算子不支持，也没有据此降精度的理由。

随后用户提供 MatMul PID16394 与 Add PID18339 的 HiLog，两者明确报 `Model exist graph op with NPUCL, but this cl not existed.`，并在 CheckCLSupported 阶段失败。当前进程未注册所需 NPUCL 是直接失败原因。前序 `libai_fmk_dnnacl.so` 加载失败是相关事实；其是否为本版本唯一所需插件、是否有其他依赖/配置/固件匹配问题仍未证明。现有自动部署只验证构建记录及同目录库哈希、ELF 架构，不能据此宣称动态后端依赖完整。未盲目改库搜索顺序、替换系统库或更换平台参数。

新增报告检查仅在原生 Load 失败、退出码非零且对应 PID 日志明确给出 NPUCL 缺失时输出 `NPUCL_NOT_REGISTERED`；缺日志、其他 PID 或单条 dlopen 警告不作此判定。IO 诊断输出全部实际输入输出形状/格式，并给出具体声明与实际不匹配项，保留拒绝执行的门槛。三个 IO 失败尚不能确认是单例维度、格式或导出描述的哪一种问题。Select 的编译失败定位到 condition CastT 未注册 kernelId，不能直接推广为 Select 全面不支持；GatherElements 的最终兼容拒绝只适用于此次小图配置。

47 项 Python 报告/传输回归与 28 个原生 fake HIAI 用例通过，README 命令通过 bash 语法检查；这只验证诊断改动，不代表板端后端恢复。经验固化在本模型的报告及原生回归，不改通用技能或共享 DDK。用户粘贴内容与独立产物验证分别记录于 `specs/operator-support-probe-user-report.json`。下一步只读核对 DDK/板端候选运行库及注册路径，恢复后先复验一个 MatMul 和一个 Add；完整模型与真实 NPU 执行、精度、性能门槛保持未通过。


## 板端后端存在与自动部署的选择边界（2026-10-23）

run：`20261023T091800Z-29927-3b58ce`。用户清单显示板端 `/vendor/lib64/chipsetsdk` 有 HCL runtime、NPUCL executor client 及 NPU graph executor 等库；因此此前进程的 NPUCL 未注册不能扩大为板端没有 NPU 运行库。DDK 的 `ai_ddk_lib/lib64` 则带有 `libhiai_hcl_model_runtime.so` 和实现库，现有自动部署会一并上传并将该目录置于局部搜索路径前面。它们是否抢先选中了不含 NPUCL 的 DDK 后端仍是待证假设。两个后端入口的 basename 不同，单纯追加 vendor 目录并不能证明发生了切换。

`hiai_pipe_lib/lib64` 中多个公共库的用户报告 BuildID 与 `ai_ddk_lib/lib64` 相同，不以目录名不同作为替换依据。主机清单中的 x86-64 simulator/TBE 库不用于 ARM64 板端。下一步核对实际 runner、libhiai、ir_build 与 HCL 两库的 DT_NEEDED/搜索路径和选择入口，再决定隔离对照应排除的准确文件；未更改自动部署默认策略、系统库或生产精度。记录见 `specs/operator-support-probe-user-report.json`。

## 复用原探针产物的 HCL 部署对照（2026-10-23）

run：`20261023T092155Z-30133-a88918`，基线 `ffe0d18`。用户回传实际 ELF 动态依赖：探针及所查保留库未直接依赖 DDK HCL 两库，`libhiai.so` 字符串同时包含 DDK 与板端 HCL 名称。它支持做隔离实验，尚不能确定选择顺序或板端 ABI 兼容性。旧 readelf 的 DWARF 警告不作为 ELF 损坏证据。

新增 `probe_hdc_runtime.py` 从原 manifest/build/result 及 SHA256 记录复用既有 runner、三份 OMC、输入和参考，不因更新仓库重编译历史探针。基线保留原完整库集合；候选组仅排除两个指定 DDK HCL 库，并在局部搜索路径加入 `/vendor/lib64/chipsetsdk`。这是库集合与搜索路径的联合对照，成功也不能单独证明是哪项变化起效。检查全部保留 ELF 的 DT_NEEDED、SONAME 和重复内容，避免直接依赖缺失或别名重新带入；冻结小产物并在两组前后核对原文件和副本。每组独立目录、回传及受限清理，不触碰已有大权重目录。

ELF 读取器通过程序头解析动态表，不执行目标文件；三份本地真实 ARM64 ELF 的结果与 SDK llvm-readelf 一致。最终相关主机回归 109 项通过；首次最终复跑漏设 PYTHONPATH 导致收集失败，补上既有 model 路径后通过，记录保留于工作目录。严格 IO 和数值阈值未改变。板端库实际映射、NPUCL 注册及执行位置仍待用户对照结果，不把测试通过升级为模型精度或性能结论。证据见 `specs/hiai-runtime-comparison-verification.json`。经验保留在模型内，无需据此改通用技能或共享 SDK/DDK。

## 后端对照通过加载后的 IO 诊断（2026-10-26）

run：`20261025T235900Z-6378-a4df2b`。用户粘贴的 A/B 报告显示：基线 MatMul、四维 Add 仍在 Load 失败，五维 Add 数值通过；候选组三项均加载成功，两个 Add 均数值通过，MatMul 停在严格 IO 形状/格式校验。随后收到候选 MatMul PID13929 与 Add PID14066 的 HiLog：均打开 `libai_npucore_itf_client.so`，注册 NpuclEventListener，完成 NPU 图执行器初始化和 LoadModelsByHcl；Add 还有 Run、modelId=1 的执行 ioctl 完成记录，区别于后续停止模型的命令。这为候选 NPUCL 加载恢复及 Add 设备提交提供了用户日志证据，不能再视为 NPUCL 完全不可用。MatMul 尚未调用 Process。缺少已排除 DDK HCL 库的前序报错后面接续成功加载，不作为本次终止原因。

实际映射 DSO 身份、逐 kernel 硬件执行证据及库排除和 vendor 路径各自的贡献尚未独立证明，自动报告仍保留 placement=UNVERIFIED。不能把当前 IO 错误归为 MatMul 算子不支持，也不能直接跳过布局检查。原始报告归档尚未收到，证据等级保持用户文本回传，见 `specs/hiai-runtime-comparison-user-report.json`。

A/B 默认精确复用旧 runner，因此更新源码不会让旧二进制获得已加入的详细 IO 输出。对照原探针提交后确认 native 差异仅为描述和错误信息；新增显式 `--refresh-runner` 只重建诊断可执行文件，并让两组使用同一个新文件。原 OMC、输入、参考及库哈希继续复用，重新核对新 runner 动态依赖，保留旧二进制身份和当前构建证据。新输出能区分具体输入/输出、声明形状/格式与实际形状/格式，但旧 HIAI 查询接口仍不能直接提供 dtype。下一步依据这些描述选择有证据的布局适配，不猜测单例维度、不降低精度或数值门槛。默认部署策略和共享资源保持原样；主机验证结果见 `specs/hiai-runner-refresh-verification.json`，未推进完整模型门槛。经验固化于显式刷新模式及回归，无需修改通用技能。

## ND 尾部单例维度的精确对应（2026-10-26）

run：`20261026T001240Z-7902-db05a6`。刷新后的用户日志给出 MatMul 两个输入及输出的实际描述：逻辑 `[8,16]`、`[16,16]`、`[8,16]` 分别变为 `[8,16,1,1]`、`[16,16,1,1]`、`[8,16,1,1]`，三者格式仍为 ND(2)。这把此前未确定的形状/格式差异收敛到尾部补单例轴；不能推广成任意同元素数的形状或格式均可接受。

探针仅接受两类对应：原样 ND 匹配；或逻辑 rank 1～3、实际恰为 rank 4、原维度前缀完全相同且新增尾维均为 1。追加单例轴不改变线性元素序列和总字节数，属于精确的描述适配。每次接受补维均写入日志，保持原合约，仍使用实际 TensorDimension 初始化 AiTensor；不转置、不补矩阵数据、不换 dtype、不改比较阈值。前/中部补维、非单例尾维、前缀变化和其他格式仍在 Process 前拒绝。此改动仅在算子探针，完整模型 executor 保持原策略。

主机 fake-HIAI 检查验证描述传递、输入输出字节、拒绝路径及生命周期，不能证明真实矩阵乘法或 NPU 执行。用户仍需以同一 OMC/参考重新运行 `--refresh-runner`，取得 MatMul Process 和数值结果。记录见 `specs/hiai-probe-nd-padding-verification.json`；本次实测的具体 ND 归一化保留在模型证据和探针回归中，没有提升成所有平台通用规则。

## 三案例数值通过后的原算子集合复验（2026-10-26）

run：`20261026T002510Z-8234-33823c`。用户回传候选组 `matmul_m8_b_dynamic`、`add_rank4`、`add_rank5` 均通过 Load、Process 与原参考数值检查；基线仍为前两项 Load 失败、CPU 对照通过。MatMul 的三个张量都记录了 ND 尾部补 1 的接受信息，FP16 和输入输出数据不变。这为该形状的部署和描述适配提供了用户板端数值证据，不能推广为其他形状、算子或完整模型通过。自动 placement 字段仍未提升为已验证，当前回传也没有独立逐 kernel 执行证据。记录见 `specs/hiai-runtime-comparison-user-report.json` 的 `nd_padding_retest`。

下一步用显式 `--replay-all`，从原 manifest 按原顺序校验并复用所有已编译案例，只在候选库集合和搜索路径下运行，并自动构建一次当前探针。旧后端基线无需在全部案例上重复；两个原编译失败案例保留为未重试，而非静默删除或算作成功。原编译分区与新加载/执行/数值结果并列，供后续选择有证据的 CPU→NPU 图改写；不根据单个 NPU 子库的候选拒绝日志判断最终分区。模式不重编模型，不放宽原参考阈值，不改变正式部署默认，仍受原哈希、依赖检查和受限清理约束。主机验证见 `specs/hiai-operator-replay-verification.json`，全量候选板端结果和完整模型门槛保持待验证；未修改通用技能。


## 2026-10-26：结构候选、最终目标与执行优先级

run：`20261026T002510Z-8234-33823c`。用户提供同板可运行工程的结构摘要，但随后明确其精度有问题、原因未知；GDN 累积误差只作为假设。当前锁定配置与摘要匹配，普通 RMSNorm 的本仓库 loader 不折叠 gamma，forward 已只加一次1，因此不复制另一工程疑似双重加1。证据与范围记录于 `specs/cpro-structure-adaptation-verification.json`。

GDN 去掉 batch=1 单例维并用静态 Slice/Concat 替代三角 BOOL Select，保留 chunk64 和 FP16 边界。旧/新图在两种主机执行器各自逐 bit 对齐，并覆盖跨 chunk、NaN/Inf/正负零掩码。静态切片增加节点，不能据此推断速度收益。MoE selected 模式保留原路由，只执行选中专家的 Gather/BMM，并按专家编号升序逐路半精度相加；按 top-k 分数顺序累加或一次 ReduceSum 会改变舍入。选中权重仍有搬运成本，全权重存储仍未解决，不能称为原生 grouped GEMM 或实测加速。

可选 GDN 主机诊断固定相同 FP16 输入，对照上游 FP32 chunk 输出和状态，并分别报告输入舍入、非有限和逐 chunk 偏差。默认合成样例只验证诊断路径，不能确认真实模型误差根因；诊断加入 CPU 初始化和规模边界。用户随后要求先打通上板链路，因此精度诊断后置，不作为执行前置门槛。

最终目标已固化到 `specs/dflash-contract.json`：精度达标后 DFlash 至少提速2倍。工作测量口径为完整 decode 循环，计入 target/draft、verify、接受/拒绝、状态和搬运；同时报告 prefill+decode。两种模式须使用同板、同目标权重、同输入和缓存策略，前缀重算/全专家正确性基线不能独自支持最终优化收益结论。当前速度为 NOT_MEASURED。

经验保留在模型级合约、文档与回归检查；未修改通用技能或共享 DDK/SDK。拒绝把摘要“能运行”、小图数值通过、IR节点减少或合成误差数值晋升为完整模型精度、NPU覆盖率或2倍提速证据。后续按用户最新优先级取得双图真实执行结果，再恢复精度与性能闭环。


## 2026-10-26：将探针验证路径接入正式双图执行

用户要求先执行后精度。将此前用户 MatMul/Add 探针已通过的部署方式接入 `run_hdc.py`：仅本次上传集合排除 DDK HCL 两库，进程搜索路径增加板端 vendor 目录。排除前解析 runner/DSO 动态依赖，检查硬依赖、SONAME 别名及同字节替名；保留文件和构建记录仍校验哈希。运行前确认 vendor 目录存在且可读可进入，不改系统库。探针 A/B 仍要求原始完整库对，正式入口允许原本已裁减的集合；这种差异明确写入共用 guard 参数。

正式 HIAI executor 复用探针已验证的限定 ND 补维规则：只有原始逻辑维度后补1到四维可以适配，原始shape和非ND仍严格校验，实际descriptor用于AiTensor初始化。target输入S和draft输出K从前两维读取后仍经完整Add检查，任意reshape、前/中插1和rank5不会放行；target→draft传递仍按原逻辑字节数。

新增独立 smoke 运行路径，强制两图各一次，即使生成下一token为EOS也不跳过draft。报告准确标记 EXECUTION_COMPLETE，返回两组Top1数组，精度与性能为NOT_EVALUATED；不运行普通/DFlash token一致性比较、不把输入参考的历史PASS转成板端精度通过。Python收取层拒绝fake证据、不足两次图调用、身份和输出大小/范围不符，并保留退出状态、SHA256和所有权清理保护。旧benchmark仍按原阈值比较；HiLog抓取失败不覆盖真实执行状态。

主机原生调度、生命周期、缓冲区、13种IO布局和probe回归通过；Python传输协议与CLI相关回归见 `specs/hiai-model-execution-verification.json`。这些测试不连接板端。下一步只重编正式runner，沿用旧OMC/参考执行 `--execution-only`，以实际Load/Process及回传输出确认链路。完整35B质量、内存、算子位置和2倍提速均未通过。经验固化在模型文档、共用部署guard及可执行测试，未修改通用技能或共享资源。

## 2026-10-26：完整图恢复失败与日志上下文

run：`20261026T005353Z-17889-cf9a01`。用户最新正式 smoke 使用已裁减的 DDK 库集合与 vendor 搜索路径，仍在 target Load 返回1。原摘要只显示9803条关键词匹配的最后60条，多为析构及服务清理，遗漏了实际错误。补齐同时间窗 PID25399 日志后，确认 CPUCL 拒绝 FP16 MatMul，随后 CheckCompatibility、Recompile 和 RestoreFromBuffer 失败。缺少被有意排除的 DDK HCL 库之后已进入板端 CPUCL 初始化，因此不能把前面的 dlopen 错误重新解释为需要补回该库。activation.mode9 与 Cast 类型也有拒绝记录，但不能仅凭无节点名的日志定位到 GDN。

直接 GE 导出不设置 CPU 执行亲和，完整图与探针使用相同的 OMG 基本参数。小 MatMul 数值通过不能证明完整图中的不同形状、相邻算子或 CPU 分区兼容；更新 runner 不会改写旧 OMC。下一证据是与失败 manifest 哈希一致的 target 编译分区和节点，区分常量折叠、候选支持与最终分区，再做有依据的布局/子图改写。未改 FP16、运行库、编译选项或模型数学，精度与两倍加速目标保持后续门槛。

摘要现在按最新观察到的 app 会话候选保留头部、首条 E/F、Restore 失败前后及尾部，包含同 PID 的 MUSL，排除 HDC 和其他进程；去重后最多60行，每行1000字符，完整快照不改写。PID/date/init 分隔并未独立绑定当前调用，首条 E/F 也可能是成功回退，报告明确保留这些限制。无可解析候选时沿用关键词尾部。85项主机回归及独立审查通过，见 `specs/hiai-log-summary-verification.json`；这只是诊断修复，不是板端 Load 修复。经验留在本模型，没有修改通用技能或共享资源。


## 2026-10-26：NPU 分区超限导致整图 CPU 回退

run：`20261026T010136Z-21603-7ab64d`。用户补齐同一失败 manifest 对应的 OMG 日志：L19509/L31879 的 CPU 分区属于 `constNodeGraph`，主编译入口在 L63103–63109；L110650 明确记录 `Partition of type: NPUCL is more than 9, will rollback to CPU`，随后 CPU 优化、保存 OMC、OMG 返回成功。这把板端 CPUCL 拒绝 FP16 MatMul 与主图分区回退连接起来，不能再把早期常量图 CPU 分区作为主图证据，也不能把 OMG 成功当成 NPU 产物可运行。日志只给出大于9，未给精确分区数或全部断点。

本地蓝区 DDK 的只读审查发现相同回退分支会清空原分区再构造 CPU 图，且跳过正常分区计数日志；未发现公开可用的 cap/fallback CLI。该审查不是用户内部 DDK 的执行证明；主图根因由用户日志单独确认，不猜参数、不改厂商二进制。

新增标准库解析模块和显式 `--reject-cpu-fallback`：区分命名常量图、主图和未知作用域；只有明确阶段边界才退出常量图。OMG 返回成功仍检查完整日志，主图或未知作用域的明确 NPUCL 超限回退会保留 OMC、日志和转储，写 failure.json 并停止发布双图成功 manifest；常量图回退可继续。无该日志不能证明全图 NPU 执行。编译入口保持无 NumPy/Torch 依赖。

图适配只改变精确布局/元数据：ReduceSum/ReduceMean 的 axes 改为蓝区公开 schema 声明的 INT32，保留负轴与顺序、验证范围，数据仍 FP16；draft 的 KV 头复制改为四维 Gather，去掉五维广播，头顺序和非有限 FP16 位模式不变。结合已有 GDN 降维，目的是减少 CPU/NPU 分割；是否足以压到分区上限内仍需内部 OMG 验证，不预报收益。

相关主机参考、序列化、诊断和 CLI 回归通过，独立审查未发现阻塞，见 `specs/npu-partition-recovery-verification.json`。下一步复用旧 prepared 的不可变权重，由当前源码重新 export/compile 到新目录；参考和已更新的 runner 可以复用。当前保持 static 专家策略，不切 selected，也不为了 CPUCL 接受而改成 FP32。用户仍需回传新版编译与 execution-only 结果。精度排查后置，最终 DFlash 至少2倍目标不变，未修改通用技能或共享资源。


## 2026-10-26：重导出仍超限，先隔离归约支持限制

run：`20261026T011730Z-23027-2711b0`。用户沿78d94ef建议重导出后，主编译窗口91094–127615仍出现NPUCL分区数大于9回退。wrapper正确保留失败产物并中止，draft编译及上板命令未执行。用户支持表有13613节点，423个ReduceSum与14个ReduceMean均仅列CPU候选；7623个类型未知项包含大量常量和图边界，不能当作CPU计算断点。72个MatMul仅列NPU候选，不能通过整图CPU回退绕过。

发现上次INT32轴修正遗漏了scan展开内部生成的192个ReduceSum。现统一调用归约轴构造器，验证最终序列化tiny target/draft全部437/11个归约轴，而非只检查高级IR节点。数据仍FP16，前缀求和算法、求和顺序、模型权重与阈值不变。但本轮已有普通INT32归约仍为CPU候选，因此不能宣称补齐该遗漏就能解除分区回退。

用户补回v18与__ge/v2109/sum_1上下文：elementary_lib仅报CheckReduceSupport(185)的`check elem support fail`，ascendc_lib查不到对应TBE算子，其他库也拒绝。蓝区只读控制流表明这个消息是选中的V0/V1检查器失败后的公共出口；源行号不能跨内部版本推断分支。蓝区V0具有特殊格式要求，V1具有ND路径，但当前用户日志不足以认定格式或芯片能力为根因。另有64输入Concat与广播形状限制线索，均保留为待证假设，不直接改格式标签或厂商二进制。

新增标准库inspect-compile读取现有失败编译记录，校验AIR manifest、recipe、OMG日志哈希，只用精确节点名关联构造形状和边。常量文件内容不打开，优化后改名节点不做模糊映射；类型未知项单列。报告中的跨设备候选边只是定位提示，不是实际分区数。准备六个FP16、INT32轴小图，分别比较RMS/前缀Sum的rank与GDN修正Sum的keepdims，先独立编译判断支持，不触发整图或板端。具体case及验证见compiler-support-diagnostic-verification.json和direct-model-air指南。

当前内部DDK和设备仍由用户执行。主机回归只能验证解析、契约与参考，不能替代NPU编译和执行证据。执行链路优先、精度后置和最终DFlash至少2倍目标不变；通用技能与共享DDK未修改。


## 2026-10-26：归约 rank 对照失败后采用隔离 MatMul 候选

run：`20261026T013946Z-24443-de03bd`。用户只读检查成功读取13,613项完整支持表，13,606个节点精确匹配原recipe；排除7,616个边界后有538个CPU-only计算候选，另7个未确定角色项。437个归约中421个数据为FP16、16个为整数。192个scan axes仍为INT64是被检查AIR早于补丁的正常结果，检查器没有修改历史产物。六个rank/keepdims编译对照全部产生OMC，前五个主图NPU0/CPU1，第六个keepdims+Reshape为NPU1/CPU1，不能把后者解释为Reduce已在NPU执行；这些结果终止了本轮仅通过单例维和keepdims寻找解法的路线。

新增探针专用FP16单轴MatMul归约helper：保持已有FP16输入，移动归约轴、展平行，以全1右权重收缩，再恢复形状。Mean仅接受2幂轴长和可精确表示的FP16倒数，直接用1/N权重；不能先求出FP16 sum再缩放，例如16个4096求均值应为4096，而FP16中间总和溢出。GDN修正的原Mul输出舍入必须保留，不能顺手融合乘法与归约。整数归约拒绝，未使用零填充三角矩阵，以免0×Inf改变非有限语义。该候选保持实数公式及一个FP16归约出口，但不同kernel累加次序仍待设备数值验证，不能称逐bit等价。

六项覆盖RMS末轴16、head末轴8、scan末轴64，以及GDN非末轴64/7/1；K1仍保留实际MatMul，不靠恒等折叠冒充kernel通过。原53案例代码和输入/参考字节一致，当前默认总数59。新增候选只供独立探针，生产target/draft没有调用新helper。原Reduce图作为板端比较参考，主机独立double参考覆盖轴顺序、消去维、非有限、抵消、次正规数和Mean溢出边界；所有原容差不变。

为减少编译与上板往返，runtime probe增加显式--native-from，仅接受无native的COMPILE_COMPLETE批次并要求--replay-all和--refresh-runner。新批次提供OMC/输入/参考，旧批次单独提供完整LINKED构建与运行库库存；两DDK root/id/OMG hash必须一致，旧库/SDK/头文件/ELF与源产物均冻结核对。只重建当前小runner，candidate部署继续排除DDK HCL两库，不覆盖系统库或历史native。残缺native不能用donor掩盖，原A/B路径不变。主机110项运行协议测试与78项归约/案例测试通过，实际prepare-only生成六项；未运行内部OMG或HDC。

蓝区现有RmsNorm schema/config及kernel键是后续融合候选线索，但不能跨版本宣称用户internal支持，也未复制共享库或更改融合边界。完整模型仍需解决其他CPU-only MoE、拼接和广播节点；本轮小图成功不会自动解除整图分区上限。执行链路优先，完整精度后置且未豁免，DFlash至少2倍尚未测量。经验固化在模型文档、两来源验证和数值回归；没有修改通用技能或共享工具链。

## 2026-10-26：MatMul 归约加载通过，修正 rank3 ND 前补维检查

run：`20261026T015924Z-25481-adbe34`。用户报告六个隔离候选全部编译为主图NPU1/CPU0并Load通过，四维`reduce_mean_head_matmul`执行与数值通过。其余五项未进入Process：运行库报告三维ND输入或输出前补一个1至四维，而旧探针只接受尾补1。这是IO合同检查阻塞，不能记为kernel执行失败或数值不通过；当前报告没有证明所有算子的实际运行位置。

探针与正式executor同步新增严格的rank3 `[a,b,c]` 对应rank4 `[1,a,b,c]`，外围仍要求ND。原有exact和尾补1规则保持，不做全局squeeze、不按元素数相等放行。实际descriptor用于AiTensor初始化，文件与特征传递仍按原逻辑字节数；rank2 token合同及sequence/draft width推导不变。主机五组CTest通过，其中probe118项覆盖用户报告的形状、三种dtype、输入/输出独立前补、真实descriptor身份、非ND/换轴/中插维/非单例/越界rank拒绝，以及分配失败和短buffer。正式executor测试包含target前补而draft尾补的特征传递，保留首尾数据及生成token检查。这些是fake HIAI协议测试，不能代替设备数值复测。

下一步只刷新独立probe，复用`reduce-mm-20261026-101758`的六份OMC和原`OP_PROBE_RUN` native身份，写新结果目录，避免重复编译或覆盖旧证据。当前命令不重编正式runner，已有正式二进制不会自动获得此源码更新。生产target/draft归约未切换MatMul，数值阈值、FP16、运行库部署策略均未改。完整执行优先、精度后置与最终DFlash至少2倍目标不变。经验留在模型合同、文档及回归中，没有修改通用技能或共享资源。
