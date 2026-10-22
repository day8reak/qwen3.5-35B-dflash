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
