# 验证记录：恢复工作后的 FP16 / DOPT / HIAI 版本

完整主机验证基线 run：`20261020T092412Z-4528-622cba`，工作区相对路径 `.work/qwen3.5-35B-moe/20261020T092412Z-4528-622cba`。代码在 run 中验证后保存到模型目录并提交独立仓库；原始日志/工具/权重不进入仓库。

Python 3.10.12 / Torch 2.12.1+cpu / Transformers 5.13.0 / safetensors 0.8.0。用户 Torch 2.11.0+cu130 尚未执行本轮测试。使用官方哈希校验过的蓝区 next-6.1.1.0 tools 与 kirinx90 插件；HIAI 头文件编译使用已声明 OHOS Native SDK 的通用 ARM64 编译器，没有借用 Kirin9030 插件或设备结果。

| 检查 | 结果及范围 | 证据（相对当前 run） |
|---|---|---|
| 最终 Python 全测试 | **118 passed**，22 个 legacy ONNX tracer/导出提示；包含真实 DOPT 可选测试，未跳过 | `log/python-final.log`、`out/python-final.xml` |
| FP16 图内部检查 | 普通区域无残留 FP32/BF16；tiny 普通/DFlash token/stop 一致 | `tests/test_precision.py`、`out/fp16-demo.json` |
| 实际 DOPT 校准 | 缩小的真实 upstream MoE/GDN，所有32个专家矩阵覆盖；校准保存、校验后恢复 | `tests/test_quantization.py`、最终 pytest 日志/XML |
| 量化数值参考 | 使用真实 vendor Cython 算子的主机桥接；FP16普通区域；普通/DFlash一致 | 同上；并非原生NPU执行 |
| 激活量化溢出修复 | 首次集成失败保留；FP32量化缩放后的饱和值与独立高精度oracle一致，重新生成通过 | `log/python-tests.log`、`log/dopt-frozen-compare-v2.log`、最终 pytest |
| 完整35B架构规划 | meta tensor统计，无权重下载；10,240专家、20,480专家矩阵；21.19GiB权重估算 | `out/quant-plan-delivery.json` |
| AIR/OMG 编排 | 输入/外部权重哈希、短路径、失败清理、拒绝旧ABI/密集fake-quant图 | `tests/test_air_pipeline.py`；测试payload是mock |
| HIAI C++ scheduler / 生命周期与故障 | **2/2 CTest**，FP16缓冲首末元素传递；fake HIAI，不是板端 | `log/ctest-host.log` |
| 真实公开HIAI头文件 / OHOS ARM64 | **HEADERS_COMPILED**，未链接内部库 | `out/hiai-headers/result.json`、`out/hiai-headers/build.log` |
| 当前导出环境 | DOPT校准可用；TorchAir/torch_npu不存在；NpuQLinear未定义 | `out/preflight.json` |

首次测试中的 DOPT NaN 是真实失败，不能当作通过。修复把量化器内部缩放改为 FP32，输出恢复 FP16，并更新候选身份；其余浮点计算仍执行 FP16。该改动不代表全权重或硬件已通过精度门槛。

上一次 run 的真实 OMG 短路径负例已检查：进入 `ParseOriginMSOrAir2IrGraph` 后因无效 AIR 失败，说明绕过了原先输出路径拒绝；这不是转换成功证据。原日志位于 `20261020T062757Z-31423-15c0e2/log/omg-air-short-path-smoke.log`。

## 未通过的完整模型门槛

完整权重校准/质量、35B参考、真实 AIR/OMC、内部HIAI链接、X90加载/执行、峰值内存、十轮板端精度/性能及设备发布均 **PENDING**。板机由用户访问，不能用CPU/fake结果代替。`cpro-blue-pending` profile未登记，环境激活不算成功；只使用已声明模型解释器和既有run内蓝区工具。

当前18G free低于候选权重估算，仍需可用内存与实际峰值确认。量化普通/DFlash严格一致只验证调度；量化相对BF16的任务质量尚未冻结独立门槛。`hardware_reference_policy=use`，没有X90能力画像/工作量账本/依赖延迟，不宣称理论余量或加速比。

下一步在用户导出机器运行 README 中的 `preflight`。需要匹配本 DDK 的 TorchAir/CANN/torch_npu，尤其是量化扩展；先做 tiny AIR→OMC，再完整量化和内部HIAI构建。权重/库无需传出，保留失败日志和板端probe/报告即可继续定位。

## 2026-10-21：复用已有 AscendC 环境

补丁 run：`20261021T061137Z-18631-50cb54`。服务器指南省略已由 `set_ascendc_env.sh` 完成的环境变量配置，项目路径保留已有 `PYTHONPATH`，OHOS/HIAI 资源延后到交叉编译时填写。OMG 子进程优先使用本次 DDK，再保留继承的库与 Python 路径。

AIR 编排与 CLI 共 **17 项测试通过**；新增回归用假 OMG 子进程实际导入继承路径中的扩展，检查所选 X90 插件优先和 simulator 库路径保留，父进程环境不变。README 与操作指南的 shell 代码块均通过语法检查。证据及修改文件哈希见 `specs/environment-verification.json`；原 `specs/verification.json` 保留上一完整主机验证基线，未将其 118 项结果冒充本次重跑。

用户提供的日志仅记录内部 AscendC 环境脚本成功；未访问或运行内部 DDK，也未完成真实 AIR/OMC、HIAI 链接或板端执行。


## 2026-10-22：tiny 上板路径与停止规则

补丁 run：`20261022T025303Z-32164-0cf8ab`。新增 [tiny 完整操作指南](tiny-validation.md)，覆盖随机小模型的 CPU 参考、AIR、OMG、正式 HIAI 构建、传输、probe、benchmark 与回传 token/stop 比较，不需要完整权重或 tokenizer。

修正 tiny AIR manifest 的 EOS：原先 `[62]` 与 `demo` 的空 EOS 不一致，可能让板端提前停止；现在两端都使用空 EOS。新增回归从 CLI demo，经 AIR/OMC 测试替身和板端命令生成，强制预测 token 62，验证普通与 DFlash 都生成四个 token 并以 `length` 停止，空 EOS 参数可经 shell 引号完整传递。

本次 AIR 编排与 CLI 共 **18 项测试通过**，证据为 run 内 `pytest.xml`。重新执行真实 FP16 tiny CPU demo（S=8、prompt=1,2,3、生成4个 token），普通/DFlash token 与停止原因一致，`tiny-reference.json` 为 `PASS`。Python 环境仍为声明的主机环境；未安装 TorchAir/CANN。

以上没有真实 AIR/OMC、内部库链接或 X90 执行证据。`cpro-blue-pending` profile 仍未登记，目标环境激活失败；测试仅使用资源清单中已声明的模型解释器。当前 NVIDIA/CPU 服务器无昇腾卡的 AIR 导出依赖尚未验证，tiny 路径也不能绕过此缺口。


## 2026-10-22：服务器缺库诊断与内部 DDK 布局

补丁 run：`20261022T030203Z-32536-b061e3`。根据用户共享日志，服务器已安装 TorchNPU `2.11.0.post2`，导入失败原因是缺少 `libhccl.so`；内部 DDK 版本已确认 `100.520.020.010`。这是用户反馈，未登录该服务器或安装其 CANN。

CLI 与包初始化改为按需加载模型依赖；缺库时 `preflight` 能输出版本和完整异常原因，OMG 转换、命令生成和 HIAI 构建入口不再为哈希工具强制加载 Torch。AIR 检查显式导入 TorchNPU 后访问其 TorchAir，避免关闭自动加载后误报未找到捆绑的 TorchAir。内部 OMG 不再硬性要求公开插件目录，保留内部环境脚本提供的路径。芯片支持仍须由真实转换确认。

隔离源码副本的 Python 全套回归 **129 passed、2 skipped**，跳过的是未配置真实 DOPT 路径的可选集成测试；没有重用历史 DOPT 结果冒充本次执行。证据：run 内 `lazy-import-check/full-results.xml`、`full-tests.log`。新增测试用失败的 Torch 导入模拟 `libhccl.so` 错误，检查报告、帮助命令和内部无 `tools/platform` 的 OMG 编排。随后确定性资源夹具更新的 3 项 preflight 测试重新通过，最终 12 个 Python 文件与测试树哈希一致。真实主机 `preflight-host.json` 成功生成，并如实报告本机没有 TorchNPU/TorchAir。文档 38 个 shell 代码块和 7 个 Python heredoc 语法通过。

真实 AIR/OMC、内部 HIAI 链接和 X90 执行仍为 **PENDING**。公开 CANN 9.1.X 仅是该 TorchNPU 的版本对应关系，不能证明无昇腾卡导出或与内部 OMG 的兼容性。`cpro-blue-pending` profile 仍未登记，本次目标环境激活失败，验证使用已声明的模型解释器。


## 2026-10-22：保留 TorchNPU 的 CPU 参考路径

补丁 run：`20261022T034854Z-1269-95b99d`。用户最新日志确认：安装自己的 CANN 9.1.0 后仍缺 `libhccl.so`，Transformers 的显式 NPU 探测使 CPU demo 在关闭 Torch 自动加载后仍失败；卸载 TorchNPU 后，CPU tiny demo 成功写出参考报告。用户随后要求改脚本，以保留 TorchNPU 供 AIR 使用。

CLI 的 `demo/reference --device cpu` 或 `cuda:*` 现在于模型导入前应用当前进程内的参考后端策略：关闭 Torch 自动加载并替换 Transformers 的 NPU 可用性探测，不修改厂商包文件。AIR 入口保持真实依赖检查，并拒绝在同一进程继承已经应用的参考策略。`preflight` 独立子进程检查 CPU Qwen/DFlash 类导入，父进程保留自然 Torch/AIR 错误，因此原始 Torch 导入失败也不会掩盖可用的 CPU 参考路径。

隔离源码副本的完整 Python 测试 **141 passed、2 skipped**；跳过真实 DOPT 可选集成。环境为声明的 Torch 2.12.1+cpu / Transformers 5.13.0。新增测试以可发现的 TorchNPU 分发信息和 `torch.backends` 入口模拟缺失 HCCL，确认未经修正的 Torch/Transformers 导入真实失败；修正后的 tiny CPU 普通/DFlash token 与干净环境完全一致，停止原因一致。AIR CLI 仍拒绝缺库环境，CPU/CUDA 策略后的同进程 AIR 也明确拒绝。CUDA 仅验证后端选择，未进行 GPU 执行。

证据：run 内 `audit-preflight-full.xml`、`audit-preflight-validation.json`、`audit-preflight-source-hashes.json`。最终 6 个 Python 文件与测试副本哈希一致；40 个 shell 代码块、7 个 Python heredoc 语法通过。目标环境 profile 仍未登记，激活未成功；没有对用户内部 CANN、HIAI 或 X90 执行验证。真实 AIR/OMC 和板端门槛仍为 **PENDING**。


## 2026-10-22：直接 GE 构图保存 AIR 的独立样例

补丁 run：`20261022T064712Z-4550-c5859a`。用户选择继续 AIR 路线；新增 `direct_air_smoke.py export/compile` 与 `ge_add_air.cpp`，只构建 FP16 ND `[1,4]` 的 Data/Const/Add 小图，保存后重新加载检查描述与常量。该入口不加载 Python 的 Torch/TorchNPU/TorchAir，不调用 ACL/GE Session；与 Qwen target/draft 双图分别标识。完整模型的直接 GE 导出和单图 HIAI runner 尚未实现。

C++ 已使用官方 `v9.1.0` 的 GE 与 metadef 头文件执行真实编译检查：GCC 11.4.0、C++17、ABI0、`-Wall -Wextra -Werror`，无警告。GE 提交为 `0ef703178896cbeb2a66e7c751ac9626df7f7439`，metadef 为 `01966fe61d57fa7ae85b1c9e3350cecfe3a125a6`。ABI0 有同版本 GE 主机构建配置依据。这里只生成对象文件；本机没有匹配的服务器 `libgraph.so`，没有链接或执行生成器。

新增入口、既有 AIR 编排和轻量导入的定向测试共 **29 passed**；最后调整显式资源清单优先级后，新入口 **13 项重新通过**。这些测试覆盖编排、FP16 fixture、资源选择、哈希/ABI 拒绝、无 Torch 导入、OMG 非零退出/缺失或空输出、日志与临时目录清理，工具边界使用测试替身。42 个 shell 代码块、7 个 Python heredoc 语法通过。证据及最终文件哈希见 `specs/direct-air-verification.json`，原始证据保存在本 run。

真实 GE 链接、AIR 保存/读取、内部 OMG 接受该 AIR、OMC 和板端执行均 **PENDING**；不能将头文件编译或测试替身的状态作为实际 AIR 成功证据。`cpro-blue-pending` 仍未登记，目标环境激活未成功，板机仍由用户访问。全模型质量、量化和内存门槛未推进，`hardware_reference_policy=use` 未改变。

下一步按 [直接 AIR 样例](direct-air-smoke.md) 在用户服务器加载自己的 CANN 环境运行 `direct_air_smoke.py export`，成功后切回内部 DDK 环境运行 `compile`。失败时保留对应日志和 `failure.json`，不要先执行不兼容的双模型板端命令。


## 2026-10-22：修复 GE 节点查找与失败日志

修复 run：`20261022T072928Z-5968-3afd6d`。用户更新至 `94c053e` 后，自己的 CANN 9.1 环境已能构建并启动 `ge_add_air`；失败发生于保存前的 `VerifyGraph`，具体为 `FindOpByName("x")` 返回 `4294967295`。这是用户执行证据，不是本地已链接 GE 的结果；尚未生成成功 AIR。

已核对官方 GE `v9.1.0` 提交 `0ef703178896cbeb2a66e7c751ac9626df7f7439`：`SetInputs` 构造 `compute_graph_`，`FindOpByName` 只查询 `op_list_`，而 AIR 回读不恢复后者。原样例使用了错误的查找入口。修复后两个验证点统一使用 `GetAllNodes`/`GNode`，保留类型、形状、FP16、Data index 和 Const 原始字节检查；不增加设备初始化，不修改计算或阈值。

修复后的 C++ 通过真实官方 9.1 头文件编译检查（GCC 11.4.0、C++17、ABI0、`-Wall -Wextra -Werror`）。17 项脚本测试通过，覆盖用户错误在终端和失败报告中的可见性、日志截尾但完整文件保留、部分 AIR 不得成功、超时/异常链、OMG 失败与已有输出保护。工具边界仍使用测试替身，不代表实际 GE/AIR/OMG 通过。

证据及最终源码哈希见 `specs/direct-air-lookup-verification.json`。共享对话原文只保留在本 run，不进入仓库。两次独立源码审查确认 GNode 属性读取不依赖 `AddOp` 表；没有把日志中内部 DDK 的 `libstdc++` 路径当作本次根因，也没有自动替换或修改库。

用户需按 [重试说明](direct-air-smoke.md) 使用新输出目录，确认保存前与读回后检查都通过，再运行 OMG。修复版真实 GE 执行、AIR、OMC 和板端验证仍 **PENDING**；目标 profile 未登记，环境激活未成功，完整模型阶段与硬件参考策略未改变。


## 2026-10-22：直接 GE target / draft 双图实现

run：`20261022T074906Z-6507-ca8b2b`。用户报告 Add 样例已生成 AIR 和 OMC，记录为 **USER_REPORTED_PASS（仅 Add）**；未收到该产物哈希或板端结果。新增 target/draft 直接 GE 路径，原 PyTorch、TorchAir、DOPT 和 HIAI 代码保留。

完整 target 结构 S=8 有 230,203 个 IR 节点、693 个权重定义，全部形状与 Transformers meta 模型匹配，权重键完整覆盖锁定的文本模型检查点索引。draft 包含原六层和目标共享 embedding/head。没有下载或加载完整权重。目标 FP16 权重约 69.32 GB，target/draft 唯一 FP16 权重约 70.09 GB；不是可部署到 32GB 板机的量化成品。

参考比较包含三个不同有效前缀、完整目标特征与 Top1、最后一层归一化语义、65-token 跨 GDN chunk 状态传递，以及 draft 的因果/双向掩码。PyTorch 原语执行器用于隔离图定义与 NumPy 内核舍入差异；严格 FP16 特征检查和 Top1 一致性通过，没有放宽门槛。实际 prepared 权重还通过普通/DFlash 调度比较，所有 target/draft 调用的 Top1 与保留的 FP16 PyTorch 路径一致。NumPy 结果只作语义诊断，不代表 GE 或设备浮点实现。

GE 的 40 种算子及 Cast 接口依据固定版本的官方原型核对。已处理 BOOL 不在部分 BasicType 算子类型集合内、TopK values/indices 共享同一节点、混合维数 MatMul 以及 Python 浮点标量过早转 FP16 的问题。实际 C++ 配方解析器已读取两张 tiny 图的内嵌和 FileConstant 模式，分别为 8,385 / 405 个降低后节点；解析模式没有 GE 执行，也不产生 AIR。

完整结构的图描述约 81 MB、常量约 80 MB；通用 C++ 生成器仅 18,513 字节。最终生成器使用真实 GE v9.1 头文件、C++17、ABI0、`-Wall -Wextra -Werror` 编译通过，最终检查用时约 1.27 秒。这里完整权重路径是明确的不可用占位路径，仅为编译检查；没有链接 GE 或生成模型 AIR。先前展开全部代码的 tiny 编译在约 143 秒、3.2 GB 编译器 RSS 时停止，未被记录为成功，已改为配方模式。

图描述不是 AIR。真正导出必须执行 GE SaveToFile，并在保存前、回读后检查节点、边、端口、属性、描述和常量内容；全部通过后才发布 AIR。外部权重及构图源码/recipe/blob 都有哈希记录，失败不产生成功 manifest。权重聚合内容哈希沿用现有 AIR→OMC→参考比较的 source 身份。

隔离源码副本的完整回归为 **222 passed、2 skipped**；跳过的是未配置真实 DOPT 路径的两项可选集成测试。前两轮回归发现两处新增参考测试的进程后端设置泄漏到旧 TorchAir 测试；修复仅限新增测试的状态恢复，原生产保护逻辑不变。修复后全套重新通过。修改文档的 43 个 shell 代码块和 5 个 Python 代码块/heredoc 语法通过。证据为本 run 的 `final-closed.xml`、`final-closed-source-hashes.json` 和 `docs-syntax.json`。

完整模型 AIR/OMG、内部 FileConstant/算子支持、HIAI 上板及真实数值结果仍为 **PENDING**；新直接 GE 的原生压缩整数权重映射 **NOT_IMPLEMENTED**。全模型量化、质量、内存和性能门槛不推进。硬件参考策略仍为 `use`，没有可访问的 X90；target profile 仍未登记，环境激活未成功。

源码与可复核证据摘要见 `specs/direct-ge-model-verification.json`；下一步按 [直接双图指南](direct-model-air.md) 在用户服务器 prepare tiny → export AIR → 内部 OMG，然后执行相同 ABI 的 HIAI 验证。


## 2026-10-22：修复直接 GE 的 Data 工厂注册

修复 run：`20261022T084136Z-11227-7c2c63`，代码基线 `aa8c6e1`。用户服务器完成 tiny `PREPARED` 和生成器编译，首次 target 构图失败；首条日志为 `BuildGraph: tensor_desc is null`，出现两次，随后报告输入缺少 Data。失败发生在 AIR 保存之前。

固定版本 GE 9.1 源码表明，`BuildGraph` 会先从 `OperatorFactory` 创建输入类型原型，再判断是否为 Data；仅在实例上注册端口不能满足此要求。之前成功的 Add 样例已使用 Data 工厂注册，而双图生成器遗漏了它。修复同时覆盖配方与小图展开生成器，补充四种 IR 输入 dtype 的 Data 工厂定义、实际工厂原型检查和 `SetInputs` 后即时失败检查。模型计算、配方格式、权重与 ABI 均不变。

隔离源码副本的定向回归 **48 passed、2 skipped**，包含真实 C++ 配方解析、两种生成模式的真实 GE 头文件编译、旧 prepared 复用、损坏产物拒绝及错误日志提取。两项跳过的是需要真实 GE 主机库的 Save/Load 测试；本机未链接或执行 GE。编译使用固定 GE/metadef v9.1 头文件、C++17、ABI0、`-Wall -Wextra -Werror`。未改动模型数学，因此未重复此前全套数值测试。

日志包装器同时展示前几个错误和有界尾部，保留完整文件；回归复现了首个错误被数百行 GE 错误码初始化日志挤出尾部的情况。源码与证据摘要见 `specs/direct-ge-data-factory-verification.json`。目标 profile 仍未登记，硬件参考策略为 `use`，没有板机访问权限。修复版真实 target/draft AIR、OMG 和 HIAI 结果仍待用户重试，未推进完整模型或性能门槛。

重试命令见 [直接 AIR 指南第 2 节](direct-model-air.md#2-服务器生成两张-air)：复用原 prepared，在新目录导出，并将该目录用于后续 OMG。


## 2026-10-22：修正 Data 原型的输入描述检查

run：`20261022T092815Z-12515-a03008`，基线 `6590f77`。用户日志已显示 Data 工厂注册成功，但上次新增的 `TryGetInputDesc("x")` 检查返回失败。GE 9.1 源码的调用链为 `Operator::TryGetInputDesc` → `OperatorImpl::InputIsSet` → `OpDescImpl::InputIsSet`；最后一项要求形状维度数大于零。工厂原型默认 rank 0，因此该接口不能用于端口是否存在的检查。这是上一补丁引入的误判，发生在实际模型构图前。

两种生成器均移除这项原型描述查询，保留工厂注册/实际创建、类型与端口数量检查。实际模型 Data 的名称、描述、边及 AIR 保存/回读检查不变。定向回归 **48 passed、2 skipped**，包括两种生成器的真实 GE 9.1 头文件编译；两项真实 GE Save/Load 仍因本机缺少匹配主机库而跳过。模型数学、权重、配方和 HIAI ABI 未变化，旧 prepared 可复用。源码与证据见 `specs/direct-ge-prototype-check-verification.json`。修复版服务器 AIR/OMG 和板端结果仍待验证，目标 profile 和硬件参考策略不变。


## 内部 OMG Select / 常量 Add 修正（2026-10-22）

run：`20261022T093951Z-12973-586101`，基线 `0097330`。用户报告 tiny target/draft AIR 成功，但 target OMG 转换失败；用户日志指向 Select 广播维度检查及 RMSNorm 常量 Add 的 CPUCL 类型检查。生成器已显式广播 Select 三输入，并将 norm 的 `1 + weight` 表达为单输入 Power shift。源码保存的输入均为 FP16，尚不推断 OMG 的类型变化来源。

本地相关 9 个测试文件共 **83 passed**：AIR/OMG 成功与故障编排、轻量导入、生成图及双图参考、Select 序列化描述与数值、RMSNorm 全部 63,488 个有限 FP16 位模式逐位等价、日志摘要。测试验证的是主机代码、CPU 参考和模拟工具边界，没有执行真实内部 OMG。独立审查未发现阻塞问题。

状态：旧修复的 AIR 成功为用户报告，未取得产物哈希；本次生成器修改后的 AIR/OMC 需要重试。HIAI 板端、全权重、原生量化和性能门槛仍为 PENDING。不能将 83 项本地测试解释为 OMC 已通过。证据及源文件哈希见 `specs/direct-ge-omg-select-verification.json`；服务器复用 prepared 的重试命令见 `docs/direct-model-air.md` 第 3 节。


## 原生 Softplus 与有限常量预计算（2026-10-22）

run：`20261022T095753Z-14065-91325c`，基线 `9ff66b2`。用户 PID157178 的 target OMG 日志明确报 SoftplusV2 缺少推导函数，同时 CPUCL 报 32B/64B 输入长度不匹配。更新的生成器使用受参数约束的原生 Softplus；只对小型静态一元算术预计算 FP16 结果，重新分析依赖并删除已替代的静态链，不改变原始 prepared 权重。

本地 11 个相关测试文件结果为 **136 passed、4 skipped**。通过范围包含全部有限 FP16 的 Softplus 阈值舍入等价、生成 recipe 的真实写入及 C++ 读取器、完整 tiny target/draft 的 Select/Softplus 描述、有限常量预算与逐节点舍入、1/3/8 有效 token 的 tiny target 输出逐字节一致、旧 prepared 导出流程及日志/编译故障处理。4 项跳过需要实际 GE 头文件/运行库，未执行真实 GE/OMG 或板端。

独立审查检查了常量来源、liveness 与引用、参数限制、未触碰大专家权重、shape 回显过滤范围。原生 Softplus kernel 的目标精度以及 CPUCL 缓冲区失败是否消失仍需服务器重试；不把数学 oracle、主机参考或 mock 工具通过解释为实际 OMC 成功。完整35B、原生压缩量化和设备性能门槛不变。证据见 `specs/direct-ge-softplus-verification.json`；重试命令见直接 GE 指南第3节。
