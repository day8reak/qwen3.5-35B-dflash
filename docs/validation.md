# 验证范围与保留结论

[首页](README.md) · [当前更新](current-status.md) · [项目结构](project-structure.md)

本页记录当前实现能证明什么。安装、编译、加载、执行、数值一致、实际执行位置和性能是不同检查；主机测试不作为设备通过记录。

## GDN诊断第二轮HDC通信失败与续跑

2026-10-27，用户回传 `gdn-diagnostic-c0cbac5ac9cf`：`0x00` 首轮两例执行完成，进入 `0xA5` 轮后在 `upload` 阶段失败。嵌套错误是 `HDC verify-checksums missing unique remote exit marker`，HDC stdout为 `[Fail]Failed to communicate with daemon`。该轮尚未进入GDN执行；这次失败是传输问题，不能判为算子Process失败，也未提供新的完整三轮数值结论。原始远端文件仍未本地取得。

修正诊断总报告隐藏具体异常、失败时不展示已完成轮次的问题；细分runner复用和库选择阶段。新增 `--resume`，在新目录重新校验原结果、每个回收文件及runner构建/上传哈希，保留同一设备、runtime、firmware、模型及输入，复用首轮runner，只跑缺少轮次，原文件不改。只对fused探针的只读checksum查询启用精确daemon通信故障的有限重试（最多3次，每次≤30秒，间隔1秒）；不重传、不重跑Process、不重启daemon，真实checksum失败不重试。完整模型上传默认超时和单次行为不变。

本轮147项主机测试通过：17项GDN诊断/续跑及130项板端编排与HDC传输回归。覆盖首轮保留而只执行余下四次Process、已完成批次再次核验不执行、输出/runner篡改拒绝、真实构建记录自动库发现的连续复用、daemon短暂故障恢复/持续失败、真实checksum失败不重试及完整模型默认超时不变。工作流JSON、4段当前文档Bash语法和CLI帮助通过。工作区证据为 `20261027T115942Z-26038-e68837`；主机使用模拟HDC/runner，没有验证真实连接已恢复，也没有新增GDN精度通过结论。

## 环境恢复后九例探针最新回传

2026-10-27，用户回传新批次 `fused-all-f6b81ead7ae8`：恢复后的环境已重新尝试全部九例，GDN、CacheUpdate、Attention各prefill/decode共六例完成AIR导出、OMC编译、加载和执行。以下来自用户粘贴报告，原始产物未在本地主机重新获取、校验或复跑。

| 用例 | AIR导出 | OMC编译 | 加载 / 执行 | 数值 |
|---|---|---|---|---|
| GDN prefill/decode | PASSED | PASSED | PASSED / PASSED | DIAGNOSTIC_COMPLETE；最新回传核心输出均全零、relL2=1 |
| CacheUpdate prefill/decode | PASSED | PASSED | PASSED / PASSED | SHAPE_ONLY_UNVERIFIED_CONTRACT；未验证精度 |
| AdnFusedInferAttention prefill/decode | PASSED | PASSED | PASSED / PASSED | SHAPE_ONLY_UNVERIFIED_CONTRACT；未验证精度 |
| ConvStateUpdate prefill/decode | PASSED | FAILED | NOT_RUN / NOT_RUN | NOT_RUN |
| AdnRmsNorm | PASSED | FAILED | NOT_RUN / NOT_RUN | NOT_RUN |

Attention已完成两种调用，先前的端口校验与导出器const接口错误未再阻止该批次。Conv仍在rank-4的y2（`[1,8192,1,S+3]`）维推后遇到格式配置失败；RMS仍找不到 `AdnRmsNorm` 维推函数。历史五类调用未全部通过，不能据此认定整个DDK没有相关kernel；随后用户选择取消Conv/RMS自定义依赖，原失败保留，不改为通过。

用户随后粘贴同批次 `board-1/comparison/report.txt`，GDN数值如下；原始远端文件未取得本地核验：

| 用例 / 输出 | 非零元素数 | relative L2 | 最大绝对误差 |
|---|---:|---:|---:|
| prefill core_attn | 0 | 1 | 0.019012451 |
| prefill last_recurrent_state | 33172 | 0.9603274111031198 | 0.08654785 |
| decode core_attn | 0 | 1 | 0.005325317 |
| decode last_recurrent_state | 23424 | 0.9745207122180418 | 0.040130615 |

`DIAGNOSTIC_COMPLETE` 只表示诊断已生成。两例核心输出全零、末态大偏差确认精度问题仍在；Cache/Attention仅做形状检查，不能归为accuracy通过。实际NPU执行位置、完整模型与连续状态精度也未验证。

下一步入口为 `run_gdn_diagnostics.py --probe-run ...`：复用原两例OMC和输入，runner只构建一次，分别以 `0x00/0xA5/0x5A` 预填输出后各执行一次，共六次Process。将记录输入缓冲区回读SHA256、输出对象/容量/填充值残留、三轮输出差异，并复制所选DDK的GDN cpp/py/h与tiling证据。不重新编译模型、不执行平台诊断、不恢复Conv/RMS候选。该诊断用于检验输入交接、输出写回和重复执行行为；尚无真实设备回传，不代表问题已定位或解决。

主机独立参考核验已将当前探针输入与固定环境的Transformers recurrent/chunk实现对照，FP16结果的relative L2均小于 `1.6e-5`。prefill/decode参考核心输出全部非零，分别有24431/4022项处于FP16正常数范围；最终输出舍入下溢解释不了板端全零。旧初态在128×128两轴严格对称，尚不能检验初态转置；如缓冲诊断排除交接问题，后续需要非对称初态读入用例。核验及复现脚本保存在本次工作区运行记录 `20261027T113904Z-21762-792bc2/gdn-reference-audit/`，没有执行DDK或设备。

GDN诊断及原编排共63项主机测试通过（初轮62项，增加有效前缀检查后最终诊断13项复验）；C++编译和5组CTest通过。模拟故障覆盖输出未写/部分写/写零、输出对象或缓冲被替换、后续输入或输出填充覆盖先前输入、输入回读/回收文件损坏、FP32末态解析及padding不误判。所有输入在拷贝和输出填充完成后、Process前统一回读；对象替换或输入不符时保留诊断并停止。三轮复用同一runner、OMC和输入，原文件不变。源码与编译证据有文件数/字节限制，仅复制可读文件，不保证找到所有tiling实现。上述主机测试及内存观察不证明设备缓冲一致性、kernel实际执行或数值正确。

本次补齐 `--fresh` 中Conv/RMS失败后的只读检查，并新增 `run_fused_probes.py --inspect-only --probe-run ...`：支持探针目录及all-ops批次，打印关键日志与安装注册文本摘要，保留完整复制证据；已成功重试的历史失败不再触发自动检查。29项板端编排主机测试通过，覆盖禁止编译/TBE/runner/HDC、原产物与DDK不变、损坏产物拒绝、模式冲突及fresh/重试回归。CLI帮助、29个文档Bash代码块及diff检查通过。本地验证使用模拟编译/设备夹具，未在用户DDK或设备执行，也未修复Conv/RMS本身。

## 当前模型算子选择与保留的Conv/RMS失败

用户随后回传 `fused-inspect-bc256ae0b0f4`：检查扫描176个文件、`limited=False`，未找到Conv/RMS注册文本。该结论只覆盖扫描范围，不证明整个DDK缺少实现；原始证据未在本地重新获取或校验。Conv的rank-4 y2后续格式配置失败及AdnRmsNorm维推缺失仍是当前已观察到的失败点。

用户明确决定取消Conv状态更新和RMS的自定义算子依赖，采用本仓库已有基础算子路径；保留GDN、CacheUpdate、Attention三类六例。原工程的自定义Conv用于Cpro，AdnRmsNorm用于ND，Cpro RMS调用内置 `npu_rms_norm`，这两种原工程路径不能混为同一自定义依赖。

`run_fused_probes.py --model-ops --probe-run ...` 复用保留批次中的六例OMC，只检查并上板这六例；`--all-ops` 仍可作为历史九例的可选诊断，原Conv/RMS失败不变。生产GE使用 `--fused-ops gdn`，沿用基础算子的Conv历史拼接/截取和RMS；当前融合接口仅支持 `gdn,conv_state,rms`，六例探针通过不表示CacheUpdate/Attention已接入生产GE图。

本次选择改变依赖范围，不构成基础Conv/RMS在完整模型目标执行或数值精度上的通过证据。GDN历史数值偏差、所选六例精度、NPU执行位置、完整模型状态连续性和性能仍待验证；不继续新增Conv/RMS融合候选。

本轮200项主机测试通过：35项板端编排、43项探针与筛选、122项GE目标构图及导出回归。新增覆盖旧Conv/RMS即使已编译也不上传、不执行、不参与模型模式比较；fresh模型模式只编译六例；旧清单和脚本保持不变。抽取后的板端脚本在全部512个用例子集上与旧生成器逐字节一致。34个文档Bash代码块、工作流JSON、CLI帮助及diff检查通过。GE/OMG/HIAI使用主机替身；没有新增真实DDK编译或上板证据。现有六例OMC未变，用户随后回传的 `board-1` 数值见上表；这些主机测试不覆盖新的GDN缓冲区诊断。

## Docker环境恢复与删除产物后的重建

2026-10-27，用户说明Docker退出，且运行目录除权重外已清空。此前粘贴的编译/上板结果继续保留为历史证据，旧路径不再作为可复用产物。新增 `source targets/cpro/scripts/env.sh`，私有配置存于仓库旁并在检查成功后恢复变量；`run_fused_probes.py --all-ops --fresh` 从选定DDK源码绑定开始重建平台诊断和九例探针。

按用户随后要求，配置仅保存固定环境，不再收录历史产物路径或运行时身份覆盖值；这些变量由调用命令提供，source保持调用者值不变。主机环境恢复10项测试通过，覆盖配置首次创建且不覆盖、不收录运行变量、重复source路径稳定、带空格/特殊字符的路径、保持调用者运行变量（即使旧配置含同名项）、配置错误时不改变调用者环境，以及显式HIAI自动查找对主机库、非OHOS路径和歧义候选的排除。板端编排主机回归22项通过，新增覆盖删除旧路径后的初始化、九例仅编译一次、初始化失败保留阶段记录、新批次重试自动继承诊断、原生平台更新成功时直接编译。GE/OMG/设备使用替身；本次没有真实目标执行。Bash语法、CLI帮助及diff检查通过。

进一步按用户要求复用已激活的vendor/conda环境：默认配置缩减为七项并附中文解释，DDK根目录取 `DDK_PATH`、Python取当前conda、HDC从PATH查找，已有vendor模块及库路径保留。环境回归更新为11项通过，新增覆盖无重复配置时的派生值，以及切换DDK后不沿用旧的派生 `CPRO_DDK_ROOT`；显式配置仍优先。未实际运行vendor环境脚本或目标工具。

随后将当前指南的执行命令统一为 `python3`，项目入口移除 `MODEL_PYTHON` 配置依赖并保持调用者PATH不变。更新后的11项环境测试通过，覆盖未设置解释器变量、旧配置含失效Python路径、首次生成配置及重复source；6份当前指南的41个Bash代码块和 `env.sh` 语法检查通过。仅验证主机配置与命令语法，未重新编译或上板。

用户随后回传HIAI自动选择为0，候选位于 `tools_sysdbg`、`ddk/ai_ddk_lib`、`ddk/hiai_pipe_lib` 的 `lib` / `lib64`。旧逻辑要求路径包含 `ohos`，因此在读取实际ELF前就排除了这些文件。已取消该目录名条件，按ELF架构、已知GLIBC依赖和真实路径去重筛选；多候选继续要求明确选择，并逐项说明原因。16项主机环境测试通过，包含无ohos目录名、ARM/x86排除、真实readelf读取GLIBC依赖、符号链接去重和多候选拒绝。尚未取得用户库的ELF/链接结果，也未恢复此前成功上板使用的确切HIAI库路径。

用户再回贴旧回答，补齐配置：`ddk/ai_ddk_lib/include`，以及 `ddk/ai_ddk_lib/lib64` 下的 `libhiai.so:libhiai_ir.so:libhiai_ir_build.so`。据此优先识别完整三库布局，避免将三个目录里的单个 `libhiai.so` 继续当作未知选择；文档给出独立于 `REPO_DIR` 的私有配置恢复命令。20项主机环境测试通过，新增完整三库恢复、排除其他同架构副本、缺库/错误架构拒绝及显式配置优先。该记录来自用户粘贴的旧配置，不据此宣称三库的OHOS ABI或设备兼容性已验证。

首次生成私有配置时现自动写入HIAI头文件与三库两项，保留DDK变量引用供加载时展开；生成前已有显式HIAI路径则原样保存。仍只需手填原来的七项，已有配置不覆盖。22项环境测试通过，覆盖模板展开、切换到含空格的新DDK路径、未加载DDK时引用配置内根目录，以及后续不再手动exportHIAI路径。

## Attention导出器C++接口修正

用户回传 `fused-all-9759eb4f9b0b`：Attention两例在构建导出器时失败，尚未进入GE序列化或OMG。CANN 9.1的 `GNode::GetInputIndexByName` 为非const方法，新增helper误传 `const GNode&`。现改为可变引用；主机测试的同名方法也按用户日志中的真实声明改为非const，先复现旧代码编译失败，再验证修正。此前测试夹具错误地将该方法声明为const，未能拦截这次回归。

该批次复用的GDN/Cache再次完成加载/执行，数值验收仍未通过或未定义。Conv/RMS未重新编译；只读扫描未找到匹配的注册文本，这不证明整个DDK缺少实现。原始产物未在本地重新核验。

## 五类算子早期批量回传

2026-10-27，用户回传 `fused-all-97b7612ef879`，来源为粘贴总表，原始产物未在本地重新核验：

| 用例 | AIR导出 | OMC编译 | 加载 / 执行 | 数值 |
|---|---|---|---|---|
| GDN prefill/decode | PASSED | PASSED（复用） | PASSED / PASSED | DIAGNOSTIC_COMPLETE；此前明显偏差尚未解决 |
| CacheUpdate prefill/decode | PASSED | PASSED | PASSED / PASSED | DIAGNOSTIC_COMPLETE；无golden，不能判精度通过 |
| ConvStateUpdate prefill/decode | PASSED | FAILED | NOT_RUN / NOT_RUN | NOT_RUN |
| AdnRmsNorm | PASSED | FAILED | NOT_RUN / NOT_RUN | NOT_RUN |
| AdnFusedInferAttention prefill/decode | FAILED | NOT_RUN | NOT_RUN / NOT_RUN | NOT_RUN |

全部执行位置仍为UNVERIFIED。随后完整日志确认：Attention两例报 `GE recipe failed: input port order mismatch`；导出器将8个连接端口当成完整18个注册端口校验，需保留可选端口间隔，修正后重试真实GE/OMG和上板。conv宽度4已通过初始维推，y2分别为 `[1,8192,1,11]` / `[1,8192,1,4]`，后续报 `Not supported OP`（`generated_lib`）及 `Get format configs failed`；这是本次编译路径的格式配置失败，不能直接断言整个DDK没有kernel。RMS仍报 `AdnRmsNorm` 维推函数缺失。conv/RMS未观察到TBE兼容适配调用（0次），不应归因于此前GDN的平台更新错误。

本次已修正Attention导出器按注册槽位校验描述符和连边。主机C++回归运行实际生成的校验函数：旧逻辑复现端口顺序失败，新逻辑接受可选槽位空缺，并拒绝错序、错误描述符/来源及额外连接；recipe解析测试本身不覆盖这一步。真实GE导出、OMG编译与上板待上述重试。

## 自定义算子调用覆盖与GDN板端回传

2026-10-27，用户回传 `fused-board-ec7f1df7f0b1`：GDN prefill/decode均加载、执行成功，但数值明显不符。prefill核心输出relL2约0.9992、末态约0.8856；decode核心输出全零（参考非零）、末态relL2约0.9696。来源为用户粘贴报告，原始结果未在本地重新核验；该记录只证明报告中的调用完成，不能作为精度或NPU执行位置通过。

按用户要求先补齐全部五类自定义算子的调用覆盖，暂不深入优化GDN数值。新增CacheUpdate与AdnFusedInferAttention各prefill/decode，共9例；保留旧三类bindings和默认五例兼容。conv宽度4是显式调用候选，双输出不使用猜测的golden；RMS缺维推注册保持独立失败，不自动替换算子。

`run_fused_probes.py --all-ops` 复用原编译产物，缺项另建批次并继续执行编译成功项，最后汇总全部9例。两批清单与ABI哈希独立保留，同次runner构建可复用。新增Cache/Attention仅为独立调用探针，生产图接入、实际数值和完整模型仍待验证。

主机验证：算子构图/绑定/候选相关39项通过，HDC及批量编排11项通过，native CMake构建与CTest 5/5通过；真实CLI生成全部9例源码成功，Attention recipe另经主机C++解析通过。GE/OMG和设备执行使用替身或未调用，这些结果不替代黄区编译与板端回传。

## 融合GDN最新真实编译回传

2026-10-27，用户回传DDK `100.520.020.010`、目标 `kirinx90` 的两例真实编译结果：

| 用例 | 构图 / AIR导出 / OMC编译 | TBE适配 | 加载 / 执行 / 数值 | 执行位置 |
|---|---|---|---|---|
| `gdn_prefill` | PASSED / PASSED / PASSED | APPLIED 1次，拒绝0次 | NOT_RUN / NOT_RUN / NOT_RUN | UNVERIFIED |
| `gdn_decode` | PASSED / PASSED / PASSED | APPLIED 1次，拒绝0次 | NOT_RUN / NOT_RUN / NOT_RUN | UNVERIFIED |

来源为用户粘贴报告，本次未取得原始产物重新核对SHA256。该结果不覆盖完整模型、设备执行或数值验收；conv/RMS仍保留此前未通过的结论。下一步复用这两例编译产物，执行[当前更新首节的上板命令](current-status.md#用户现在执行什么)，不重复编译或扩展探针。

上板前发现runner仍硬编码FP32末态，已改为接受绑定中显式声明的FP16/FP32，保留FP16初态→FP32末态兼容；更新测试替身按对应字节数写出，避免FP16缓冲区越界。主机CMake构建及CTest 5/5通过，探针/回收比较23项通过，覆盖prefill和decode的FP16末态及三种状态dtype组合。无需重编OMC，只重建runner；未修改数值阈值，真实设备结果仍为NOT_RUN。

新增单命令入口 `run_fused_probes.py`，复用已编译OMC，自动构建runner、发现并上传私有库、查询运行版本/固件、执行、回收与比较。9项主机协议测试通过，覆盖零参数入口、自动库部署、失败回收、HDC假成功、传输损坏、路径越界及身份错配；构建和设备执行使用替身，主机Shell、SHA256与比较逻辑实际执行。板端验证仍待用户运行，数值验收阈值保持未定义。

## 融合探针编译排查历史

2026-10-27，用户回传DDK `100.520.020.010` 的5例结果：构图与AIR导出通过，OMC编译均失败，未上板。conv维推明确要求 `x2=[1,8192,4]`，与文档绑定的3不符；RMS当前注册表找不到 `AdnRmsNorm` 维推函数；GDN进入 `chunk_gated_delta_rule.py` 编译入口后子进程退出。GDN编译日志的末态是FP16，导出绑定为FP32，尚不能断定变化原因。`Ascend310B1` 内部日志及缺库提示也不能单独证明目标选错或崩溃根因。

本轮修复编译工作目录被删除的问题，新增只读 `inspect-fused-probes`，读取现有日志与匹配的安装接口文本；7项主机测试通过。没有修改文档绑定、生产缓存语义或共享DDK，也没有重新执行真实编译。下一步命令见[当前更新](current-status.md#用户现在执行什么)。

随后用户回传只读检查结果：两例GDN都在TBE `platform_info.py` 的 `te_update_version()` 抛出 `RuntimeError`，说明本次直接失败点是编译子进程的平台设置；尚不能判定参数、包版本或配置文件中的哪一项不匹配。路径位于Conda的 `site-packages` 不足以证明环境混用，`Ascend310B1` 与 `kirinx90` 的日志也不足以证明需要切换目标。安装目录确有GDN的Python编译入口及C++源码；仅凭通用 `DTYPE_MAP` 不能确认状态dtype支持范围。

针对首次检查遗漏调用帧的问题，检查器补充完整traceback、保存的OMG环境记录、报错指向的TBE平台函数源码、选定DDK配置目录及GDN状态读写片段。8项主机测试通过，覆盖调用帧保留、仅读源码、环境字段过滤及状态读写片段提取；独立代码审查通过。该次诊断修正只读源文件、不执行TBE，当时尚未验证编译修复。

第二次检查回传进一步确认：安装版GDN的 `isGlobal`、`lsGlobal` 均为 `GlobalTensor<half>`，分别以 `reinterpret_cast<__gm__ half *>` 绑定 `initial_state` 和 `last_recurrent_state`。这为该源码的FP16状态ABI提供依据，与文档的FP32末态不同；不推广到其它DDK。新增 `bindings-from-document --gdn-state-source`，校验两处直接half指针后生成FP16末态绑定，并用组合来源文件绑定原文档和安装源码SHA256；默认文档绑定不变。

TBE源码显示：包装函数调用 `cce.te_update_version("kirinx90", "AiCore", "", "false")` 后丢弃非success返回值，抛出通用异常。新增 `check-tbe-platform` 在独立Python进程中读取该底层返回值，使用与OMG相同的环境构造，记录调用前后平台信息、模块哈希及加载库。仅导入初始化，不能宣称复现OMG工作进程；无论成功失败，都不能单独判断目标支持或编译通过。该次工具改动时尚未取得黄区执行结果。

本轮40项主机定向测试通过；命令行测试另验证源码覆盖、两例GDN的FP16输出描述和参考字节数，以及模拟底层返回成功/失败、字符串对象的退出状态。使用模拟TBE验证子进程流程，未调用真实DDK或设备。

用户随后回传：安装源码绑定生成成功，GDN C++ SHA256为 `01634d23da746856c1119a9743d7f6db5f66aef9560d745423556a9c3162125a`。轻量平台调用返回字符串 `error`；调用前后SOC为 `Ascend310`、core为unknown，未复现原OMG工作进程的 `Ascend310B1` 初始化状态。进程同时加载本DDK TBE目录的 `libtvm.so` 和OMG目录的 `libplatform.so`。这些是定位线索，尚不能证明该DDK不支持目标或两库不兼容。

随后增加当前DDK内的库搜索顺序与已安装初始化接口对照。最多四个独立子进程，均使用 `kirinx90`；不更改正式OMG环境，不自动采用任何对照结果。各进程的平台初始化、实际加载库和原始返回值分别记录；报告收集完成不等于编译环境通过。

对照工具17项主机测试通过，覆盖子进程隔离、同目标初始化、库顺序、缺失/不匹配接口、初始化失败和全部失败时的状态。命令行与文档Shell语法验证通过；使用模拟TBE。

用户回传的四组结果均失败：两种库顺序下，仅导入时SOC为 `Ascend310`，调用 `te_set_version("kirinx90")` 后为 `Ascend310B1`，随后底层更新再次收到 `kirinx90` 均返回 `error`。这说明单改库顺序不足以解决该次失败；初始化与更新对平台名的处理差异是下一项假设。`Ascend310B1` 是该DDK的实测返回值，不能据此把OMG目标改为该名字；用户分析文档的 `Ascend310P3` 仍只描述原工程平台。

新增 `check-tbe-platform --diagnose`，把包来源（当前环境/当前DDK的TE wheel副本）、库顺序（OMG/TBE优先）、更新参数（请求名/初始化后读取的SOC名/空字符串）组成最多12组、4路并行。每组都先初始化 `kirinx90`，独立进程、独立输出和超时；包副本仅解压到运行目录，不安装或改写DDK。记录实际导入位置、映射库、更新前后SOC与原始返回值；成功仅代表该次API调用成功，需后续真实OMG验证。此轮不修改生产转换配置或算子计算。

批量诊断工具24项主机测试通过，覆盖并发上限、别名与空参数、崩溃后继续、wheel来源隔离及不产生错误候选。真实CLI用模拟DDK完成12组及逐例进度检查；命令互斥和文档Shell语法通过，独立代码审查通过。

随后用户回传12组真实API结果：传 `kirinx90` 的4组均返回 `error`，传初始化后查询到的 `Ascend310B1` 或空字符串的8组均返回 `success`，SOC均保持。当前环境与DDK wheel所列7个模块哈希一致；改变OMG/TBE库顺序没有改变结果。这把本次平台API失败定位到名称处理差异，不支持继续把已列模块版本或库顺序当作主要原因，也不证明OMG完整编译已修复。

新增显式 `probe-fused-ops --tbe-diagnosis`：核对该诊断与选定DDK的来源，在运行目录复制TE wheel，只适配 `platform_info.te_update_version` 的底层平台名称参数。只有请求为 `kirinx90` 且当前SOC与已验证的初始化结果一致时才转换，其余参数保持；记录真实工作进程调用。OMG平台参数仍为 `kirinx90`，DDK及已安装Python包保持不变。`--cases gdn_prefill,gdn_decode` 只重试两张真实尺寸GDN图，conv/RMS另行处理。

兼容适配的28项环境测试与40项探针/绑定测试通过；最终AST守卫的4项定向复测通过。实际CLI以模拟GE/OMG工具联通两例GDN，独立Python工作进程均加载运行目录副本并留下 `APPLIED` 记录；参数仍为 `--platform=kirinx90`。这验证主机接线与审计，不是实际DDK编译通过。文档命令及独立审查通过；后续真实OMG编译回传见本页首节。

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
