# Lessons and evidence scope

- 2026-11-05用户第二轮日志越过平台初始化，在build_single_op的import阶段失败；静态检查确认注册指定py不存在，只有cpp。此前把日志中的File路径描述成“找到文件”不准确，已修正文档。分别核验host注册、实现文件发布、Python包解析、kernel编译；不要从其中一个阶段推定其他阶段成功。内部DDK的impl搜索根同时补齐，但不把它宣称为缺文件修复；使用当前allops构建脚本查明生成/发布环节，不手工拼厂商包装层。
- 新终端环境变量可能未设置；ALLOPS_DIR为空的find错误表示未检查工程，不是工程构建失败。通过静态预检在缺文件时停止OMG，复用现有AIR与已有故障日志，避免无效重编。

- 2026-11-05入口等待时没有输出：既有compile_existing_air支持progress和30秒compiler_heartbeat，但新入口未接线。沿用该观察器，stderr立即刷新阶段/日志路径并转发有界日志摘要，stdout保持机器可读结果；不另建进程控制/超时/重试实现。已运行旧入口用tail -F观察原日志，静默不构成卡死或编译通过证据。43项host检查验证接线，真实用户运行结果待回传。

- 2026-11-05进一步日志证明host注册及tiling已执行，当前阻塞在worker的te_update_version。对照既有toolchain/compile.py和历史Draft结果，发现新GE入口遗漏 `--tbe-diagnosis` 转发。复用已经证据绑定的CPro/TBE兼容处理；不改目标SOC、不改全局包、不重新扫描内置算子支持范围。入口测试现在验证参数从CLI到compiler，避免仅覆盖默认编译路径。
- 编译环境修复不需要重复GE导出。新增 `--reuse-export` 校验旧AIR、recipe、host/kernel/reference哈希与fixtures，允许仅诊断CLI代码变化；新输出保留旧失败现场，41项主机检查覆盖。实测兼容是否APPLIED与OMG是否成功分别记录，不能把修复接线测试当目标编译成功。

- 2026-11-05用户已走通GE入口到OMG，但旧入口只向终端抛CalledProcessError，隐藏了现有compiler日志。修复为输出原错误、失败阶段与有界日志摘要，并提供 `--inspect-compilation` 只读旧目录；不要求重导AIR/重编译来收集证据。33项主机检查覆盖此行为。OMG退出码1不是根因，allops源码一致不是构建安装成功。没有据此改kernel或认定DDK不支持。

- 2026-11-05，用户确认DDK服务器已有allops且没有TorchAir，权重服务器才有TorchAir。默认工作流改为向既有工程复制三个host/kernel文件，使用GE C++构造单节点AIR测试；Torch schema/Meta/converter留在权重侧。不能把框架导出依赖当成算子编译前提，也不重建多算子工程。
- 复用模型已有GE emitter、OMG隔离环境、OMC ABI/归属检查、上板启动/回收协议。每个shape只编译一份OMC，不因七组输入重复编译。DDK入口在禁用所有Torch系列导入的主机测试中通过；实际SDK和设备仍待用户执行。源码准备成功、GE主机程序成功、OMG成功、上板数值通过是独立阶段。
- 本次经验已落入ENVIRONMENT/README及test_ddk_workflow.py；未提升为共享平台能力规则，因为allops部署和服务器分工是本项目配置。未解决假设仍是所选DDK兼容性、kernel真实执行及性能，没有新设备支持结论。

- 2026-11-04，用户报告最后内置payload编译失败。tokens_bits的选定Bitcast注册是INT32→INT4，图中请求FP16入口；GetOutputFormats/OpPreBuild失败。两项树SKIPPED、NOT_RUN，不能当作两次独立编译失败。没有有效OMC或设备输出。不以日志裸dtype数字跨枚举解码，也不推断所有X90 Bitcast均不支持。
- 此结果满足用户“再试一次，不行就写算子”的条件授权。终止内置矩阵扩张，进入固定Top1；残差AddCustom继续独立暂停，完整Draft未变。
- 归约索引和图INT32节点是不同接口：公开X90 ReduceMax基础API不能证明当前CPro注册和OMG代码生成可用。先收集选定SDK的完整样例/注册/构建/API证据再写适配。
- 60字节输出的物理写回与跨核cacheline风险必须先设计。公开API资料仅作问题线索；所选SDK支持、实际buffer分配和设备安全仍需验证。
- 经验已进入本算子合同、只读收集器、host比较器；未修改共享技能，因为目前证据局限于用户选定内部DDK。未解决假设：具体kernel注册、归约index位格式、精确尾写、UB/core配置与性能。

- 2026-11-04更新：用户不能回传归档，并明确要求先交付ops源码。官网CANN Kit已提供msopgen、host/kernel注册、X90 BlockDim=1、DataCopyPad有效字节搬运等接口；旧“必须先收到SDK包”的下一步已撤销。源码按公开契约准备，所选DDK和设备验证仍不推定通过。
- 阅读原Target的convstate、RMS、attention、GDN注册后，采用通用Torch schema/Meta与显式GE converter分层；新算子不依赖已有torch_npu符号。PrivateUse1直接设备执行另行适配，避免误称310P自动支持。
- 用户确认更多shape的Top1优先，命名收敛到StableArgMax；TopK不作为伪参数占位。真实torch.export用于检查不展开成旧ArgMax图，原型/端口/kernel入口有对应校验。
- 厂商源码发布build.sh成功不等于kernel编译通过。准备脚本使用所选msopgen生成完整模板，记录身份并只覆盖自有源码；测试shim不进入厂商include目录。
- 共享技能经验审查：本次新增是公开CANN Kit与项目接口事实，保存在OFFICIAL_SOURCES/CONTRACT；未将未实测的具体kernel或310P能力提升为共享事实。
