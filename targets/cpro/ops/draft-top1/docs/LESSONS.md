# Lessons and evidence scope

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
