# Lessons and evidence scope

- 2026-11-04，用户报告最后内置payload编译失败。tokens_bits的选定Bitcast注册是INT32→INT4，图中请求FP16入口；GetOutputFormats/OpPreBuild失败。两项树SKIPPED、NOT_RUN，不能当作两次独立编译失败。没有有效OMC或设备输出。不以日志裸dtype数字跨枚举解码，也不推断所有X90 Bitcast均不支持。
- 此结果满足用户“再试一次，不行就写算子”的条件授权。终止内置矩阵扩张，进入固定Top1；残差AddCustom继续独立暂停，完整Draft未变。
- 归约索引和图INT32节点是不同接口：公开X90 ReduceMax基础API不能证明当前CPro注册和OMG代码生成可用。先收集选定SDK的完整样例/注册/构建/API证据再写适配。
- 60字节输出的物理写回与跨核cacheline风险必须先设计。公开API资料仅作问题线索；所选SDK支持、实际buffer分配和设备安全仍需验证。
- 经验已进入本算子合同、只读收集器、host比较器；未修改共享技能，因为目前证据局限于用户选定内部DDK。未解决假设：具体kernel注册、归约index位格式、精确尾写、UB/core配置与性能。
