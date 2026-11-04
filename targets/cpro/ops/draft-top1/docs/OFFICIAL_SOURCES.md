# 接口来源与名称对应

2026-11-04核对以下官方页面；源码是依照接口实现的候选，仍待所选DDK编译。页面可能随版本更新，本次下载摘要在工作流run的`evidence/web/`保留。

- [CANN Kit开发入口](https://developer.huawei.com/consumer/cn/doc/doccenter-capabilities/cannkit-operator-development)：op_host/op_kernel工程、kernel入口和OpDef/OP_ADD注册。
- [msopgen](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-creating-operator-project-msopgen)：原型JSON与`ai_core-kirinx90`目标。
- [Host tiling](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-tiling-implementation-on-the-host)：X90 BlockDim仅1，tiling序列化、workspace。
- [平台信息](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-platformascendc-introduction)与[UB容量](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-getcorememsize)：host容量检查。
- [DataCopyPad](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-datacopypad)：X90 half/int32有效字节搬运、输入尾部和输出非32B倍数。
- [ReduceMax](https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/reduction_compute/ReduceMax.html)：联合值/索引归约、同值较小索引、half索引uint16位存储、独立scratch。8192块长度小于局部索引范围上限。
- [标量访问与同步](https://asc.gitcode.com/guide/technical_appendix/concepts_and_terms/memory_access/scalar_read_write.html)：GetValue/SetValue与Vector/MTE流水的依赖。
- [编译发布](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-operator-project-compilation)：源码发布不代表kernel编译通过，厂商build.sh与build_devices.sh作用不同。

本地已阅读原Target的`export_glm_model_qwen3_5_moe.py`：convstate定义/schema、register_fake与converter（185–218行），以及RMS、attention、GDN的Meta和GE映射。复用注册分层方式，不复制这些算子的业务实现。已有torch_npu算子的npu命名空间依赖各自扩展库，不等于任意新算子只改Python注册就能直接在310P执行。

当前命名链：`custom_ops::stable_arg_max`（Torch）→ `StableArgMax`（GE、JSON op、C++ class/OP_ADD、TilingData注册）→ `stable_arg_max`（msopgen snake-case文件、kernel入口）。输入名称各层均`x`，输出均`indices`。测试覆盖真实torch.export保留自定义节点和Meta输出dtype/shape；GE调用参数用记录stub检查，真实AIR仍待用户导出。
