# Environment

目标固定 Huawei X90/kirinx90。用户选定内部 CPro DDK，记录 ID 100.520.020.010、运行时800.631.120.010；这些是用户配置记录，尚未为此算子采集二进制哈希。工具路径由 CPRO_DDK_ROOT 提供。

本地主机无该DDK、无设备。workspace target profile cpro-blue-pending 未找到对应manifest。公开CANN/GE和Kirin9030不能代替本目标。TBE日志Ascend310B1字符串也不构成更换目标的理由。

scripts/inspect_ddk.py一次只读收集：实际AddCustom/自定义样例、注册配置、host/kernel构建文件、ReduceMax/calIndex/ReinterpretCast、DataCopyPad/尾写、platform查询和编译器身份。可选已有omg-environment.json必须属于同一个DDK；外部自定义源码需显式source-root。完整小源码带哈希保存在bundle，不只截几行接口。

收集不会导入SDK模块、运行shell脚本、编译、加载共享库或触碰设备。PARTIAL/未发现只表示收集缺口，不表示算子不支持。缺失libai_npucore_ascendc_kernel.so等路径不单独作为当前Bitcast失败根因。

后续接入应在run内建私有注册overlay，禁止写共享SDK。先确认包含路径、编译器参数、目标宏、kernel签名、host tiling/infer注册、FP16输入/INT32输出配置及OMG发现方式，再编译同一Top1验证包。当前build/on_board入口明确返回PENDING和非零退出码，防止模板误调用其他芯片builder。
