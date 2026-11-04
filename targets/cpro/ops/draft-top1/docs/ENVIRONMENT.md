# Environment

本次目标 Huawei X90/kirinx90；用户报告 CPro DDK 100.520.020.010、runtime 800.631.120.010。它们尚未为本算子做二进制哈希绑定。本地主机无所选DDK或设备，workspace `cpro-blue-pending` manifest也缺失，因此不声称厂商构建、注册或上板已通过。

用户明确要求复制ops源码后自行编译安装。准备脚本调用其所选环境的msopgen生成完整工程，只写新输出目录，固定kirinx90，记录工具及源码哈希；不运行build/install、不连接设备。用户再运行厂商build.sh，将源码发布和host注册安装到自己的DDK。此用户操作范围不等于允许代理修改工作区共享SDK。

官网提供kernel入口、OpDef/OP_ADD、shape/dtype推导、TilingData、平台UB查询、DataCopyPad和工程编译接口；已据此实现源码。所选内部DDK仍可能存在版本差异，须以实际编译结果收敛，不将文档支持直接标为设备通过。

`scripts/inspect_ddk.py --paste --output <existing-context>` 可选读取旧报告并输出有长度上限的纯文本，无需archive；若没有旧报告，加 `--ddk-root` 会只读采集。未发现接口仅为收集缺口，不是硬件不支持。正常下一步是编译本源码，无需再发送SDK包。

Torch模块不依赖torch_npu即可定义schema/Meta并做CPU参考。AIR导出显式加载所选torch_npu/torchair并注册converter。当前没有PrivateUse1直接执行桥；将来310P直接Torch执行还需自己的设备adapter。不能将CPro内置工具打印Ascend310B1等字符串当作310P兼容证据。
