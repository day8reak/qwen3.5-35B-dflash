# 自定义算子源码

[draft-top1](draft-top1/README.md) 提供通用 `StableArgMax` 的 AscendC kernel、host tiling/注册、Torch schema/Meta/GE converter 和独立验收工具。目录名保留任务来源；算子不绑定 Qwen，Qwen Draft 是其中一个验收尺寸。

整体复制 `draft-top1/` 到算子编译主机即可。由该主机所选 CANN Kit 的 `msopgen` 生成构建工程，再放入本目录代码，避免混用其他版本的 CMake/SDK 模板。无需回传打包文件。

当前交付是待所选 DDK 编译和设备验证的源码；X90 为本次目标，310P 接入与 TopK 均未宣称支持。
