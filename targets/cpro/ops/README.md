# 自定义算子源码

[draft-top1](draft-top1/README.md) 提供通用 `StableArgMax` 的 AscendC kernel、host tiling/注册、Torch schema/Meta/GE converter 和独立验收工具。目录名保留任务来源；算子不绑定 Qwen，Qwen Draft 是其中一个验收尺寸。

DDK服务器将 `draft-top1/op_host/` 的两个文件和 `op_kernel/` 的一个文件复制到现有 `allops` 同名目录，沿用现有工程编译安装。该服务器没有TorchAir：单算子测试用 `scripts/stable_arg_max_ge.py` 构造 GE C++ → AIR → OMG/OMC，再复用已有上板工具。

权重服务器保留 `python/stable_arg_max.py` 的 Torch schema/Meta/converter 和 `scripts/export_probe.py` 的 TorchAir 导出。无需回传打包文件；完整命令与分工见算子README。

当前交付是待所选 DDK 编译和设备验证的源码；X90 为本次目标，310P 接入与 TopK 均未宣称支持。
