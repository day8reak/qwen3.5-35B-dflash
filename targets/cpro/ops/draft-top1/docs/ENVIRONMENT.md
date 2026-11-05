# Environment

目标 Huawei X90/kirinx90。用户报告 CPro DDK 100.520.020.010、runtime 800.631.120.010；本算子尚未绑定实际安装后的二进制和设备结果。本地所选 cpro-blue-pending manifest 缺失，无所选 DDK/设备，不声称厂商构建或上板通过。

2026-11-05用户明确两台服务器的边界：

- **DDK服务器**已有多算子 allops。只放入 `op_host/stable_arg_max.cpp`、`op_host/stable_arg_max_tiling.h`、`op_kernel/stable_arg_max.cpp`，使用其已有构建安装流程。`prepare_allops.py` 只复制并记录自有文件，保留工程配置/其他算子；`--replace` 备份有差异的旧文件。不要求 msopgen 或新工程。
- 该服务器**没有TorchAir**。`stable_arg_max_ge.py` 使用 Python/NumPy 和 GE C++ API生成 AIR，复用已有 OMG 环境工具及 OMC ABI/placement 检查。`CANN_ROOT` 是 GE 主机头文件/库路径，`CPRO_DDK_ROOT` 是实际安装自定义算子的 DDK；二者不可从 TorchAir 路径推断。生成脚本不导入 torch、torch_npu 或 torchair。
- **权重服务器**有 TorchAir。保留通用 Torch schema/Meta 与显式 converter，完整模型或 TorchAir 单节点导出在该环境运行。直接 PrivateUse1 执行/310P adapter 仍未实现。

用户负责 allops 编译安装和板端命令，本次本地仅完成源码与工具检查；这不授权改动工作区共享 SDK。厂商 build.sh 成功可能仅发布源码/host 注册，kernel 编译仍须以实际 OMG 与板端结果为准。

`inspect_ddk.py --paste --output <已有报告目录>` 可输出有长度上限的纯文本，无需归档。未收集到接口是证据缺口，不是硬件不支持。所选内部DDK可能有接口差异，以实际编译诊断为准；不将内部打印的 Ascend310B1 字符串当作 310P 支持证据。
