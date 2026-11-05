# Environment

目标 Huawei X90/kirinx90。用户报告 CPro DDK 100.520.020.010、runtime 800.631.120.010；本算子尚未绑定实际安装后的二进制和设备结果。本地所选 cpro-blue-pending manifest 缺失，无所选 DDK/设备，不声称厂商构建或上板通过。

2026-11-05用户明确两台服务器的边界：

- **DDK服务器**已有多算子 allops。只放入 `op_host/stable_arg_max.cpp`、`op_host/stable_arg_max_tiling.h`、`op_kernel/stable_arg_max.cpp`，使用其已有构建安装流程。`prepare_allops.py` 只复制并记录自有文件，保留工程配置/其他算子；`--replace` 备份有差异的旧文件。不要求 msopgen 或新工程。
- 该服务器**没有TorchAir**。`stable_arg_max_ge.py` 使用 Python/NumPy 和 GE C++ API生成 AIR，复用已有 OMG 环境工具及 OMC ABI/placement 检查。`CANN_ROOT` 是 GE 主机头文件/库路径，`CPRO_DDK_ROOT` 是实际安装自定义算子的 DDK；二者不可从 TorchAir 路径推断。生成脚本不导入 torch、torch_npu 或 torchair。
- **权重服务器**有 TorchAir。保留通用 Torch schema/Meta 与显式 converter，完整模型或 TorchAir 单节点导出在该环境运行。直接 PrivateUse1 执行/310P adapter 仍未实现。

用户负责 allops 编译安装和板端命令，本次本地仅完成源码与工具检查；这不授权改动工作区共享 SDK。厂商 build.sh 成功可能仅发布源码/host 注册，kernel 编译仍须以实际 OMG 与板端结果为准。

用户首轮OMG日志已识别StableArgMax、记录custom/stable_arg_max.py实现路径并产出正确host tiling，worker在平台 `te_update_version` 初始化时退出。路径出现在日志不证明文件存在。仓库此前在同一选定DDK已用 `common/tbe-diagnosis` 验证并应用运行目录TBE兼容；新入口遗漏参数已修复。`--tbe-diagnosis` 复用该证据校验，不能无依据硬编码Ascend310B1或修改全局site-packages。旧AIR可复用，参数修复本身不改变host/kernel。

下一轮worker已进入build_single_op，但import custom.stable_arg_max失败。用户只读检查确认指定py不存在，DDK内只有同名cpp。omg_environment遗漏的内部ddk/ascendc/ops/impl目录已为编译子进程补入；文件缺失不能靠路径修复解决。GE入口的静态安装预检不导入厂商代码，缺失时停止在OMG之前。

最新allops证据已定位配置遗漏：preset选ascend310b，add_ops_impl_target读取autogen/aic-ascend310b-ops-info.ini，而该INI无StableArgMax；旧host只AddConfig(kirinx90)。host库的源码glob和kernel cpp的安装glob独立存在，因此能有host tiling和cpp，却没有生成的py。现host增加ascend310b并保留kirinx90；OMG仍用kirinx90，不修改allops preset、不生成替代py、不推断310B/310P可用。用户重建安装后须核对选定INI及STATIC_PATH_RESOLVED，再复用旧AIR编译一次。build.sh安装退出码变量混用、opbuild失败仅打印信息，末尾成功字样不能作为安装证据。

`inspect_ddk.py --paste --output <已有报告目录>` 可输出有长度上限的纯文本，无需归档。未收集到接口是证据缺口，不是硬件不支持。所选内部DDK可能有接口差异，以实际编译诊断为准；不将内部打印的 Ascend310B1 字符串当作 310P 支持证据。
