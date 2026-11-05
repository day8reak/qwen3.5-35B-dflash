# Environment

| 执行位置 | 依赖与职责 |
|---|---|
| DDK服务器 | 现有allops构建安装；Python/NumPy＋GE C++生成AIR；OMG编译；HDC上板与回收 |
| 权重服务器 | PyTorch/TorchAir；Torch schema、Meta、GE converter及模型导出 |

用户回传环境：Huawei X90 / kirinx90，CPro DDK `100.520.020.010`。2026-11-05安装预检及OMG已通过；当前证据是用户报告，本地未连接服务器或设备。完整工具链归档及设备身份仍待后续验收补齐。

所选allops使用 `ASCEND_COMPUTE_UNIT=ascend310b` 生成INI和包装PY，因此host同时AddConfig(ascend310b/kirinx90)；实际OMG目标仍是kirinx90，不表示其他芯片支持。已安装 `custom.stable_arg_max` 的文件哈希见[验证](VALIDATION.md)。

内部DDK的 `ddk/ascendc/ops/impl` 加入编译子进程PYTHONPATH。TBE兼容通过 `--tbe-diagnosis` 引用已有诊断，仅作用于本次编译目录；本轮用户报告 `APPLIED`。不修改全局Python或共享SDK。

执行命令统一见[构建说明](../README.md)和[当前上板页](../../../../../docs/current-status.md)。用户负责allops安装及设备执行；host测试shim不得加入厂商include路径。
