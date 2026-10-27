# AIR 环境排障

[环境配置](environment-setup.md) · [PyTorch 执行](pytorch-guide.md) · [GE 执行](ge-guide.md)

先确认失败发生在哪一阶段，再检查该阶段的依赖。现有可用环境直接复用；早期安装记录不作为当前安装指令。

| 现象 | 检查与处理 |
|---|---|
| CPU/CUDA 参考导入 Torch 时触发 TorchNPU 缺库 | 本项目 CLI 的参考命令会在进程内隔离可选 NPU 探测；使用 `demo` / `reference`，无需卸载 TorchNPU |
| 显式导入 `torch_npu` 报 `libhccl.so` 等缺失 | 使用匹配的服务器 CANN 及其完整原生组件，加载该安装环境脚本；Python wheel 安装成功不代表原生库齐备 |
| TorchAir 导入成功但导出失败 | 查本次导出日志、实际 converter 与设备初始化要求；不能仅凭 CPU dummy 输入断定导出不需要设备 |
| GE C++ 编译找不到头文件或 `libgraph.so` | `CANN_ROOT` 应指主机 GE 安装，核对 include/lib64、依赖与 C++ ABI；板端 `libgraph_host.so` 不能改名代替主机库 |
| AIR 成功，OMG 不存在或启动失败 | 检查目标 `CPRO_DDK_ROOT`、其环境脚本和 `tools/tools_omg/master/omg`；不要用服务器 CANN 路径替代目标 DDK |
| OMG 找不到外部权重 | 保留原 AIR 导出目录和全部引用文件，再核对清单；不要单独移动 AIR |
| 融合算子不支持或端口错误 | 核对实际安装的 op-proto、dtype、shape、属性类型及目标支持；文档生成的绑定 JSON 不提供 kernel |

可先执行以下只读检查，日志放在本次运行目录：

```bash
export AIR_CHECK_RUN="$(mktemp -d "$AI_RUN_DIR/air-check-XXXXXXXX")"
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli preflight \
  --output "$AIR_CHECK_RUN/preflight.json"
"$MODEL_PYTHON" -m pip check
```

`preflight` 区分自然 Torch/AIR 导入与隔离后的 CPU 参考导入。参考可用而 AIR 不可用是有效结果，不能用参考通过代替导出通过。导出使用新进程，避免沿用参考进程的后端隔离。

只有确实使用 PyTorch 导出时才检查 TorchNPU/TorchAir；直接 GE 的源码生成不依赖它们。具体检查命令统一在[环境配置](environment-setup.md)，当前已知结果统一在[当前更新](current-status.md)。
