# 验证记录：恢复工作后的 FP16 / DOPT / HIAI 版本

完整主机验证基线 run：`20261020T092412Z-4528-622cba`，工作区相对路径 `.work/qwen3.5-35B-moe/20261020T092412Z-4528-622cba`。代码在 run 中验证后保存到模型目录并提交独立仓库；原始日志/工具/权重不进入仓库。

Python 3.10.12 / Torch 2.12.1+cpu / Transformers 5.13.0 / safetensors 0.8.0。用户 Torch 2.11.0+cu130 尚未执行本轮测试。使用官方哈希校验过的蓝区 next-6.1.1.0 tools 与 kirinx90 插件；HIAI 头文件编译使用已声明 OHOS Native SDK 的通用 ARM64 编译器，没有借用 Kirin9030 插件或设备结果。

| 检查 | 结果及范围 | 证据（相对当前 run） |
|---|---|---|
| 最终 Python 全测试 | **118 passed**，22 个 legacy ONNX tracer/导出提示；包含真实 DOPT 可选测试，未跳过 | `log/python-final.log`、`out/python-final.xml` |
| FP16 图内部检查 | 普通区域无残留 FP32/BF16；tiny 普通/DFlash token/stop 一致 | `tests/test_precision.py`、`out/fp16-demo.json` |
| 实际 DOPT 校准 | 缩小的真实 upstream MoE/GDN，所有32个专家矩阵覆盖；校准保存、校验后恢复 | `tests/test_quantization.py`、最终 pytest 日志/XML |
| 量化数值参考 | 使用真实 vendor Cython 算子的主机桥接；FP16普通区域；普通/DFlash一致 | 同上；并非原生NPU执行 |
| 激活量化溢出修复 | 首次集成失败保留；FP32量化缩放后的饱和值与独立高精度oracle一致，重新生成通过 | `log/python-tests.log`、`log/dopt-frozen-compare-v2.log`、最终 pytest |
| 完整35B架构规划 | meta tensor统计，无权重下载；10,240专家、20,480专家矩阵；21.19GiB权重估算 | `out/quant-plan-delivery.json` |
| AIR/OMG 编排 | 输入/外部权重哈希、短路径、失败清理、拒绝旧ABI/密集fake-quant图 | `tests/test_air_pipeline.py`；测试payload是mock |
| HIAI C++ scheduler / 生命周期与故障 | **2/2 CTest**，FP16缓冲首末元素传递；fake HIAI，不是板端 | `log/ctest-host.log` |
| 真实公开HIAI头文件 / OHOS ARM64 | **HEADERS_COMPILED**，未链接内部库 | `out/hiai-headers/result.json`、`out/hiai-headers/build.log` |
| 当前导出环境 | DOPT校准可用；TorchAir/torch_npu不存在；NpuQLinear未定义 | `out/preflight.json` |

首次测试中的 DOPT NaN 是真实失败，不能当作通过。修复把量化器内部缩放改为 FP32，输出恢复 FP16，并更新候选身份；其余浮点计算仍执行 FP16。该改动不代表全权重或硬件已通过精度门槛。

上一次 run 的真实 OMG 短路径负例已检查：进入 `ParseOriginMSOrAir2IrGraph` 后因无效 AIR 失败，说明绕过了原先输出路径拒绝；这不是转换成功证据。原日志位于 `20261020T062757Z-31423-15c0e2/log/omg-air-short-path-smoke.log`。

## 未通过的完整模型门槛

完整权重校准/质量、35B参考、真实 AIR/OMC、内部HIAI链接、X90加载/执行、峰值内存、十轮板端精度/性能及设备发布均 **PENDING**。板机由用户访问，不能用CPU/fake结果代替。`cpro-blue-pending` profile未登记，环境激活不算成功；只使用已声明模型解释器和既有run内蓝区工具。

当前18G free低于候选权重估算，仍需可用内存与实际峰值确认。量化普通/DFlash严格一致只验证调度；量化相对BF16的任务质量尚未冻结独立门槛。`hardware_reference_policy=use`，没有X90能力画像/工作量账本/依赖延迟，不宣称理论余量或加速比。

下一步在用户导出机器运行 README 中的 `preflight`。需要匹配本 DDK 的 TorchAir/CANN/torch_npu，尤其是量化扩展；先做 tiny AIR→OMC，再完整量化和内部HIAI构建。权重/库无需传出，保留失败日志和板端probe/报告即可继续定位。

## 2026-10-21：复用已有 AscendC 环境

补丁 run：`20261021T061137Z-18631-50cb54`。服务器指南省略已由 `set_ascendc_env.sh` 完成的环境变量配置，项目路径保留已有 `PYTHONPATH`，OHOS/HIAI 资源延后到交叉编译时填写。OMG 子进程优先使用本次 DDK，再保留继承的库与 Python 路径。

AIR 编排与 CLI 共 **17 项测试通过**；新增回归用假 OMG 子进程实际导入继承路径中的扩展，检查所选 X90 插件优先和 simulator 库路径保留，父进程环境不变。README 与操作指南的 shell 代码块均通过语法检查。证据及修改文件哈希见 `specs/environment-verification.json`；原 `specs/verification.json` 保留上一完整主机验证基线，未将其 118 项结果冒充本次重跑。

用户提供的日志仅记录内部 AscendC 环境脚本成功；未访问或运行内部 DDK，也未完成真实 AIR/OMC、HIAI 链接或板端执行。
