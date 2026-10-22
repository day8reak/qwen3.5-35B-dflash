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


## 2026-10-22：tiny 上板路径与停止规则

补丁 run：`20261022T025303Z-32164-0cf8ab`。新增 [tiny 完整操作指南](tiny-validation.md)，覆盖随机小模型的 CPU 参考、AIR、OMG、正式 HIAI 构建、传输、probe、benchmark 与回传 token/stop 比较，不需要完整权重或 tokenizer。

修正 tiny AIR manifest 的 EOS：原先 `[62]` 与 `demo` 的空 EOS 不一致，可能让板端提前停止；现在两端都使用空 EOS。新增回归从 CLI demo，经 AIR/OMC 测试替身和板端命令生成，强制预测 token 62，验证普通与 DFlash 都生成四个 token 并以 `length` 停止，空 EOS 参数可经 shell 引号完整传递。

本次 AIR 编排与 CLI 共 **18 项测试通过**，证据为 run 内 `pytest.xml`。重新执行真实 FP16 tiny CPU demo（S=8、prompt=1,2,3、生成4个 token），普通/DFlash token 与停止原因一致，`tiny-reference.json` 为 `PASS`。Python 环境仍为声明的主机环境；未安装 TorchAir/CANN。

以上没有真实 AIR/OMC、内部库链接或 X90 执行证据。`cpro-blue-pending` profile 仍未登记，目标环境激活失败；测试仅使用资源清单中已声明的模型解释器。当前 NVIDIA/CPU 服务器无昇腾卡的 AIR 导出依赖尚未验证，tiny 路径也不能绕过此缺口。


## 2026-10-22：服务器缺库诊断与内部 DDK 布局

补丁 run：`20261022T030203Z-32536-b061e3`。根据用户共享日志，服务器已安装 TorchNPU `2.11.0.post2`，导入失败原因是缺少 `libhccl.so`；内部 DDK 版本已确认 `100.520.020.010`。这是用户反馈，未登录该服务器或安装其 CANN。

CLI 与包初始化改为按需加载模型依赖；缺库时 `preflight` 能输出版本和完整异常原因，OMG 转换、命令生成和 HIAI 构建入口不再为哈希工具强制加载 Torch。AIR 检查显式导入 TorchNPU 后访问其 TorchAir，避免关闭自动加载后误报未找到捆绑的 TorchAir。内部 OMG 不再硬性要求公开插件目录，保留内部环境脚本提供的路径。芯片支持仍须由真实转换确认。

隔离源码副本的 Python 全套回归 **129 passed、2 skipped**，跳过的是未配置真实 DOPT 路径的可选集成测试；没有重用历史 DOPT 结果冒充本次执行。证据：run 内 `lazy-import-check/full-results.xml`、`full-tests.log`。新增测试用失败的 Torch 导入模拟 `libhccl.so` 错误，检查报告、帮助命令和内部无 `tools/platform` 的 OMG 编排。随后确定性资源夹具更新的 3 项 preflight 测试重新通过，最终 12 个 Python 文件与测试树哈希一致。真实主机 `preflight-host.json` 成功生成，并如实报告本机没有 TorchNPU/TorchAir。文档 38 个 shell 代码块和 7 个 Python heredoc 语法通过。

真实 AIR/OMC、内部 HIAI 链接和 X90 执行仍为 **PENDING**。公开 CANN 9.1.X 仅是该 TorchNPU 的版本对应关系，不能证明无昇腾卡导出或与内部 OMG 的兼容性。`cpro-blue-pending` profile 仍未登记，本次目标环境激活失败，验证使用已声明的模型解释器。
