# Validation

2026-11-05后续用户报告：三个allops源码SHA256与交付源码一致；GE流程已进入OMG，OMG退出码1，无OMC成功证据。用户未贴allops构建结果或compiler具体错误，尚不能定位注册/安装、接口或kernel问题。源码哈希一致不构成安装成功证据。

诊断工具修复run `20261105T010637Z-20281-c9f19f`：33项无Torch主机流程检查通过，见 `evidence/diagnostics-check/validation.json`。新增覆盖失败JSON携带真实读取的日志摘要、旧失败目录只读检查且不要求源码哈希、缺失/损坏日志报告保留、拒绝其他artifact、预检失败不覆盖旧结果。SDK调用仍为mock，用户失败没有在本地重跑，host/kernel不变。

2026-11-05两服务器流程更新，run `20261105T005058Z-19077-dfa957`：

| 新检查 | 结果 | 证据 |
|---|---|---|
| allops只复制3文件、旧内容备份、GE节点/形状、共享OMC、板端准备、失败阶段/哈希/CPU落点拒绝、禁止框架导入 | 27项通过 | evidence/ddk-workflow-final/validation.json |
| GE探针Qwen七组及通用shape参考与PyTorch独立比较 | 19组通过 | evidence/ge-fixture-crosscheck.json |

流程检查中的GE/OMG是mock、输出是主机合成数据，**未进行真实GE编译、AIR导出、OMG或设备执行**。无框架检查会主动拒绝torch、torch_npu、torchair导入，涵盖GE工具和已有board adapter。第10次输出单字节错误会判失败。host/kernel和Torch绑定未改变，既有主机证据继续保留。

当前默认是用户既有allops构建。DDK端用 `scripts/stable_arg_max_ge.py`，权重端才用 `scripts/export_probe.py` 的TorchAir路径；下文msopgen结果是可选旧入口的历史检查，不是本次构建前提。

本次run：`20261104T083807Z-10795-c00fd7`。当前仅HOST_PASSED，厂商编译/真实AIR/OMG/设备/性能均待验证。

| 检查 | 结果 | 证据（run相对路径） |
|---|---|---|
| 实际kernel控制流 + Torch schema/Meta/export | 63项通过 | evidence/source-check-2/source-validation.json |
| 原独立C++参考、NumPy/PyTorch、比较器及收集器 | 35项通过 | evidence/host-check-2/host-validation.json |
| 工程生成/覆盖保护/名称/目标配置 | 11项通过 | evidence/prepare-check/validation.json |
| 有限文本收集与离线复用 | 9项通过 | evidence/paste-check-final/validation.json |
| Qwen形状真实torch.export保存pt2 | EXPORTED | evidence/pt2-probe/export-report.json |

kernel检查将生产`op_kernel/stable_arg_max.cpp`直接编译到主机driver；只在测试include路径使用同步API shim，理想ReduceMax返回第一个最大值。它验证实际循环、tile offset、raw uint16索引读取、输入/UB/输出范围和输出canary。**不证明SDK兼容性、硬件ReduceMax、DMA或流水同步。**

覆盖rank1标量输出、rank8、单元素轴、奇数轴长、15/16/17/31/32/33行、127/128/129、485/512/513、8191/8192/8193、16385/65537/248321；以及Qwen七组完整尺寸，包括65535/65536大索引、跨块并列、正负零、subnormal、极值、相邻FP16、随机与全部有限binary16模式。真实Torch export在四个形状均只保留一个custom op，Meta输出为正确shape/int32。未注册PrivateUse1执行，不使用CPU fallback冒充NPU。

工程准备使用synthetic msopgen验证扁平/嵌套输出、示例默认compute unit修正为kirinx90、missing文件/错误入口/符号链接/缺少compute-unit拒绝、已有输出和DDK内部输出拒绝。准备脚本没有运行厂商build或安装。收集器纯文本测试允许直接粘贴，无需归档回传。

旧参考比较器仍固定七组Qwen配置，要求每组10份结果并核对source/fixture哈希、重算golden；70份合成文件不是70次设备执行。通用尺寸的目标设备覆盖还需使用同一算子源码分别编译静态shape，并对照直接argmax，不以Qwen单一形状通过代表全部形状通过。

首轮待做：所选DDK工程host/kernel编译；真实AIR里type=StableArgMax、输入x/输出indices；OMC计算归属全部NPU、无CPUCL或主图遗留计算；每例3次预热+10次真实执行，输出精确字节、canary及稳定性；最后再测延迟。最先验证Qwen60B输出及8193/17行尾部，再成组覆盖声明的静态尺寸。没有整Draft替换、真实logits有限性、310P或性能结论。
