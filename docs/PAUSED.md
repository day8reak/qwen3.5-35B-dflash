> **已恢复：用户明确说“继续吧”。当前精度为权重/激活可量化，其余 FP16。下文是历史暂停记录，最新状态见 README 和 validation。**

# 用户暂停检查点（2026-10-20）

历史指令：先暂停。此暂停已被后续“继续吧”解除。

最新部署要求是 **量化后的 Qwen3.5-35B-A3B DFlash / CPro X90**，链路仍为 PyTorch → AIR → OMG → OMC → HarmonyOS HIAI。BF16 仅作为参考。用户正在确认板机可用内存以及内部支持的 W4A16/W8A8 等格式，未确定具体位宽/分组/算法/质量门槛；量化尚未实现或执行。

本地源码已保存。此前已实现 BF16 参考模型、DFlash greedy 调度、HIAI C++ 适配、AIR/OMG 编排、构建与验证 CLI。主机完整测试 101 项通过；随后新增 6 项 CLI 测试，与 5 项 AIR 测试一起通过。C++ 2 组 CTest 通过，公开 HIAI 头文件的 OHOS ARM64 编译通过。没有真实 35B AIR/OMC、内部 HIAI 链接或板端结果。

最后一个修改是 `model/qwen35_moe_dflash/compile.py`：真实 OMG 拒绝长输出路径，因此改为隔离目录内的 `input.air` 和短输出前缀，再移动产物。**修改后的编排测试尚未重跑**，不得沿用修改前的 PASS。真实负例启动日志已保存，恢复后先检查其错误阶段。

蓝区 DDK 发现 `tools/tools_dopt/dopt_pytorch_py3/dopt/dopt_lm/opt_main.py` 暴露 w_bits（默认 4）、act_bits（默认 16）、group_size（默认 128）与 stage1/2/3。只能说明存在此工具入口，不能据此宣称 Qwen3.5 MoE、X90、内部 DDK 或 AIR 路线已兼容。尚未执行量化，下一步先确认支持与容量，再锁定量化方案及精度对照。

资源限制：权重在用户不可访问的板机；内部 HIAI 库不能传出；双方尚未有可用 TorchAir/torch_npu/CANN 导出环境。用户现有 Torch 是 2.11.0+cu130，本机已验证环境是 2.12.1+cpu / Transformers 5.13.0。

原始工作 run：`.work/qwen3.5-35B-moe/20261020T062757Z-31423-15c0e2`。源码工作副本在该 run 的 `out/source`，工具/日志/构建均在 run 内。目标目录的源码是暂停时副本。Git 尚未初始化、提交或推送；远端 `git@github.com:day8reak/qwen3.5-35B-dflash.git` 已确认可访问且为空。恢复后处理源码差异与提交，不操作其他模型的已有修改。

恢复入口：先 `kit/bin/ws context qwen3.5-35B-moe --task model --target cpro`，开启新的 run；阅读本文件、`specs/model-workflow.json`、`docs/target-plan.md`。重新核对量化需求后再继续。旧 README 的 BF16 导出命令属于参考基线，不能直接作为最终部署方案。

## 暂停期间补充的内存信息

用户补充：RAM 约 32 GB，已用约 13 G，free 约 18 G；Hugepagesize=2048 KB，Hugetlb=4096 KB。按 Linux meminfo 语义，默认大页为 2 MiB，大页池总计 4 MiB；不能据此推导 NPU 可分配内存。18 G 的单位与 free/available 字段来源尚待确认，MemAvailable 尚未提供。结构化记录见 `specs/device-memory-report.json`。

容量初算：按名义 35B 参数全部打包为 4 bit，单权重已约 17.5 GB（16.3 GiB），未计 scales、保留高精度参数、草稿、跨 OMC 重复 embedding/head 和工作区。因此不能把“18 G free”视为 W4A16 双模型常驻可行的证据。量化格式仍待内部支持确认。

这次仅保存资源信息，用户暂停仍有效；尚未开始量化实现、重跑编译或 Git 发布。
