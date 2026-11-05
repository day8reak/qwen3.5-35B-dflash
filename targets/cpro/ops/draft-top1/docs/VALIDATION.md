# Validation

2026-11-05用户回传：**Qwen配置OMG及图层面NPU-only检查通过；七组设备数值与重复稳定性全部通过。** 配置为有限FP16 `[1,15,248320]` → INT32 `[1,15]`，DDK 100.520.020.010，目标kirinx90。

| 门禁 | 结果 |
|---|---|
| allops选定INI、包装安装 | StableArgMax条目存在；STATIC_PATH_RESOLVED |
| TBE兼容 | APPLIED |
| OMC与公开ABI | COMPILED；FP16输入、INT32输出与预期shape一致 |
| 解码归属 | NPUCL=1；CPU操作/主图遗留计算均为空 |
| 设备正确性/稳定性 | PASSED；七组、70次输出、1050个INT32精确一致；3预热＋10次/组 |
| API测量 | 总均值约1.232ms；范围0.922–1.695ms；非kernel-only |
| 补充17×8193尾部shape | PASSED；4组／40次精确稳定；单NPU子图 |
| 其他未测shape、独立构建、整模型、性能优化 | PENDING |

报告目录：`draft/tests/stable-arg-max-ge-config`。来源为用户粘贴，未独立下载产物。下表编译身份取自此前用户报告；回收摘要未重复列出模型digest，按同一执行目录关联，未独立下载复核。当前身份：

| 文件 | SHA256 |
|---|---|
| OMC | `2e76eda684ae0a4ac201af1c1ba511ca6b84e864a698193cfd1cfdfc1de2598f` |
| AIR | `9f51c2d6803e75b53baa612947f63f60e72b50d424d170fd57461c06230ee805` |
| 安装包装PY | `52a313c85e26ee47b7fbcde6e63259aa86f8983552df5747d6ee31492820d152` |
| host CPP | `a92876b59d72c2d430987d1b242d2f36fd3e77b601b9d955baacb1872ec91796` |
| kernel CPP | `f81371611cbb2aa85ed41b3a2ca6cd4b498cf4953ed437459f6547daa3da723e` |

保留的主机检查：kernel控制流/Torch export63项、独立参考35项、准备/文本交付20项、无Torch的DDK流程55项、NumPy/PyTorch参考交叉19组。主机API shim和mock SDK不能证明设备行为。报告位置由operator.json记录；编译观察摘要在run `20261105T015409Z-22517-5473bc` 的 `evidence/user-compilation-summary.json`。

本轮执行 `qwen35_vector_442c4081a48343b58e12a647f18e204c` 已完成回收，native/archive及全部case退出码为0，锁已释放。结果在 `draft/tests/stable-arg-max-validation-result`；用户摘要保留于run `20261105T020607Z-23101-092df2` 的 `evidence/user-device-summary.json`。均值/极值见[结果页](../../../../../docs/results.md)，未从摘要推算P99或完整Draft加速。

Qwen组覆盖大索引、跨块并列、零与极值、subnormal、相邻FP16、随机及全部有限binary16输入组。补充 `[17,8193]` 的first_last/equal/tile_ties/subnormal四组已上板通过，40次／680个索引精确稳定，覆盖16＋1行输出、8192＋1尾块及未对齐行起点；同步API等权均值0.51495ms，范围0.434–0.886ms。

尾部请求 `qwen35_vector_840699127bf2427eb5552d6eecf3b49a` 全部退出码0，已回收且锁释放。结果在 `draft/tests/stable-arg-max-tail/result`；用户摘要保留于run `20261105T021938Z-23422-d80bfd` 的 `evidence/user-tail-summary.json`。摘要未列shape和OMC digest，shape按已给编译命令关联，未独立取得二进制。

两轮共11组／110次通过，单算子探针已收敛。已增加可选完整Draft Torch导出接线；缩小六层模型的14路输出与原路径逐位相等，真实PT2导出保留一个custom op，主机检查共30项（21项模型／PT2／边界，9项导出工具／安装预检／CLI），见同run的 `evidence/integration-check.json` 与 `evidence/integration-tool-check.json`。实际TorchAir整图导出、OMG及设备验证仍待执行，命令见[当前阶段](../../../../../docs/current-status.md)。

kernel、host和单算子绑定保持设备验证版本。本轮不宣称全部rank/shape、NaN/Inf、310P或完整Draft已通过。
