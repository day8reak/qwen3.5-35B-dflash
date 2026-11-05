# Validation

2026-11-05用户回传：**OMG编译通过、图层面NPU-only；设备数值和性能未运行。** 配置为有限FP16 `[1,15,248320]` → INT32 `[1,15]`，DDK 100.520.020.010，目标kirinx90。

| 门禁 | 结果 |
|---|---|
| allops选定INI、包装安装 | StableArgMax条目存在；STATIC_PATH_RESOLVED |
| TBE兼容 | APPLIED |
| OMC与公开ABI | COMPILED；FP16输入、INT32输出与预期shape一致 |
| 解码归属 | NPUCL=1；CPU操作/主图遗留计算均为空 |
| 设备正确性/稳定性 | NOT_RUN；七组fixture，每组3预热＋10次逐字节比较 |
| 其他shape、独立构建、整模型、性能 | PENDING |

报告目录：`draft/tests/stable-arg-max-ge-config`。来源为用户粘贴，未独立下载产物。当前身份：

| 文件 | SHA256 |
|---|---|
| OMC | `2e76eda684ae0a4ac201af1c1ba511ca6b84e864a698193cfd1cfdfc1de2598f` |
| AIR | `9f51c2d6803e75b53baa612947f63f60e72b50d424d170fd57461c06230ee805` |
| 安装包装PY | `52a313c85e26ee47b7fbcde6e63259aa86f8983552df5747d6ee31492820d152` |
| host CPP | `a92876b59d72c2d430987d1b242d2f36fd3e77b601b9d955baacb1872ec91796` |
| kernel CPP | `f81371611cbb2aa85ed41b3a2ca6cd4b498cf4953ed437459f6547daa3da723e` |

保留的主机检查：kernel控制流/Torch export63项、独立参考35项、准备/文本交付20项、无Torch的DDK流程55项、NumPy/PyTorch参考交叉19组。主机API shim和mock SDK不能证明设备行为。报告位置由operator.json记录；最新用户观察摘要在run `20261105T015409Z-22517-5473bc` 的 `evidence/user-compilation-summary.json`。

首轮验证覆盖大索引、跨块并列、零与极值、subnormal、相邻FP16、随机及全部有限binary16。正确性通过后验证17行/8193列等尾部及其他声明shape，再考虑整模型接入。性能需要同功能全词表基线配对，不用旧局部CPU归约或混合图耗时推断收益。下一条命令见[当前上板页](../../../../../docs/current-status.md)。

本轮仅更新文档和状态：10个Shell代码块、42处本地链接及当前CLI参数检查通过；五份产物绑定源码哈希与用户编译报告一致。见同run的 `evidence/documentation-check.json`，未重新运行模型。
