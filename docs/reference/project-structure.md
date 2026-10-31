# 代码结构

当前主流程：文字输入 → 一次Prefill → 常驻Decode与Draft循环 → 文字输出。服务器负责准备、下发和回收，迭代在板端完成。

| 路径 | 职责 |
|---|---|
| `model/qwen35_moe_dflash/model.py`、`config.py`、`ops.py`、`target_features.py` | 模型和公共接口 |
| `model/qwen35_moe_dflash/execution/` | 运行配置、词表、部署、回收、目录管理 |
| `model/qwen35_moe_dflash/draft/incremental/` | 增量Draft导出、编译、ABI和KV适配 |
| `model/qwen35_moe_dflash/draft/` | Draft准备和参考检查 |
| `model/qwen35_moe_dflash/target/` | 原Target适配及状态映射 |
| `model/qwen35_moe_dflash/graph/` | GE图构建与降级 |
| `model/qwen35_moe_dflash/toolchain/` | AIR、OMG、OMC适配 |
| `model/qwen35_moe_dflash/common/` | 文件与资源公共工具 |
| `model/qwen35_moe_dflash/reference/`、`fused/` | 参考和融合实现 |
| `targets/cpro/scripts/` | 主流程命令入口 |
| `targets/cpro/tools/` | 按需板端诊断和profiling |
| `targets/cpro/runtime/` | 板端C++运行器 |
| `specs/` | 模型配置、接口约束和简短结果 |

本地测试和fake HIAI只在仓库外维护，不上传GitHub。旧的一次性探针和已替代命令删除，历史实现通过Git历史查看。运行时不依赖本地测试代码。
