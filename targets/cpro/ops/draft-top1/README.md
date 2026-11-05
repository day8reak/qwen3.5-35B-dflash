# StableArgMax：通用 Top1

源码已准备，**所选 X90 DDK 编译、设备数值与性能仍待验证**。有限、连续 ND FP16 输入，静态 rank 1–8、正维度、总元素数 ≤ INT32_MAX；沿最后维归约，返回去掉最后维的 INT32 索引，同值取最小索引。Qwen `[1,15,248320] → [1,15]` 是验收配置。TopK、其他轴、FP32 输入及 310P 执行适配暂未实现。

| 位置 | 工作 | 环境 |
|---|---|---|
| DDK 服务器的 allops | 编译安装 host/kernel | 现有 CANN Kit / CPro DDK 工程 |
| DDK 服务器的模型仓库 | GE C++ → AIR → OMG/OMC、单算子上板 | Python + NumPy、GE C++ 头文件/libgraph、OMG；不需要 torch、torch_npu、TorchAir |
| 权重服务器 | Torch 调用、Meta、TorchAir 模型导出 | 现有 PyTorch / TorchAir 环境 |

## DDK 服务器：放入现有 allops

只复制这三个文件到现有 `allops` 的同名目录，文件名保持不变：

```text
op_host/stable_arg_max.cpp
op_host/stable_arg_max_tiling.h
op_kernel/stable_arg_max.cpp
```

可以手工复制，也可以使用复制工具；`ALLOPS_DIR` 设置为现有多算子工程目录：

```bash
OP_DIR="$REPO_DIR/targets/cpro/ops/draft-top1"
python3 "$OP_DIR/scripts/prepare_allops.py" \
  --allops-root "$ALLOPS_DIR" \
  --output "$AI_RUN_DIR/draft/tests/stable-arg-max-copy"
```

工具只复制上述三个文件并记录哈希，不生成新工程、不修改 CMake 或其他算子、不调用编译安装。相同文件可重复复制；有不同内容的同名文件时使用 `--replace`，旧内容保留在本次输出的 `previous/`。输出目录须为新的、位于 allops 外的目录。

然后在 allops 中执行你已有的 **kirinx90 编译安装命令**。若工程显式枚举源码，将三个文件加入其既有列表；不要用单算子工程覆盖整个 allops。本目录 `scripts/build.sh` 只是复制工具包装，不是厂商构建命令。原 `prepare_project.py` / `stable_arg_max.json` 仅供没有现成工程时可选使用，**本流程无需 msopgen**。

安装成功不等于 kernel 已通过 OMG 编译；接着执行下面的单算子检查。`tests/kernel_operator.h` 是主机测试 shim，不能复制到 allops 或加入 NPU include 路径。

## DDK 服务器：GE → AIR → OMC

在 DDK 服务器保留本模型仓库。`CANN_ROOT` 指向 GE 主机 C++ SDK（有 `include/graph/graph.h`、`lib64/libgraph.so`），`CPRO_DDK_ROOT` / `CPRO_DDK_ID` 指向刚安装算子的所选 DDK；这些不是 TorchAir 路径。

```bash
python3 "$OP_DIR/scripts/stable_arg_max_ge.py" \
  --ge-root "$CANN_ROOT" \
  --ddk-root "$CPRO_DDK_ROOT" --ddk-id "$CPRO_DDK_ID" \
  --output "$AI_RUN_DIR/draft/tests/stable-arg-max-ge"
```

这条命令生成 `probe.cpp`，编译 GE 主机程序，构造 **Data(x) → StableArgMax(x → indices) → 输出**，保存 `air/stable_arg_max.air`，再调用现有 OMG 工具链生成 OMC。它不加载 Torch/TorchAir，不使用内置 ArgMax 替换自定义节点，不编译整个 Draft。

默认只编译一份 Qwen shape 的 OMC，准备七组 NumPy 直接 argmax 参考输入，覆盖大索引、跨块并列、正负零、subnormal、相邻可表示数、随机和全部有限 binary16 模式，全部复用同一 OMC。报告明确记录 DDK 端未做 PyTorch 交叉计算；对应参考另在权重侧主机检查中与 PyTorch 对照。

`result.json` 记录阶段和错误，`build.log` / `export.log` 是 GE 日志，`omc/compile.log` 是 OMG 日志，`placement-inspection.json` 核对实际公开 IO 和 CPU/NPU 归属。仅有 NPU GraphOp、无 CPU/未知归属/主图遗留计算时允许上板；此检查不证明 NPU 内部指令或设备正确性。

其他选项：

- `--shape 17 8193`：输出尾写与输入 tile 尾部检查，四组输入共用一份 OMC，使用另一个新输出目录。
- `--export-only`：只做 GE C++ → AIR，无需 DDK/OMG。
- `--plan-only`：仅生成 C++ 和 NumPy 输入/参考，无需 GE SDK。
- `--cxx11-abi 1`：仅当所选 GE 主机库要求该 ABI 时使用；默认沿用现有 GE 探针的 ABI 0。

每份 AIR/OMC 是静态 shape。探针 IO 总量限 16 MiB，避免巨量测试数据；这不是 kernel 的 shape 上限。已有上板 runner 本入口限输入 rank 2–4；rank1 标量输出和 rank5–8 可生成 AIR，但需要后续 runner ABI 适配，不能据此宣称设备支持。

## DDK 服务器：复用现有上板工具

沿用此前探针的设备配置文件，将其路径放入 `VECTOR_CONFIG`。启动前复核 OMC、ABI、源码和 fixture 哈希：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --compiled "$AI_RUN_DIR/draft/tests/stable-arg-max-ge" \
  --ddk-root "$CPRO_DDK_ROOT" --config "$VECTOR_CONFIG" \
  --output "$AI_RUN_DIR/draft/tests/stable-arg-max-validation" --apply
```

默认跑该目录全部输入，每组预热 3 次、实测并保存 10 次输出，INT32 逐字节相等。API 墙钟时间不等于纯 NPU 时延。启动后仅回收同一次任务：

```bash
python3 "$REPO_DIR/targets/cpro/tools/validate_draft_vector.py" \
  --recover "$AI_RUN_DIR/draft/tests/stable-arg-max-validation" \
  --output "$AI_RUN_DIR/draft/tests/stable-arg-max-validation-result"
```

未完成时继续回收同一执行目录，不重复 `--apply` 或删除锁。以上命令交付给用户执行，本地未连接设备。

## 权重服务器：Torch / Meta / 模型导出

把本目录 `python/` 加入模块搜索路径，导出前注册 converter：

```python
import torch
import torchair
from stable_arg_max import register_ge
register_ge(torchair)
tokens = torch.ops.custom_ops.stable_arg_max.default(logits)
```

| 注册层 | 名称 |
|---|---|
| Torch schema / Meta | `custom_ops::stable_arg_max` |
| TorchAir converter / GE / host OP_ADD | `StableArgMax`，输入 `x`，输出 `indices` |
| kernel 入口与文件名 | `stable_arg_max` |

`scripts/export_probe.py --format air` 保留为**权重服务器的 TorchAir 导出入口**；`--format pt2` 检查真实 Torch export。DDK 服务器使用上面的 `stable_arg_max_ge.py`。名称、端口和数据类型保持一致，没有改变完整 Draft 或注册未经验证的 PrivateUse1/310P adapter。

DDK 流程的无框架主机检查：`python3 "$OP_DIR/tests/test_ddk_workflow.py" --output <新目录>`。`scripts/validate_source.py`、`validate_host.py`、`reference.py golden` 含 PyTorch 交叉检查，放在权重服务器运行。

[合同](docs/CONTRACT.md) · [官方来源](docs/OFFICIAL_SOURCES.md) · [环境](docs/ENVIRONMENT.md) · [验证](docs/VALIDATION.md) · [计划](docs/PLAN.md)
