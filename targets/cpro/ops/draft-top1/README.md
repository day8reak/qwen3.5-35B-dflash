# StableArgMax：通用 Top1

可整体复制本目录到算子编译主机。包含 AscendC kernel、host tiling/原型注册、Torch schema/Meta、TorchAir converter 和主机验收代码；无需回传打包文件。

当前是**源码交付，待所选 X90 DDK 编译与上板验证**。算法沿最后一维求 Top1：连续 ND FP16，静态 rank 1–8，所有维度正数，总元素数不超过 INT32_MAX；返回 INT32，shape 为输入去掉最后一维。同值取最小索引，输入须有限。Qwen `[1,15,248320] → [1,15]` 是验收用例，目录名保留任务来源。TopK、其他归约轴、FP32 输入和 310P 执行适配尚未实现。

| 层次 | 唯一名称 / 文件 |
|---|---|
| Torch | `torch.ops.custom_ops.stable_arg_max.default(x)` |
| GE / 原型 / OP_ADD | `StableArgMax`，输入 `x`，输出 `indices` |
| kernel 入口及文件名 | `stable_arg_max` / `op_kernel/stable_arg_max.cpp` |
| host 注册与 tiling | `op_host/stable_arg_max.cpp`、`stable_arg_max_tiling.h` |
| msopgen 原型 | `stable_arg_max.json` |
| Torch / Meta / GE converter | `python/stable_arg_max.py` |

## 在所选 DDK 生成和编译

先加载你现有 CANN Kit 的 AscendC 环境，确认 `msopgen` 来自所选 SDK。按[官方工程流程](https://developer.huawei.com/consumer/cn/doc/doccenter-capabilities/cannkit-operator-development)，用该 SDK 生成完整 CMake 工程，脚本再放入本目录的三个 C++/头文件。不会复制其他版本的厂商模板。

```bash
OP_DIR="$REPO_DIR/targets/cpro/ops/draft-top1"
TOP1_BUILD="$AI_RUN_DIR/draft/tests/stable-arg-max-build"
python3 "$OP_DIR/scripts/prepare_project.py" \
  --ddk-root "$CPRO_DDK_ROOT" --output "$TOP1_BUILD"
TOP1_PROJECT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["project"])' "$TOP1_BUILD/prepare-report.json")"
cd "$TOP1_PROJECT"
bash ./build.sh
```

准备脚本只生成新目录、核对名字并覆盖工程中的自有源码，固定 `kirinx90`，不运行安装。最后一行是你执行的 SDK 编译安装步骤，会将交付件装入当前 DDK。保留 `prepare-report.json` 和 `msopgen.log`；已有目录不覆盖。

官方源码发布模式下，`build.sh` 成功仅代表 host/注册及 kernel 源码发布，**不等于 kernel 编译成功**。可再执行生成工程的 `bash ./build_devices.sh` 检查 kernel；最终仍需下面的 AIR→OMG→OMC 编译以及上板数值验证。构建脚本的发布行为见[官方编译说明](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-operator-project-compilation)。

如果已有多算子编译工程，保留其中其他算子的源码和配置，将本目录 `op_host/` 的两个文件和 `op_kernel/` 的一个文件放到同名目录，并确认原型为 `StableArgMax`、端口 `x/indices`、compute unit 为 `kirinx90`；不要改名成旧的 `Qwen35DraftTop1`。

## Torch 调用和导出

在模型 Python 环境把本目录的 `python/` 加入模块搜索路径，并在导出前注册 converter：

```python
import torch
import torchair
from stable_arg_max import register_ge
register_ge(torchair)
# logits 必须已满足有限、连续 FP16 合同；这里不添加隐式 cast。
tokens = torch.ops.custom_ops.stable_arg_max.default(logits)
```

Schema + `register_fake` + 显式 GE converter 参考了原 Target 的 `linear_attention_convstate_update`；GDN、RMS、attention 的 converter 同样通过固定 GE 类型和端口接入。Torch namespace 与底层实现分离，未来 310P 可复用接口；当前未注册 `PrivateUse1` 直接设备执行，不会偷偷退回 CPU。CPU dispatch 仅作参考。Meta 不读 tensor 数值、不使用占位 shape。

独立导出一个节点，默认 Qwen 尺寸；`--shape` 可指定其他静态形状：

```bash
python3 "$OP_DIR/scripts/export_probe.py" --format air \
  --output "$AI_RUN_DIR/draft/tests/stable-arg-max-air"
```

此步骤使用所选 `torch_npu/torchair` 环境，只导出、不编译或执行模型。`--format pt2` 可在普通 PyTorch 环境检查 export 图。当前完整 Draft 没有接入此算子。

## 验证

```bash
python3 "$OP_DIR/scripts/validate_source.py" \
  --output "$AI_RUN_DIR/draft/tests/stable-arg-max-source-check"
python3 "$OP_DIR/scripts/reference.py" golden \
  --output "$AI_RUN_DIR/draft/tests/stable-arg-max-golden"
```

第一条编译**实际 kernel 循环的主机 API shim**并检查通用尺寸及真实 Torch export；不替代厂商编译或硬件模拟。`tests/kernel_operator.h` 仅用于主机检查，准备脚本不会将它复制到厂商工程，也不能加入 NPU 编译 include 路径。

第二条生成七组 Qwen 完整尺寸的 NumPy/PyTorch 交叉参考。设备用同一 OMC/输入先预热 3 次、再实测 10 次，每次输出保存为 `<outputs>/<case>/000.bin` 至 `009.bin`，再运行 `scripts/reference.py compare --fixtures <golden> --outputs <outputs> --report <new-report.json>`。仍须核对 NPU 归属、完整形状用例和全部10次结果；主机参考通过不证明 NPU 支持所有声明形状。

[合同](docs/CONTRACT.md) · [官方来源与注册对应](docs/OFFICIAL_SOURCES.md) · [环境](docs/ENVIRONMENT.md) · [验证状态](docs/VALIDATION.md) · [计划](docs/PLAN.md)
