# AIR 导出环境排障

截至 2026-10-22，用户服务器已安装 `torch-npu==2.11.0.post2`，实际失败点是导入时找不到 `libhccl.so`。这发生在模型导出之前；不是 tiny 模型或 OMG 转换报错。现有日志没有真实 AIR、OMC 或板端执行成功证据。

## 先区分三套环境

| 环节 | 所需环境 | 当前已知信息 |
|---|---|---|
| 服务器 PyTorch → AIR | 配套的 PyTorch、TorchNPU、TorchAir 和服务器 CANN 运行库 | wheel 已装；`libhccl.so` 未能加载 |
| 服务器 AIR → OMC | 内部 DDK 的 OMG 及 X90 编译环境 | 内部 DDK 版本为 `100.520.020.010`，有 OMG |
| 板端 OMC 推理 | HarmonyOS ARM64 HIAI 库、固件和正式 runner | 内部库不能传出，由用户本地构建、执行 |

公开版本配套表把 **`torch_npu 2.11.0.post2` 对应到 TorchNPU 26.1.1、PyTorch 2.11.0、CANN 9.1.X**。这不等于内部 DDK 也应升级到该版本，更不能证明其 OMG 能接受该 TorchAir 导出的所有算子。[官方 TorchNPU 版本表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)

CANN 9.1 将 Toolkit 与 ops 分为不同组合包；HCCL 属于 ops 内的通信组件。只装 Toolkit 不能直接判定 HCCL 已就绪，须核对实际安装组件和动态库。具体包还依赖服务器架构、系统及适用产品，不能从目标板名字 X90 推导出应该安装哪一个公开 Ascend ops 包。[官方 CANN 9.1 组件与升级说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/softwareinst/instg/instg_0030.html)

AIR 导出和 OMG 转换使用各自清楚记录的环境；建议分开终端，避免两套不同版本的库互相覆盖。`CPRO_DDK_ROOT` 继续指向内部 DDK，不改成服务器 CANN 目录。生成的 AIR 及外部权重仍需留在原导出路径，转换端使用同一路径。

## 1. 记录服务器架构、系统和包版本

以下只读命令在服务器执行，不需要板机权限，也不会加载 TorchNPU：

```bash
uname -m
cat /etc/os-release
getconf LONG_BIT
getconf GNU_LIBC_VERSION
export MODEL_PYTHON="$(command -v python3)"
"$MODEL_PYTHON" - <<'PY'
import sys
from importlib.metadata import version, PackageNotFoundError
print("Python:", sys.version.split()[0])
for name in ("torch", "torch-npu", "transformers", "tokenizers", "huggingface-hub", "safetensors"):
    try:
        print(name, version(name))
    except PackageNotFoundError:
        print(name, "NOT INSTALLED")
PY
```

`x86_64`、`aarch64` 指服务器 CPU 架构，不能根据板端 ARM64 来替服务器选包。包版本查询成功只说明 Python 安装记录存在，不代表其原生 `.so` 能加载。

## 2. `libhccl.so` 缺失与临时恢复 Torch

安装 TorchNPU 后，`import torch` 可能自动加载它，连原来的 CPU/CUDA 操作也因此失败。临时检查原 Torch 可用以下命令，仅为该进程关闭自动加载：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$MODEL_PYTHON" - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
```

用户日志中它恢复了 `2.11.0+cu130` 的导入，但 `torch.cuda.is_available()` 曾为 `False`；`+cu130` 只是包构建信息，尚不能按 GPU 可用来执行 `--device cuda:0`。

**`TORCH_DEVICE_BACKEND_AUTOLOAD=0` 不修复 HCCL，也不禁用显式 `import torch_npu`。** 下面是导出依赖检查，缺库时仍应失败，不能把上一条成功当作 AIR 环境就绪：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$MODEL_PYTHON" - <<'PY'
import torch
import torch_npu
import torchair
print("TorchNPU:", torch_npu.__version__)
print("TorchAir:", torchair.__file__)
assert callable(torchair.dynamo_export)
PY
```

解决方向是提供本任务可用、版本配套的完整服务器 CANN 依赖，并加载该安装提供的环境脚本；若已安装，检查其真实运行库是否进入当前进程的库搜索路径。不要用开发包中的 stub/fake 库充当运行库，也不要将别人目录里的同名 `.so` 加入路径。未确定受支持的服务器安装组合前，本指南不提供猜测的 ops 下载或安装命令。

即使 import 全部成功，仍须实际执行 tiny AIR 导出。TorchAir 可能在构图阶段初始化 GE/NPU；CPU 张量或“离线导出”名称不能保证当前版本在无昇腾卡服务器上可用。完整量化还另需 DOPT 的原生量化扩展，tiny 浮点图通过也不覆盖它。

## 3. Transformers 依赖不要逐个无约束升级

用户日志中的 `transformers==5.13.0` 要求 `tokenizers>=0.22.0,<=0.23.0`、`huggingface-hub>=1.5.0,<2.0`。曾装入的 tokenizers 0.23.3 和 hub 2.2.0 不满足要求；不要因为 0.23.0 在当前包源不存在就直接选择最新版。可固定 tokenizers 0.22.2，它落在声明范围内。[Transformers 5.13.0 官方依赖声明](https://github.com/huggingface/transformers/blob/v5.13.0/setup.py)

在选定的模型 Python 环境中先预览完整约束的解析结果：

```bash
"$MODEL_PYTHON" -m pip install --dry-run \
  "transformers==5.13.0" "tokenizers==0.22.2" \
  "huggingface-hub>=1.5.0,<2.0" "safetensors==0.8.0"
"$MODEL_PYTHON" -m pip check
```

独立模型环境确认解析结果后，去掉 `--dry-run` 执行同一组约束，让解析器补齐 regex、typer 等依赖；不要用 `--no-deps` 留下半套安装。此命令不要求替换 Torch，但仍应检查实际解析计划。

现有 vendor 环境还报过 TensorFlow/protobuf 冲突及 `te` 平台不支持。它们不等于 Qwen 导出需要 TensorFlow；不要为了让 `pip check` 全绿而盲目升级 protobuf、卸载 te 或改动共享 vendor 环境。保留检查结果，模型依赖与导出依赖应在确认过的环境内处理。

## 4. 内部 DDK 目录和下一步

内部 DDK 没有蓝区的 `tools/platform/kirinx90` 目录，本身不能证明工具缺失。当前转换代码支持已加载环境的内部布局，使用实际存在的 `tools/tools_omg/master/omg`；不要补一个指向别的芯片或别人的插件目录的链接来满足目录检查。内部环境能启动 OMG，也仍需真实 AIR 才能验证转换。

处理顺序为：原 Torch/模型依赖可导入 → TorchNPU/TorchAir 原生依赖可导入 → [tiny AIR 导出](tiny-validation.md) → 内部 OMG 生成 OMC → 正式 HIAI runner 上板。每一步保留新的输出路径和对应错误日志；尚未生成成功 AIR 时，重复执行 OMG 或修改板端命令不能消除当前缺库问题。
