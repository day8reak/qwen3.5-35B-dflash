# 环境准备

[返回首页](README.md) · [当前进展与下一步](current-status.md)

按要执行的阶段准备依赖即可。保留现有 PyTorch 和已验证的 DDK/SDK，不用为了直接 GE 构图重新安装 TorchNPU，也不修改共享工具链。

| 要做的事 | Python / 主机依赖 | 额外资源 |
|---|---|---|
| 生成文档中的 GE 融合接口 JSON | Python、NumPy | 无需权重、CANN、DDK 或设备；目标/DDK 名称只是声明 |
| 直接 GE `prepare --tiny` / `plan-stateful` | Python、NumPy | `plan-stateful` 读取已有 prepared；完整权重准备另需完整 safetensors 权重文件 |
| 直接 GE 导出 AIR | 上述依赖、主机 C++ 编译器、匹配的 GE 头文件和 `libgraph.so` | `CANN_ROOT` |
| PyTorch CPU/CUDA 参考 | 现有 Torch、Transformers、safetensors 等[模型依赖](../model/requirements-reference.txt) | tiny 无需下载权重；完整模型另需权重与足够内存/显存 |
| PyTorch → AIR | 配套的 Torch、TorchNPU、TorchAir、服务器 CANN 原生运行库 | 实际 converter、自定义算子注册及所选导出栈所需设备条件 |
| AIR → OMC | 支持目标平台的内部 DDK、OMG 及其配套环境 | `CPRO_DDK_ROOT`、`CPRO_DDK_ID` |
| 构建 HIAI 程序 | CMake、OHOS Native SDK、匹配的 HIAI 头文件/ARM64 库 | `OHOS_NATIVE_SDK`、`HIAI_INCLUDE_DIR`、`HIAI_LIBRARIES` |
| 上板执行 | HDC、板端固件、已构建程序和 OMC | `SN`、`BIP`；板端不安装 PyTorch/CANN/编译 SDK |

## 1. 公共变量

从仓库根目录执行，先激活已有模型 Python 环境。工作区用户沿用 `ws context`、`ws session start`、`ws env` 分配的环境和 `AI_RUN_DIR`；不要另外创建环境副本。

独立 clone 且尚无 `AI_RUN_DIR` 时，先把下面的存储目录换成实际的**仓库外**目录。已有当前任务目录则跳过这段：

```bash
export QWEN35_RUN_STORAGE='/your/run/storage'
mkdir -p "$QWEN35_RUN_STORAGE"
export AI_RUN_DIR="$(mktemp -d "$QWEN35_RUN_STORAGE/qwen35-XXXXXXXX")"
```

然后设置项目路径并记录解释器。若当前 shell 的 `python3` 是算子工具链专用解释器，应先把 `MODEL_PYTHON` 设置为模型环境解释器的绝对路径。

```bash
export REPO_DIR="$(pwd -P)"
test -f "$REPO_DIR/project.yaml"
export MODEL_PYTHON="${MODEL_PYTHON:-$(command -v python3)}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO_DIR/model:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
: "${AI_RUN_DIR:?请先设置本次仓库外输出目录}"
mkdir -p "$AI_RUN_DIR"
export ENV_CHECK_RUN="$(mktemp -d "$AI_RUN_DIR/env-check-XXXXXXXX")"

"$MODEL_PYTHON" - <<'PY' > "$ENV_CHECK_RUN/python-packages.txt"
import sys
from importlib.metadata import PackageNotFoundError, version
print("interpreter:", sys.executable)
print("python:", sys.version)
for name in ("numpy", "torch", "transformers", "safetensors", "torch-npu", "torchair"):
    try:
        print(name, version(name))
    except PackageNotFoundError:
        print(name, "NOT INSTALLED")
PY
cat "$ENV_CHECK_RUN/python-packages.txt"
```

包版本记录不会导入 TorchNPU，也不证明原生库可加载。后续所有产物都放在 `AI_RUN_DIR` 下的新路径，失败目录保留；脚本拒绝覆盖已有证据。

资源默认读取 [specs/resources.json](../specs/resources.json) 并展开环境变量。已有私有资源清单时设置 `QWEN35_RESOURCES=/实际路径/resources.json`；其中的具体路径优先于模板环境变量。私有权重、库和安装路径不提交到仓库。

## 2. 直接 GE 路径

生成绑定和 tiny prepared、构图源码只需要 Python/NumPy。检查当前解释器即可：

```bash
"$MODEL_PYTHON" -c 'import numpy; print("NumPy:", numpy.__version__)'
"$MODEL_PYTHON" "$REPO_DIR/targets/cpro/scripts/direct_model_air.py" --help
```

需要真正导出 AIR 时，再设置服务器 CANN 的实际版本目录。下面的根目录应包含 `include/`、`lib64/` 和该安装的环境脚本：

```bash
export CANN_ROOT='/your/server/cann-version'
source "$CANN_ROOT/set_env.sh"
test -f "$CANN_ROOT/include/graph/graph.h"
test -f "$CANN_ROOT/include/graph/operator.h"
test -f "$CANN_ROOT/lib64/libgraph.so"
"${CXX:-c++}" --version
```

生成器链接真实主机 GE 库；同套库的依赖也必须齐全。`--cxx11-abi` 默认 `0`，应匹配所选 GE 库。端侧 `libgraph_host.so` 不能改名替代它。模型构图与导出命令见 [GE 指南](ge-guide.md)。

## 3. PyTorch 路径

参考模型与 AIR 导出分别检查。CPU/CUDA 参考使用现有 Torch；当前 CLI 会在参考进程内隔离可选 NPU 后端探测，不需要卸载 TorchNPU。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$MODEL_PYTHON" - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
```

PyTorch 导出 AIR 则必须能真正导入匹配的 TorchNPU/TorchAir 及其原生依赖。使用已加载服务器 CANN 环境的新进程检查：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$MODEL_PYTHON" - <<'PY'
import torch
import torch_npu
import torchair
print("Torch:", torch.__version__)
print("TorchNPU:", torch_npu.__version__)
print("TorchAir:", torchair.__file__)
assert callable(torchair.dynamo_export)
PY
```

导入成功后仍需实际导出验证。`libhccl.so`、Transformers 可选后端探测和依赖版本问题见 [AIR 环境排障](air-troubleshooting.md)；当前状态以[进展页](current-status.md)为准。不要照抄历史安装步骤替换已经可用的 Torch。新 PyTorch 模型与缓存入口见 [融合算子与缓存模型](pytorch-guide.md)。

完整权重路径只在加载完整模型时设置；下载与锁定版本见[权重来源](weights.md#权重来源)。

## 4. 目标 DDK：AIR → OMC

`CANN_ROOT` 是服务器 AIR 生成环境；`CPRO_DDK_ROOT` 是目标 OMC 编译工具链，两个目录用途不同。切回已经加载内部 DDK 环境脚本的终端，保留该脚本设置的路径；尚未加载时 source 实际的 `set_ascendc_env.sh`，再执行：

```bash
export CPRO_DDK_ROOT="$(realpath "${CPRO_DDK_ROOT:-${DDK_PATH:?请先加载内部 DDK 环境}}")"
export CPRO_DDK_ID='填写当前安装包的准确版本'
export CPRO_TARGET=kirinx90
test -x "$CPRO_DDK_ROOT/tools/tools_omg/master/omg"
```

本仓库现有编译入口使用 `--framework=1 --target=omc --platform=kirinx90`，与已记录的转换命令一致。`CPRO_DDK_ID` 由用户声明，不是脚本自动识别结果；内部包不必具有公开插件的 `tools/platform/kirinx90` 布局。保持已有安装，不向共享 DDK 安装开发包。新融合算子的注册与目标支持仍需另行核对，不能由文档生成的接口 JSON 证明。

生成接口 JSON **不要求先执行本节检查**：其中 `--target` / `--ddk-id` 只记录拟使用的标识。当前 stateful GE 产物也不能直接交给旧 target/draft 双图编译及上板入口，具体边界见[当前进展](current-status.md)。

## 5. OHOS / HIAI / HDC

需要构建现有板端程序时才配置以下资源；复用此前成功的 SDK、头文件和同套 HarmonyOS ARM64 库：

```bash
export OHOS_NATIVE_SDK='/your/ohos/native'
export HIAI_INCLUDE_DIR='/your/hiai/include'
export HIAI_LIBRARIES='/your/hiai/ohos/libhiai.so'
test -f "$OHOS_NATIVE_SDK/build/cmake/ohos.toolchain.cmake"
test -x "$OHOS_NATIVE_SDK/llvm/bin/clang++"
test -d "$OHOS_NATIVE_SDK/sysroot"
test -f "$HIAI_INCLUDE_DIR/HiAiModelManagerService.h"
cmake --version
```

多个链接库用冒号分隔；私有资源 JSON 中非空的 `hiai_libraries` 优先于 `HIAI_LIBRARIES`。缺少 Native SDK 时按 [OHOS SDK 获取与检查](ohos-native-sdk.md)操作，不能只复制一份 toolchain 文件或板端 `.so`。

连接板端时设置实际设备序列号和 HDC server 地址。`BIP` 是 HDC server 的 `IP:端口`，不是随意填写的板端 IP：

```bash
export HDC="${HDC:-hdc}"
export SN='实际设备序列号'
export BIP='实际HDC服务器IP:端口'
"$HDC" -s "$BIP" list targets
```

现有 `run_hdc.py` 从 runner 构建目录的 `command.json` / `result.json` 找到并校验同套运行库；保留这些记录及其引用的 SDK/DDK 路径。具体构建和执行命令按[当前进展](current-status.md)选择，旧 ABI v2 流程另见[上板指南](board-guide.md)。

## 内存预算：文档 W4A8 + DFlash

按batch=1、上下文2048、文本主干、真实INT4打包target和FP16 draft估算：target权重及量化参数约17.35 GiB；draft自身约0.72 GiB，若独立OMC另带FP16 embedding/head再加约1.89 GiB。单份target/draft缓存及目标层特征合计约0.18 GiB；激活、状态分支、HIAI工作区及加载副本另计。

先按模型进程可用24–28 GiB规划，**不是已经验证的峰值上限，也不是物理总内存**。若HIAI额外保留大份权重副本，峰值可能更高。此前32GB总内存、约18G空闲的报告不足以证明能同时常驻；要区分MemFree与MemAvailable。该预算也不适用于当前未量化的GE FP16基线。计算假设保存在[内存记录](../specs/device-memory-report.json)，最终以采用的OMC及runner实测为准。
