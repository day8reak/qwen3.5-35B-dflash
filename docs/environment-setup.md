# 环境准备

[首页](README.md) · [PyTorch 指南](pytorch-guide.md) · [GE 指南](ge-guide.md) · [板端指南](board-guide.md)

沿用已安装的模型环境和工具链，只配置本次所需阶段；无需替换已有 PyTorch 或共享 DDK。

| 阶段 | 依赖 / 资源 |
|---|---|
| 完整 PyTorch 结构（meta） | 现有 Torch、Transformers 等[模型依赖](../model/requirements-reference.txt)，不需权重 |
| 完整 GE 结构、文档 bindings | Python/NumPy，不需权重、CANN、DDK 或设备 |
| 完整参考 / 权重准备 | 两套锁定检查点；PyTorch 参考另需足够内存/显存 |
| 直接 GE → AIR | C++ 编译器、主机 GE 头文件与 `libgraph.so` |
| PyTorch → AIR | 配套 TorchNPU、TorchAir、CANN 及实际 converter/算子注册 |
| AIR → OMC | 目标 DDK、OMG 及其环境 |
| 构建 / 上板 | OHOS Native SDK、HIAI 头文件及 ARM64 库 / HDC、设备固件 |

## 1. 公共变量

从仓库根目录执行，先激活模型 Python 环境；工作区用户沿用 `ws` 分配的 `AI_RUN_DIR`。独立 clone 若无任务目录，先将下面路径改为实际仓库外存储目录：

```bash
if [ -z "${AI_RUN_DIR:-}" ]; then
  export QWEN35_RUN_STORAGE='/实际路径/仓库外运行目录'
  mkdir -p "$QWEN35_RUN_STORAGE"
  export AI_RUN_DIR="$(mktemp -d "$QWEN35_RUN_STORAGE/qwen35-XXXXXXXX")"
fi
export REPO_DIR="$(pwd -P)"
test -f "$REPO_DIR/project.yaml"
export MODEL_PYTHON="${MODEL_PYTHON:-$(command -v python3)}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO_DIR/model:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$AI_RUN_DIR"
"$MODEL_PYTHON" -c 'import sys, numpy; print(sys.executable); print("NumPy:", numpy.__version__)'
```

资源默认读取 [specs/resources.json](../specs/resources.json) 并展开环境变量；已有私有清单时设置 `QWEN35_RESOURCES=/实际路径/resources.json`。完整权重路径与锁定版本见[权重准备](weights.md)，每次命令使用新的输出目录。

## 2. PyTorch 环境

完整结构和 CPU/CUDA 参考使用现有 Torch；先检查导入。完整结构命令见 [PyTorch 指南](pytorch-guide.md)。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$MODEL_PYTHON" - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
```

导出 AIR 时再检查配套 TorchNPU/TorchAir；先加载下节服务器 CANN 环境。原生库或版本问题见 [AIR 排障](air-troubleshooting.md)。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$MODEL_PYTHON" - <<'PY'
import torch_npu
import torchair
print("TorchNPU:", torch_npu.__version__)
print("TorchAir:", torchair.__file__)
assert callable(torchair.dynamo_export)
PY
```

## 3. 主机 GE：生成 AIR

`CANN_ROOT` 指实际服务器 CANN 版本目录；直接 GE 构图的源码阶段不需要它，保存真实 AIR 才需要。`--cxx11-abi` 默认 0，需与所选 GE 库匹配。

```bash
export CANN_ROOT='/实际路径/服务器CANN版本目录'
source "$CANN_ROOT/set_env.sh"
test -f "$CANN_ROOT/include/graph/graph.h"
test -f "$CANN_ROOT/include/graph/operator.h"
test -f "$CANN_ROOT/lib64/libgraph.so"
"${CXX:-c++}" --version
```

## 4. 目标 DDK：编译 OMC

在已加载内部 DDK 环境脚本的终端设置准确版本；`CANN_ROOT` 与 `CPRO_DDK_ROOT` 用途不同。当前编译入口固定使用 OMG 的 `--platform=kirinx90`，文档 bindings 不证明 kernel 已安装或兼容。

```bash
export CPRO_DDK_ROOT="$(realpath "${CPRO_DDK_ROOT:-${DDK_PATH:?请先加载内部DDK环境}}")"
export CPRO_DDK_ID='当前安装包的准确版本'
export CPRO_TARGET=kirinx90
test -x "$CPRO_DDK_ROOT/tools/tools_omg/master/omg"
```

## 5. OHOS / HIAI / HDC

构建 runner 时配置实际 SDK、头文件和同套 HarmonyOS ARM64 库；多个库用冒号分隔，私有资源 JSON 中非空 `hiai_libraries` 优先。SDK 获取方法见 [OHOS SDK](ohos-native-sdk.md)。

```bash
export OHOS_NATIVE_SDK='/实际路径/ohos/native'
export HIAI_INCLUDE_DIR='/实际路径/hiai/include'
export HIAI_LIBRARIES='/实际路径/hiai/ohos/libhiai.so'
test -f "$OHOS_NATIVE_SDK/build/cmake/ohos.toolchain.cmake"
test -x "$OHOS_NATIVE_SDK/llvm/bin/clang++"
test -d "$OHOS_NATIVE_SDK/sysroot"
test -f "$HIAI_INCLUDE_DIR/HiAiModelManagerService.h"
cmake --version
```

连接设备时设置序列号和 HDC server 地址；`BIP` 为 server 的 `IP:端口`。保留 runner 构建目录中的 `command.json`、`result.json` 及其引用路径，上传脚本据此查找运行库。

```bash
export HDC="${HDC:-hdc}"
export SN='实际设备序列号'
export BIP='实际HDC服务器IP:端口'
"$HDC" -s "$BIP" list targets
```

## 内存预算：文档 W4A8 + DFlash

完整未量化 FP16 target+draft 去重权重约 **65.28 GiB**；不要以 32GB 板端容量运行它。文档 W4A8 + FP16 draft 的规划权重约 20 GiB，激活、缓存、HIAI workspace 和加载副本另计，真实峰值尚未验证；假设见[内存记录](../specs/device-memory-report.json)。

## tiny 回归

小模型只免除完整权重依赖；AIR、OMC 和板端阶段仍需对应工具链。命令放在 [PyTorch](pytorch-guide.md#tiny-回归) / [GE](ge-guide.md#tiny-回归) 指南末尾。
