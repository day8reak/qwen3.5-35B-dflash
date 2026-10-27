# 环境准备

[首页](README.md) · [PyTorch 指南](pytorch-guide.md) · [GE 指南](ge-guide.md) · [板端指南](board-guide.md)

沿用已安装的模型环境和工具链，只配置本次所需阶段；无需替换已有 PyTorch 或共享 DDK。

## Docker 重进后的配置恢复

当前算子探针流程可以使用仓库自带的 Bash 入口：

```bash
source targets/cpro/scripts/env.sh
```

首次执行会在**仓库旁边**创建 `qwen35-cpro.env`（权限600），复制当前已设置的相关变量并保留缺项为空，然后提示填写。配置文件不放在运行目录里，清理运行结果时不会被一起删掉。编辑一次该文件，再执行同一条 `source` 即可；也可指定私有配置路径：`source targets/cpro/scripts/env.sh /absolute/path/cpro.env`。

| 必填配置 | 内容 |
|---|---|
| `REPO_DIR` | 项目绝对路径，首次自动填当前仓库 |
| `AI_RUN_DIR` | 仓库外运行根目录，默认是项目路径加 `-run`；每次source保持不变，各任务自动分配新子目录 |
| `MODEL_PYTHON` | 已安装模型环境的Python绝对路径，如选定conda环境的 `bin/python3` |
| `CANN_ROOT` | 用于GE导出AIR的CANN目录 |
| `CPRO_DDK_ROOT`、`CPRO_DDK_ID` | 当前目标DDK目录及准确版本 |
| `OHOS_NATIVE_SDK` | 包含 `build/cmake/ohos.toolchain.cmake`、`llvm/bin/clang++`、`sysroot/` 的 `native/` 目录 |
| `HIAI_INCLUDE_DIR` | 包含 `HiAiModelManagerService.h` 的目录；或 `auto` |
| `HIAI_LIBRARIES` | 主机上HarmonyOS ARM64链接库文件；多个路径用冒号分隔；或 `auto` |
| `HDC`、`BIP`、`SN` | HDC可执行文件或PATH中的命令名、**HDC服务器IP:端口**、设备序列号；BIP不是自动推断的手机IP |

配置只保存上表的固定环境。板端私有库目录 `BOARD_HIAI_LIB_DIR` 和两套权重路径可按需自行加入，不生成空占位。首次初始化复制已有声明，支持从已有 `OHOS_SDK` 获取 `OHOS_NATIVE_SDK`；没有HIAI路径时在模板中写 `auto`。设备、安装包和历史产物不自动选择。

`FUSED_PROBE_RUN`、`TBE_DIAG_JOB`、`GE_CUSTOM_BINDINGS`、`FIRMWARE_ID`、`HIAI_RUNTIME_VERSION` 属于单次运行参数，不放入配置；需要时在运行命令前设置或使用对应CLI参数。`source env.sh` 不读取、覆盖或清除调用者的这些变量；旧配置里若有这些行，可直接删掉。

HIAI的 `auto` 只在**已指定的DDK目录**内查找，深度最多8层；头文件必须唯一，`libhiai.so` 必须唯一且位于含 `ohos` 或 `*-ohos` 目录的路径下，并通过 `readelf` 的ELF64、小端、AArch64、共享库头检查。不会使用OMG的x86库、猜其他库名或跟随目录链接进入其他安装。多个候选或未找到会列出路径并要求显式填写；这只是构建候选选择，不是ABI兼容性或设备通过证明。

该入口检查必填变量和工具文件，检查成功后才修改当前shell；自动设置 `GE_MODEL_TOOL`、`CPRO_TARGET=kirinx90`、模型 `PYTHONPATH`、Python的bin路径以及环境变量驱动的 `specs/resources.json`。重复source不会累加路径。它不激活conda、不运行vendor环境脚本、不全局混入CANN/DDK原生库，也不连接设备；GE/OMG调用继续由现有脚本设置子进程环境。原私有 `QWEN35_RESOURCES` 会被上述仓库清单替换，所选头文件/库应填写到配置变量中。

**如果旧运行目录已经清空：** 保留权重目录，恢复环境后一次重建全部探针，无需设置任何历史产物变量：

```bash
"$MODEL_PYTHON" "$REPO_DIR/targets/cpro/scripts/run_fused_probes.py" --all-ops --fresh
```

`--fresh` 依次生成安装源码约束的FP16 GDN状态绑定、12例TBE平台诊断、全部五类九例AIR/OMC，然后执行编译成功的case。它忽略旧产物环境变量、不读取权重、不删除目录，也不会在本批中重复编译刚失败的case。失败项仍保留在总表；这是重新执行流程，不是宣称Conv/RMS或数值问题已经修复。SDK或HIAI库若此前也放在已删除目录内，必须恢复那些文件，仅重新export无法恢复工具链。

后续重试使用打印出的批次目录；新批次会记录TBE诊断路径，重试自动沿用，不必另找 `TBE_DIAG_JOB`：

```bash
"$MODEL_PYTHON" "$REPO_DIR/targets/cpro/scripts/run_fused_probes.py" --all-ops \
  --probe-run /absolute/path/to/fused-all-new-batch \
  --retry-cases attention_prefill,attention_decode
```

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
