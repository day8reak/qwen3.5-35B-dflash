# 环境准备

[首页](README.md) · [PyTorch 指南](pytorch-guide.md) · [GE 指南](ge-guide.md) · [板端指南](board-guide.md)

沿用已安装的模型环境和工具链，只配置本次所需阶段；无需替换已有 PyTorch 或共享 DDK。

## Docker 重进后的配置恢复

先激活已安装的模型conda环境，并沿用本机已确认的当前DDK环境入口；然后在仓库根目录执行：

```bash
source targets/cpro/scripts/env.sh
```

首次执行会在**仓库旁边**创建 `qwen35-cpro.env`（权限600），保存固定配置并提示补齐缺项。配置不放在运行目录里，清理结果时不会被一起删掉。填写一次后，每次重进Docker、恢复conda/DDK环境，再执行同一条 `source`。也可指定文件：`source targets/cpro/scripts/env.sh /absolute/path/cpro.env`。

默认配置只保留下面七项。已有值会填入模板；项目路径默认取当前仓库，运行根目录默认取项目路径加 `-run`，其余不能确定的值需要填写。

| 固定配置 | 含义 |
|---|---|
| `REPO_DIR` | 项目绝对路径，用来定位脚本、模型代码和资源清单 |
| `AI_RUN_DIR` | 仓库外的固定产物根目录；不是某次临时批次目录，每个任务会在下面分配新子目录 |
| `CANN_ROOT` | 服务器端CANN目录，用其中的GE头文件和库将图导出为AIR；与目标DDK用途不同 |
| `CPRO_DDK_ID` | 所选DDK的准确版本标识，用于绑定/编译产物一致性检查；不会从路径猜版本 |
| `OHOS_NATIVE_SDK` | 编译手机端runner的OHOS Native SDK，必须指向含 `build/cmake/ohos.toolchain.cmake`、`llvm/bin/clang++`、`sysroot/` 的 `native/`；已有 `OHOS_SDK` 可用于填初值 |
| `BIP` | **HDC服务器的IP:端口**，例如远端转发的HDC服务；不是根据手机网络地址自动推断的值 |
| `SN` | HDC列出的目标设备序列号，用于选择具体手机 |

下面这些值可以从当前环境派生，不用把vendor脚本已经export的内容再抄一遍。必要时也可在私有配置中显式指定覆盖值。

| 可省略配置 | 默认来源与作用 |
|---|---|
| `CPRO_DDK_ROOT` | 使用已选DDK的 `DDK_PATH`；定位OMG、TBE和安装版算子源码。未加载DDK时需填写绝对路径 |
| `MODEL_PYTHON` | 使用已声明的模型Python，或当前conda环境 `CONDA_PREFIX/bin/python3`；执行探针数据、平台诊断及编译调度，不重新安装环境 |
| `HDC` | 已声明的HDC路径，或PATH中的 `hdc`；执行设备传输与命令。找不到时需填写实际可执行文件 |
| `HIAI_INCLUDE_DIR` | 默认 `auto`，在所选DDK内查找唯一 `HiAiModelManagerService.h` 的父目录；也可明确指定 |
| `HIAI_LIBRARIES` | 默认 `auto`，在所选DDK内查找手机端链接库；也可指定主机上的HarmonyOS ARM64库文件，多文件用冒号分隔 |

vendor环境中常见的变量含义如下；入口保留它们，不生成重复配置。路径存在只说明环境已声明，不代表芯片编译或上板通过。

| vendor变量 | 含义 |
|---|---|
| `DDK_PATH` | 当前目标DDK的根目录，供 `CPRO_DDK_ROOT` 派生 |
| `TOOLCHAIN_HOME` | vendor工具链的根目录/内部查找基准 |
| `HIAI_VERSION` | 此DDK环境中通常指OMG的master目录；不是板端 `HIAI_RUNTIME_VERSION` 字符串 |
| `PATH` | OMG、编译器、Python等可执行文件的查找路径 |
| `LD_LIBRARY_PATH` | 主机工具运行时的动态库查找路径；不是给手机部署的库列表 |
| `PYTHONPATH` | vendor Python模块查找路径；项目入口仅补入缺少的仓库模块路径 |

这些vendor变量不能替代 `CANN_ROOT`、OHOS Native SDK、准确DDK版本或实际手机端HIAI链接库的选择。`HIAI_LIBRARIES=auto` 只在已选DDK目录内查找，深度最多8层；候选必须是唯一的 `libhiai.so`，位于含 `ohos` 或 `*-ohos` 目录的路径下，并通过 `readelf` 的ELF64、小端、AArch64、共享库头检查。头文件也必须唯一。多个候选或未找到会列出路径，改为明确填写即可；不猜其他库名，不使用OMG的x86库，不跟随目录链接扫描其他安装。这只是构建候选选择，ABI仍需链接及设备验证。

下面是按需使用的固定配置，不生成空占位：

| 可选变量 | 含义 |
|---|---|
| `BOARD_HIAI_LIB_DIR` | **手机上**已有的私有HIAI库目录；与主机端 `HIAI_LIBRARIES` 不同。留空时脚本按runner构建记录打包部署库 |
| `QWEN35_TARGET_WEIGHTS` | 完整目标模型的检查点目录；算子探针不读取它 |
| `QWEN35_DRAFT_WEIGHTS` | DFlash草稿模型的检查点目录；算子探针不读取它 |
| `QWEN35_ENV_CONFIG` | 私有环境配置文件路径；命令行给 `source env.sh` 的路径优先，其次此变量，最后仓库旁默认文件 |

入口检查成功后才修改当前shell，并自动导出以下值，无需手填：

| 自动变量 | 含义 |
|---|---|
| `GE_MODEL_TOOL` | 仓库内 `targets/cpro/scripts/direct_model_air.py` 的绝对路径 |
| `CPRO_TARGET` | 固定为当前部署目标 `kirinx90` |
| `QWEN35_RESOURCES` | 指向仓库 `specs/resources.json`，由上面的变量解析资源；原私有清单会被此设置替换 |
| `PYTHONDONTWRITEBYTECODE` | 固定为1，防止Python在代码/工具链目录写缓存 |
| `PATH`、`PYTHONPATH` | 保留已有vendor路径，加入模型Python的bin及缺少的仓库模块路径；重复source不累加 |

入口不激活conda、不执行vendor脚本、不追加全局CANN/DDK库路径，也不连接设备；GE/OMG继续通过现有脚本设置各自子进程环境。

以下变量属于单次运行，**不写入固定配置**，需要时在执行前设置或使用CLI参数。`source env.sh` 不读取、覆盖或清除调用者的这些值；旧配置有这些行可删除。

| 单次运行变量 | 含义 / 替代入口 |
|---|---|
| `FUSED_PROBE_RUN` | 已生成探针或批次目录；可直接使用 `--probe-run` |
| `TBE_DIAG_JOB` | 包含 `diagnosis/` 的诊断任务目录；新批次重试可自动继承记录，无需另行export |
| `GE_CUSTOM_BINDINGS` | 直接调用GE工具时使用的bindings JSON路径；`--fresh` 自动重新生成 |
| `FIRMWARE_ID` | 本次设备固件身份的显式值；未设置时脚本从设备查询 |
| `HIAI_RUNTIME_VERSION` | 本次预期板端HIAI版本；若设置，必须与实际runtime probe一致 |

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
