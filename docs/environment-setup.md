# 环境准备

[首页](README.md) · [PyTorch 指南](pytorch-guide.md) · [GE 指南](ge-guide.md) · [板端指南](board-guide.md)

沿用已安装的模型环境和工具链，只配置本次所需阶段；无需替换已有 PyTorch 或共享 DDK。

## Docker 重进后的配置恢复

按下面顺序执行。第1步每次重进Docker恢复；第2、3步首次配置时执行；已有 `qwen35-cpro.env` 时，第1步后直接进入第4步。示例中的安装路径和设备值需替换为本机实际值，私有路径和设备配置保存在仓库外。

### 第1步：恢复conda和DDK环境

```bash
conda activate custom_pypto
source '/实际DDK根目录/tools/tools_ascendc/set_ascendc_env.sh'
```

脚本相对路径来自用户成功回传的入口：`tools/tools_ascendc/set_ascendc_env.sh`，不是 `tool/tools_ascendc/set_ascend_env.sh`。若当前终端已经激活该conda并成功source该DDK，跳过本步，不重复加载。

后续命令直接使用当前环境的 `python3`，无需export `MODEL_PYTHON`。平台诊断和runner构建子进程沿用启动脚本的解释器。

该次source输出已包含 `DDK_PATH`、`TOOLCHAIN_HOME`、`HIAI_VERSION`、`PATH`、`LD_LIBRARY_PATH`、`PYTHONPATH`，后续不再手工export这六项。它没有给出项目目录、运行根目录、主机CANN、准确DDK版本、OHOS Native SDK及设备连接信息，首次需要补下面七项。

### 第2步：只补充项目需要的七项

进入项目仓库根目录后执行一次。下列变量会在第3步保存，不需要每次重进Docker重新输入。

```bash
# 项目源码；先cd到仓库根目录
export REPO_DIR="$(pwd -P)"
# 固定运行产物根目录，不是某次fused-all批次
export AI_RUN_DIR="${REPO_DIR}-run"
# 主机GE头文件和libgraph，用于导出AIR；不是上面的DDK
export CANN_ROOT='/实际CANN版本目录'
# 填安装包准确版本；此前回传版本为100.520.020.010
export CPRO_DDK_ID='100.520.020.010'
# 编译手机端runner的Native SDK
export OHOS_NATIVE_SDK='/实际SDK目录/openharmony/native'
# HDC服务器IP:端口，以及其连接的手机序列号
export BIP='实际HDC服务器IP:端口'
export SN='实际设备序列号'
```

不用另填 `CPRO_DDK_ROOT`、`HDC` 或重写搜索路径：项目入口会读取 `DDK_PATH` 和PATH中的 `hdc`。第3步会自动写入HIAI头文件及完整三库的默认配置，无需再手工填写两项。不要把OMG的主机 `LD_LIBRARY_PATH` 当作手机端链接库路径。

### 第3步：首次保存配置

```bash
source "$REPO_DIR/targets/cpro/scripts/env.sh"
```

首次会在**仓库旁边**创建 `qwen35-cpro.env`（权限600），保存第2步的七项，并自动增加带注释的 `HIAI_INCLUDE_DIR`、`HIAI_LIBRARIES`，正常场景共九项。HIAI默认表达式在每次读取配置时随 `DDK_PATH` 展开，指向 `ddk/ai_ddk_lib/include` 和 `lib64` 下完整三库；生成时已有明确的非 `auto` HIAI值会原样保留。若尚未加载DDK、模板需要保存 `CPRO_DDK_ROOT`，HIAI表达式改为引用该项。

首次创建返回1表示配置尚未加载，提示再次加载，不是编译失败。已有文件不会被覆盖，也不会因再次export而更新它；需要变更时编辑原文件，无需删除重建。若第2步使用了示例占位值，先在文件里改为实际值，再继续第4步。

配置不放在运行目录里，清理结果时不会被一起删掉。也可指定文件：`source targets/cpro/scripts/env.sh /absolute/path/cpro.env`。

### 第4步：加载项目配置并使用已有批次

在仓库根目录更新代码并加载配置：

```bash
git pull --ff-only &&
source targets/cpro/scripts/env.sh
```

当前已保留新批次 `fused-all-f6b81ead7ae8`，GDN、CacheUpdate、Attention六例已完成调用，无需再次 `--fresh`。加载成功后按[GE指南](ge-guide.md#融合算子调用探针)用 `--model-ops` 复用这六例；Conv/RMS自定义算子已退出当前必需范围。

**仅首次开始或运行产物已清空时**，才从零重建：

```bash
source targets/cpro/scripts/env.sh &&
  python3 "$REPO_DIR/targets/cpro/scripts/run_fused_probes.py" --model-ops --fresh
```

此入口自动生成FP16 GDN状态绑定、TBE平台诊断，尝试导出/编译所选三类六例AIR/OMC，再构建runner并执行编译成功的case。无需另export bindings、诊断或探针路径；不读取或修改保留的权重。编译失败仍列入总表，命令不是宣称已修复Conv/RMS或数值问题。

### 配置字段说明

默认配置包含手工声明的七项和自动写入的两项HIAI配置。已有值会填入模板；项目路径默认取当前仓库，运行根目录默认取项目路径加 `-run`，其余不能确定的值需要填写。

| 固定配置 | 含义 |
|---|---|
| `REPO_DIR` | 项目绝对路径，用来定位脚本、模型代码和资源清单 |
| `AI_RUN_DIR` | 仓库外的固定产物根目录；不是某次临时批次目录，每个任务会在下面分配新子目录 |
| `CANN_ROOT` | 服务器端CANN目录，用其中的GE头文件和库将图导出为AIR；与目标DDK用途不同 |
| `CPRO_DDK_ID` | 所选DDK的准确版本标识，用于绑定/编译产物一致性检查；不会从路径猜版本 |
| `OHOS_NATIVE_SDK` | 编译手机端runner的OHOS Native SDK，必须指向含 `build/cmake/ohos.toolchain.cmake`、`llvm/bin/clang++`、`sysroot/` 的 `native/`；已有 `OHOS_SDK` 可用于填初值 |
| `BIP` | **HDC服务器的IP:端口**，例如远端转发的HDC服务；不是根据手机网络地址自动推断的值 |
| `SN` | HDC列出的目标设备序列号，用于选择具体手机 |
| `HIAI_INCLUDE_DIR` | 自动写入，默认引用当前 `DDK_PATH` 下 `ddk/ai_ddk_lib/include`，用于编译runner；生成时已有非 `auto` 值则保留 |
| `HIAI_LIBRARIES` | 自动写入，默认引用同一DDK的 `ddk/ai_ddk_lib/lib64/libhiai.so`、`libhiai_ir.so`、`libhiai_ir_build.so`，用冒号连接；生成时已有非 `auto` 值则保留 |

下面这些值可以从当前环境派生，不用把vendor脚本已经export的内容再抄一遍。必要时也可在私有配置中显式指定覆盖值。

| 可省略配置 | 默认来源与作用 |
|---|---|
| `CPRO_DDK_ROOT` | 使用已选DDK的 `DDK_PATH`；定位OMG、TBE和安装版算子源码。未加载DDK时需填写绝对路径 |
| `HDC` | 已声明的HDC路径，或PATH中的 `hdc`；执行设备传输与命令。找不到时需填写实际可执行文件 |

vendor环境中常见的变量含义如下；入口保留它们，不生成重复配置。路径存在只说明环境已声明，不代表芯片编译或上板通过。

| vendor变量 | 含义 |
|---|---|
| `DDK_PATH` | 当前目标DDK的根目录，供 `CPRO_DDK_ROOT` 派生 |
| `TOOLCHAIN_HOME` | vendor工具链的根目录/内部查找基准 |
| `HIAI_VERSION` | 此DDK环境中通常指OMG的master目录；不是板端 `HIAI_RUNTIME_VERSION` 字符串 |
| `PATH` | OMG、编译器、Python等可执行文件的查找路径 |
| `LD_LIBRARY_PATH` | 主机工具运行时的动态库查找路径；不是给手机部署的库列表 |
| `PYTHONPATH` | vendor Python模块查找路径；项目入口仅补入缺少的仓库模块路径 |

这些vendor变量不能替代 `CANN_ROOT`、OHOS Native SDK或准确DDK版本。新模板的HIAI默认值来自用户回贴的旧配置：`ddk/ai_ddk_lib/include` 与 `lib64` 三库。旧配置省略HIAI值或使用 `auto` 时，入口也优先识别该布局，检查每个库的ELF64、小端、AArch64、共享库类型，并排除明确的GLIBC依赖。该布局缺任一库时报告缺失，不换用 `tools_sysdbg` 或 `hiai_pipe_lib` 的单个同名库。显式配置仍优先，配置恢复或布局识别不表示OHOS ABI或设备验证已通过。

其他布局的自动查找只扫描已选DDK，深度最多8层，不跟随目录链接到其他安装；对 `libhiai.so` 做相同ELF/依赖检查，无 `ohos` 目录名要求，同一真实文件只计一次。唯一候选才自动采用；多个候选列出排除理由，要求显式配置。没有GLIBC依赖也不证明OHOS ABI兼容。

### 旧配置迁移：补齐HIAI三库

新生成的配置已经包含这两项，跳过本节。只有旧文件缺少这两项或仍保留旧HIAI值时，才在原文件补齐，既不用删除文件，也不用重新创建。DDK环境已source后，可在仓库根目录执行下面两步；不依赖尚未成功导出的 `REPO_DIR`。

```bash
cat >> ../qwen35-cpro.env <<'EOF'

# 之前使用的ai_ddk_lib头文件及完整三库；路径跟随已加载DDK。
HIAI_INCLUDE_DIR="${DDK_PATH:?先加载DDK环境}/ddk/ai_ddk_lib/include"
HIAI_LIBRARIES="${DDK_PATH}/ddk/ai_ddk_lib/lib64/libhiai.so:${DDK_PATH}/ddk/ai_ddk_lib/lib64/libhiai_ir.so:${DDK_PATH}/ddk/ai_ddk_lib/lib64/libhiai_ir_build.so"
EOF
source targets/cpro/scripts/env.sh
```

若此前指定了其他私有配置文件，将 `../qwen35-cpro.env` 换成报错显示的路径。末尾两项覆盖文件前面的旧HIAI值，其余配置保留。只有项目入口校验成功后才导出变量；失败时 `echo "$REPO_DIR"` 仍为空是整体加载未完成。无需因此重新设置所有项目变量。

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
| `PYTHONPATH` | 保留已有vendor路径，补入缺少的仓库模块路径；重复source不累加 |

入口保留当前 `PATH`，不再读取或导出 `MODEL_PYTHON`；旧配置中的这一行可删除。入口不激活conda、不执行vendor脚本、不追加全局CANN/DDK库路径，也不连接设备；GE/OMG继续通过现有脚本设置各自子进程环境。

以下变量属于单次运行，**不写入固定配置**，需要时在执行前设置或使用CLI参数。`source env.sh` 不读取、覆盖或清除调用者的这些值；旧配置有这些行可删除。

| 单次运行变量 | 含义 / 替代入口 |
|---|---|
| `FUSED_PROBE_RUN` | 已生成探针或批次目录；可直接使用 `--probe-run` |
| `TBE_DIAG_JOB` | 包含 `diagnosis/` 的诊断任务目录；新批次重试可自动继承记录，无需另行export |
| `GE_CUSTOM_BINDINGS` | 直接调用GE工具时使用的bindings JSON路径；`--fresh` 自动重新生成 |
| `FIRMWARE_ID` | 本次设备固件身份的显式值；未设置时脚本从设备查询 |
| `HIAI_RUNTIME_VERSION` | 本次预期板端HIAI版本；若设置，必须与实际runtime probe一致 |

`--fresh` 忽略旧产物环境变量，不会在本批中重复编译刚失败的case。SDK或HIAI库若此前也放在已删除目录内，必须恢复那些文件，仅重新export无法恢复工具链。

### 后续仅重试指定用例

用 `--model-ops --probe-run` 复用保留批次；只有所选六例缺项才会编译。确需限制重试范围时，附加 `--retry-cases` 和所选用例名。Conv/RMS已改走基础路径，不再作为模型模式的缺项重试；`--all-ops` 保留旧五类诊断口径，历史失败不改写。

## 按需检查与其他模型流程

完成上面的算子探针入口后，不需要依次再执行下方全部检查。以下用于完整模型、TorchAir或单独阶段排障。

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

已加载项目 `env.sh` 的终端无需重新export公共变量，只按需检查解释器。工作区托管任务沿用 `ws` 分配的 `AI_RUN_DIR`；独立clone使用配置中的固定运行根目录，不在每次source时重新分配目录。

```bash
test -f "$REPO_DIR/project.yaml"
python3 -c 'import sys, numpy; print(sys.executable); print("NumPy:", numpy.__version__)'
```

资源默认读取 [specs/resources.json](../specs/resources.json) 并展开环境变量；已有私有清单时设置 `QWEN35_RESOURCES=/实际路径/resources.json`。完整权重路径与锁定版本见[权重准备](weights.md)，每次命令使用新的输出目录。

## 2. PyTorch 环境

完整结构和 CPU/CUDA 参考使用现有 Torch；先检查导入。完整结构命令见 [PyTorch 指南](pytorch-guide.md)。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
```

仅走TorchAir导出路径时，才在单独终端按该TorchNPU/TorchAir安装要求加载服务器CANN环境并检查导入；当前直接GE融合探针不需要这一步。原生库或版本问题见 [AIR 排障](air-troubleshooting.md)。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 - <<'PY'
import torch_npu
import torchair
print("TorchNPU:", torch_npu.__version__)
print("TorchAir:", torchair.__file__)
assert callable(torchair.dynamo_export)
PY
```

## 3. 主机 GE：生成 AIR

`CANN_ROOT` 已从项目配置加载；直接GE构图的源码阶段不需要它，保存真实AIR才需要。当前直接GE导出器会为自己的子进程设置CANN库路径，无需再全局source CANN的 `set_env.sh`。`--cxx11-abi` 默认0，需与所选GE库匹配。

```bash
test -f "$CANN_ROOT/include/graph/graph.h"
test -f "$CANN_ROOT/include/graph/operator.h"
test -f "$CANN_ROOT/lib64/libgraph.so"
"${CXX:-c++}" --version
```

## 4. 目标 DDK：编译 OMC

第1步的 `DDK_PATH` 已由项目入口派生为 `CPRO_DDK_ROOT`，版本来自配置，无需再export。当前编译入口固定使用OMG的 `--platform=kirinx90`，文档bindings不证明kernel已安装或兼容。

```bash
test -x "$CPRO_DDK_ROOT/tools/tools_omg/master/omg"
```

## 5. OHOS / HIAI / HDC

项目入口已加载SDK并解析HIAI头文件/库路径，无需再抄一组export。若自动查找失败，只在私有配置中补 `HIAI_INCLUDE_DIR` 和 `HIAI_LIBRARIES`（多个库文件用冒号分隔），再次source项目入口。SDK获取方法见 [OHOS SDK](ohos-native-sdk.md)。

```bash
test -f "$OHOS_NATIVE_SDK/build/cmake/ohos.toolchain.cmake"
test -x "$OHOS_NATIVE_SDK/llvm/bin/clang++"
test -d "$OHOS_NATIVE_SDK/sysroot"
test -f "$HIAI_INCLUDE_DIR/HiAiModelManagerService.h"
cmake --version
```

设备序列号和HDC server地址已从配置加载；`BIP` 为server的 `IP:端口`。需要单独检查连接时执行下面命令；正常上板脚本会自行检查。保留runner构建目录中的 `command.json`、`result.json` 及其引用路径，上传脚本据此查找运行库。

```bash
"$HDC" -s "$BIP" list targets
```

## 内存预算：文档 W4A8 + DFlash

完整未量化 FP16 target+draft 去重权重约 **65.28 GiB**；不要以 32GB 板端容量运行它。文档 W4A8 + FP16 draft 的规划权重约 20 GiB，激活、缓存、HIAI workspace 和加载副本另计，真实峰值尚未验证；假设见[内存记录](../specs/device-memory-report.json)。

## tiny 回归

小模型只免除完整权重依赖；AIR、OMC 和板端阶段仍需对应工具链。命令放在 [PyTorch](pytorch-guide.md#tiny-回归) / [GE](ge-guide.md#tiny-回归) 指南末尾。
