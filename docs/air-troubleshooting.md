# AIR 导出环境排障

**当前优先入口已增加 [GE 直接构图保存 AIR](direct-air-smoke.md)。** AIR 不要求必须从 PyTorch 导出。该最小加法图不加载 TorchNPU/TorchAir；需要 CANN 的主机 `graph/graph.h`、`libgraph.so` 及匹配依赖，后续仍由内部 OMG 转 OMC。下面的 TorchNPU/HCCL 排障针对现有 PyTorch/TorchAir 模型导出入口，不是直接 GE 样例的前置步骤。完整 Qwen/DFlash 尚未迁移到直接构图。

截至 2026-10-22 的最新用户日志：服务器曾安装 `torch-npu==2.11.0.post2`，因缺少 `libhccl.so` 无法导入；安装自己的 CANN 9.1.0 后仍缺该库。随后用户卸载 TorchNPU，CPU FP16 tiny demo 成功，生成 `tiny-reference.json`。日志未展示具体 token，也没有真实 AIR、OMC 或板端执行成功证据。

用户仍需保留 TorchNPU 以准备 AIR，因此项目现已为 CLI 的 CPU/CUDA 参考命令增加进程内后端隔离，不再把卸载 TorchNPU 作为参考流程的解决方法。

## 先区分三套环境

| 环节 | 所需环境 | 当前已知信息 |
|---|---|---|
| 服务器 PyTorch → AIR | 配套的 PyTorch、TorchNPU、TorchAir 和服务器 CANN 运行库 | 用户已卸载 wheel；重装后仍须补齐真实 CANN 原生依赖 |
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

**普通 Torch 可以导入，仍不等于 Qwen 模型可以导入。** Transformers 5.13.0 导入 Qwen MoE 的 Flash Attention 集成时会调用 `is_torch_npu_available()`，进而显式导入已安装的 TorchNPU。用户 traceback 已确认：关闭 Torch 自动加载后，CPU demo 仍在此处报 `libhccl.so`。这是可选后端探测失败，不代表 CPU 参考需要 HCCL。[Transformers 5.13.0 探测源码](https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/utils/import_utils.py)

当前 CLI 在 `demo` 或 `reference` 选择 `--device cpu` / `cuda:*` 时，于模型导入前关闭 Torch 后端自动加载，并在**当前 Python 进程内**把 Transformers 的两处 NPU 可用性探测设为 False。它不修改 site-packages、不卸载包，也不改变模型计算；进程退出后此设置消失。这是本项目对固定 Transformers 版本的兼容处理，不是官方环境开关。直接导入 `reference.tiny` 的自定义 Python 程序不会自动应用该策略，应使用 CLI，或在任何模型导入前显式调用 `configure_reference_backend("cpu")`。

`preflight` 在独立子进程中使用相同 CPU 策略检查真实 Qwen/DFlash 类，分别报告 `torch_import_ready`、`reference_import_ready`、`reference_import_error` 和 `reference_backend="cpu"`。导入检查不加载完整权重，不等于推理或 AIR 已通过。

**AIR 命令保留真实 NPU 依赖检查。** 从参考切换到导出时启动新的 CLI 进程；同一 Python 进程应用过参考隔离后，项目会拒绝继续导出 AIR，避免把被禁用的探测结果带入导出。

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

解决方向是提供本任务可用、版本配套的完整服务器 CANN 依赖，并加载该安装提供的环境脚本；若已安装，检查其真实运行库是否进入当前进程的库搜索路径。不要用开发包中的 stub/fake 库充当运行库，也不要将别人目录里的同名 `.so` 加入路径。只有 `libhccl_v2.so`、`libhccl_legacy.so` 等不能证明它们可直接代替 `libhccl.so`；确认官方安装规则前，不手工改名或补软链接。未确定受支持的服务器安装组合前，本指南不提供猜测的 ops 下载或安装命令。

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


## 5. 最新日志对应的下一步

用户已确认安装包为 `Ascend-cann-toolkit_9.1.0_linux-x86_64.run`。这确认了 Toolkit 包身份，尚不能证明对应 ops/HCCL 组件齐备。实际安装根目录中有 `set_env.sh`；用户已经成功 source 它，但仍没有 `libhccl.so`，所以继续重复添加相同库路径不能补出缺失文件。

先更新源码，再恢复本次确实被卸载的 TorchNPU wheel。以下重装命令适用于用户已确认的 PyTorch 2.11.0 / Python 3.11 环境；只恢复先前精确版本，不替换现有 CUDA Torch，也不代表原生依赖已解决：

```bash
git -C "$REPO_DIR" pull --ff-only
"$MODEL_PYTHON" -m pip install --no-deps 'torch-npu==2.11.0.post2'
cd "$AI_RUN_DIR"
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli demo \
  --device cpu --precision fp16 --sequence-length 8 \
  --prompt-token-ids 1,2,3 --max-new-tokens 4 \
  --output "$AI_RUN_DIR/tiny-reference-with-npu.json"
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli preflight \
  --output "$AI_RUN_DIR/preflight-with-npu.json"
```

CPU demo 不再需要用户设置 `TORCH_DEVICE_BACKEND_AUTOLOAD`；CLI 会选择 CPU 参考策略。`preflight` 如实保留自然 Torch/AIR 导入失败，并独立检查隔离后的 CPU 参考，所以可以出现 `torch_import_ready=false`、`reference_import_ready=true`、`air_export_ready=false`。不要把这个组合误判为矛盾。输出拒绝覆盖，已存在时更换文件名。

`The fast path is not available ... Falling back to torch implementation` 是本次已成功 CPU 运行中的回退提示，不要求为 tiny 验证安装 Flash Linear Attention 或 causal-conv1d。

AIR 缺库的下一项证据是实际安装命令、安装摘要和组件清单。只检查自己指定的根目录，例如：

```bash
export CANN_ROOT=/your/own/ascend/cann-9.1.0
source "$CANN_ROOT/set_env.sh"
bash "$CANN_ROOT/query_pkg_version.sh"
cat "$CANN_ROOT/x86_64-linux/ascend_toolkit_install.info"
cat "$CANN_ROOT/x86_64-linux/ascend_ops_install.info"
cat "$CANN_ROOT/share/info/hccl/ascend_install.info"
```

文件缺失也有诊断价值，请保留输出。官方 9.1 安装体系区分 Toolkit、ops 和 `hccl` 子包，可通过 `query_pkg_version.sh` 查询组件版本；应据清单补齐匹配组件，不能仅凭目标芯片 X90 猜一个公开 Ascend 产品的 ops 包。[官方组件查询说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/softwareinst/instg/instg_0030.html)

以上不修改库；`CPRO_DDK_ROOT` 继续指向内部 DDK。脚本修复覆盖 CPU/CUDA 参考与诊断入口，不能生成缺失的 CANN 运行库，也不能消除所选 TorchAir 实现的设备初始化要求。当前没有真实 AIR 或板端通过证据。


公开 TorchAir 的 CPU 示例也不能保证无卡导出。已核对的官方源码提交 `e1118e8b0fc1b95190ab53b732c0872989550e20` 中，GE 会话初始化之后仍执行 `aclrtSetDevice`；官方 CPU 调试构建使用的 fake 库不能替代生产运行库。这是源码观察，不是用户的 2.11.0.post2 wheel 已在无卡环境测试失败。[会话初始化源码](https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/torchair/concrete_graph/session.cpp#L42)
