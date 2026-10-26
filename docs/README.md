# Qwen3.5-35B-A3B DFlash / CPro X90

部署链路为 **AIR → OMG → OMC → HarmonyOS HIAI**。新增直接 GE 构图的 target/draft 双图路径，保留原有 PyTorch/TorchAir 模型与导出入口。板端 C++ 使用 `hiai::AiModelMngerClient` 同步接口。精度要求是 **权重和激活可以量化，其余浮点计算 FP16**。

**当前进度：用户已反馈 tiny target/draft 双图 AIR→OMC 编译成功。下一步：[直接 GE tiny OMC 上板指南](direct-ge-board.md)。** 按服务器参考、HIAI交叉编译、文件传输、板端运行、服务器对比分步执行。已成功的OMC和prepared无需重建。该进度是用户报告，尚未取得独立产物哈希或X90执行结果。

**服务器没有 OHOS SDK：[获取 Native SDK 与配置编译路径](ohos-native-sdk.md)。** 已找到的 HIAI `.so` 是链接/运行库，还需匹配的交叉编译器与 sysroot；SDK 留在服务器。

**HDC 自动上板：[传文件、设置环境、运行与结果回传](hdc-board-run.md)。** 设置 `SN`（设备ID）和 `BIP`（远程HDC服务端地址），统一使用 `hdc -t "$SN" -s "$BIP"`。准备好正式程序、OMC和匹配参考后，在服务器执行：

```bash
export SN='填写设备ID'
export BIP='填写远程HDC服务端IP:端口'
export HDC_RUN_DIR="$AI_RUN_DIR/hdc-$(date +%Y%m%d-%H%M%S)"
python3 "$REPO_DIR/targets/cpro/scripts/run_hdc.py" \
  --omc "$GE_OMC_DIR" \
  --runner "$HIAI_BUILD_DIR/build/qwen35_dflash_hiai" \
  --reference "$GE_BOARD_RUN/reference.json" \
  --output "$HDC_RUN_DIR"
```

无需设置库目录或增加库参数。脚本从 `--runner` 对应的构建目录读取 `command.json` / `result.json`，验证程序及链接库 SHA256，自动上传同套 HIAI 库，并从构建时的 OHOS SDK 补充可找到的 ARM64 C++ 运行库。保留构建目录、DDK 与 SDK 路径即可。

脚本会执行 probe、benchmark、结果回传与参考比较。全部成功后删除本次创建的板端目录；失败时保留排查，服务器结果始终保留。probe 或 benchmark 失败时会直接显示已回传的错误日志末尾，并自动抓取板端 HiLog 缓冲区快照作为排障材料。日志摘要会扫描整个快照、排除 HDC 传输记录，并避免将 `input` 误匹配为 `NPU`；完整日志始终保留。

遇到 `HIAI Load failed: 1`，且 HiLog 显示 HCL `CreateLocalBuffer` 的 `data != nullptr && size > 0` 检查失败，需要更新并重新编译推理程序：现将 OMC 只读映射后通过 `SetModelBuffer` 提供数据，保留路径，并检查加载状态及后续张量。映射保持到模型卸载和客户端销毁之后。该缓冲区接口仅接受不超过 `UINT32_MAX` 字节的单个 OMC，完整 35B 大文件加载仍待验证。OMC 和参考文件无需重建，命令见[板端排障与重试](hdc-board-run.md#结果与失败排查)。

最新用户反馈已进入实际 `Load`，target 返回 `1`（运行库 `800.631.120.010`），尚未进入推理。该通用返回码不足以确定原因；新脚本会自动回传 `hiai-hilog.txt`，先根据底层日志定位。此次诊断增强无需重建已有二进制和 OMC。

以后要保留大体积独立权重时，加 `--board-cleanup omc`，只删除清单中的 OMC 文件，保留目录、独立权重、程序、库和日志；加 `--board-cleanup none` 则全部保留。默认是 `--board-cleanup all`。嵌入 OMC 内部的权重随 OMC 一起删除。

[直接 GE 构图与导出](direct-model-air.md)保留全部生成命令及兼容性修复记录。生成端只需NumPy与主机GE，不加载PyTorch/TorchNPU/TorchAir。当前tiny使用FP16随机权重；完整35B的压缩整数权重GE映射尚未实现。

[Add 最小探针](direct-air-smoke.md)继续保留，可用于区分基础环境和模型算子转换问题。

**部署操作入口：[服务器生成与板端运行指南](server-board-guide.md)。** 按服务器校准/导出/交叉编译、文件传输、板端探测/推理、服务器对比/解码分别列出命令和产物路径。

**板端联调：[tiny AIR → OMC → HIAI 上板验证](tiny-validation.md)。** 第 2–5 节可用于新的直接 GE 双图；该页第 1 节保留旧 TorchAir 导出方式。两条 tiny 路径的随机权重不同，必须分别生成匹配参考，不混用报告。

**导出环境报错：[AIR 环境排障](air-troubleshooting.md)。** 说明 `libhccl.so` 缺失、Transformers 的显式 NPU 探测、CPU/CUDA 参考隔离及内部 DDK 布局。

这是待板端联调的源码版本。已有目标/草稿模型、贪心 DFlash、HIAI ABI v2、FP16 图检查、DOPT 校准和转换编排。**没有完整 35B AIR/OMC、内部 HIAI 链接或 X90 执行结果。** 主机验证范围与当前阻塞见 [validation.md](validation.md)。

## 权重来源

DFlash 官方代码：[z-lab/dflash（GitHub）](https://github.com/z-lab/dflash)。目标模型和配套草稿权重托管在 Hugging Face，以下下载链接固定到本工程锁定的版本：

| 用途 | 权重下载 | 锁定版本 |
|---|---|---|
| 目标模型 | [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/tree/59d61f3ce65a6d9863b86d2e96597125219dc754) | `59d61f3` |
| DFlash 草稿模型 | [z-lab/Qwen3.5-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash/tree/52cb554b4995dede3e2e1bdb129cdb1f3529332b) | `52cb554` |

完整 revision 和已核对配置/索引的哈希见 [upstream.lock.json](../specs/upstream.lock.json)。完整权重需另行下载，本仓库不包含权重或量化成品；当前开发环境仅下载并核对了配置/索引，完整 35B 权重验证仍待完成。

### 在服务器下载

在能访问 Hugging Face 及模型文件下载服务的服务器执行。没有 `hf` 命令时，使用下面的 Python 方式即可。两种方式都使用 `huggingface_hub`；当前 Python 环境已安装该包时跳过安装：

```bash
python3 -m pip install 'huggingface_hub>=1.5,<2'
```

沿用本次任务的 `AI_RUN_DIR`。未设置时，把下面的 `/your/run/storage/qwen35-cpro-001` 换成实际的仓库外目录；工作区用户直接使用 `ws session start` 分配的目录。

```bash
export AI_RUN_DIR="${AI_RUN_DIR:-/your/run/storage/qwen35-cpro-001}"
mkdir -p "$AI_RUN_DIR/weights"

export QWEN35_TARGET_WEIGHTS="$AI_RUN_DIR/weights/Qwen3.5-35B-A3B"
export QWEN35_DRAFT_WEIGHTS="$AI_RUN_DIR/weights/Qwen3.5-35B-A3B-DFlash"
```

下面两种下载方式任选一种，均固定到相同版本。

**方式一：Python 下载，不依赖 `hf` 命令。** 在设置好上述路径的同一终端执行：

```bash
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3.5-35B-A3B",
    revision="59d61f3ce65a6d9863b86d2e96597125219dc754",
    local_dir=os.environ["QWEN35_TARGET_WEIGHTS"],
)

snapshot_download(
    repo_id="z-lab/Qwen3.5-35B-A3B-DFlash",
    revision="52cb554b4995dede3e2e1bdb129cdb1f3529332b",
    local_dir=os.environ["QWEN35_DRAFT_WEIGHTS"],
)
PY
```

**方式二：已有 `hf` 命令时使用 CLI。**

```bash
# 目标模型：权重、配置和 tokenizer
hf download Qwen/Qwen3.5-35B-A3B \
  --revision 59d61f3ce65a6d9863b86d2e96597125219dc754 \
  --local-dir "$QWEN35_TARGET_WEIGHTS"

# 配套 DFlash 草稿模型
hf download z-lab/Qwen3.5-35B-A3B-DFlash \
  --revision 52cb554b4995dede3e2e1bdb129cdb1f3529332b \
  --local-dir "$QWEN35_DRAFT_WEIGHTS"
```

这里下载的是原始权重，后续在服务器量化。目标权重约 72 GB，下载目录建议至少留 100 GB 磁盘空间；校准、AIR、OMC 还需额外空间。下载中断后可重跑同一条命令或整段 Python，已完成且未变化的文件会复用，保留下载目录中的 `.cache/huggingface/` 元数据。参数与下载机制见 [Python 下载文档](https://huggingface.co/docs/huggingface_hub/guides/download#download-an-entire-repository)和 [CLI 文档](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-download)。

下载完成后，当前 shell 已设置好后续流程需要的两个权重路径。新开终端时重新设置它们，指向实际下载目录。

## 模型与精度

目标为锁定版本的 `Qwen/Qwen3.5-35B-A3B`，草稿为 `z-lab/Qwen3.5-35B-A3B-DFlash`。复用了 4B 工程的草稿代码和调度，不能直接使用 4B 草稿权重：隐藏维度分别为 2048/2560，拼接特征宽度分别为 16384/20480。35B 配套草稿约 3.86 亿参数。

目标取层 `[1,6,11,16,22,27,32,37]` 的特征，草稿 block 为 16，使用目标 embedding 和实际 untied LM head。固定 S、batch 1、纯文本、strict greedy；当前重算前缀，没有增量 KV/GDN 缓存。静态导出计算全部专家再动态选择 top-8，**尚不是高性能稀疏 MoE**，不能承诺 DFlash 加速。

FP16 部署会显式改写上游主动升为 FP32 的 GDN、归一化、路由、RoPE 等图节点，再捕获检查；这会改变数值，不等同于原始 BF16/FP32 模型。量化算子内部的参数/统计类型单独记录。图间特征为 FP16，token/mask 为 INT32；旧 FP32 特征 OMC 必须重新生成。

量化候选：蓝区 DOPT `Quant_act_weight_eco`，权重 4 bit、group 128、`group_min_max`，激活 16 bit、EMA。这里 A16 是 **DOPT 激活量化**，不能直接解释为 FP16 矩阵输入。embedding、LM head、草稿、路由和小型 GDN gate 等保留 FP16。该候选尚未通过全权重质量门槛。激活量化器内部用 FP32 完成缩放/统计，输出回 FP16，避免已复现的 FP16 缩放溢出；这属于量化区域的精度策略。

完整结构统计覆盖 40×256=10,240 个专家、20,480 个专家矩阵（gate/up 合并）。权重及 FP16 scale/offset、两份 embedding/head 估算合计 **22.75 GB / 21.19 GiB**；未计布局、LUT、HIAI 工作区、中间张量或加载副本。用户当前约 18G free 无法满足这一估算；实际可用 32GiB 也只能表示有余量，不能证明峰值可容纳。大页池不作为额外 RAM。

## 主机入口

独立 clone 后从仓库根目录设置；输出始终放在仓库外的当前任务目录。已有内部 DDK 时，先在当前 Python 环境 source 其 AscendC 脚本，再设置以下项目变量；后续追加项目 Python 路径时保留脚本已有路径。保留现有 Torch，不自动替换用户的 `2.11.0+cu130`。本机验证解释器是 Torch `2.12.1+cpu` / Transformers `5.13.0`，现有 PyTorch 导出入口需要另外匹配 TorchAir、torch_npu 和服务器 CANN 用户态库；直接 GE 样例使用独立的主机 GE 依赖。内部 DDK 负责后续 OMG 转换，不能仅凭其环境脚本成功就视为满足任一种 AIR 生成入口的依赖。

```bash
export REPO_DIR="$PWD"
export MODEL_PYTHON="$(command -v python3)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO_DIR/model:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export AI_RUN_DIR="${AI_RUN_DIR:-/your/run/storage/qwen35-cpro-001}"
mkdir -p "$AI_RUN_DIR"
cd "$AI_RUN_DIR"

"$MODEL_PYTHON" -m qwen35_moe_dflash.cli demo --sequence-length 8 \
  --max-new-tokens 4 --output "$AI_RUN_DIR/demo.json"
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli quant-plan --available-gib 18 \
  --output "$AI_RUN_DIR/quant-plan.json"
```

`quant-plan` 使用 meta tensor，无需权重，输出每个量化矩阵的名称/形状和预算。`--available-gib` 是可用进程预算，不是整机标称 RAM。

工作区场景先运行 `ws context`、`ws session start` 和 `ws env`，使用分配的输出路径；当前 `cpro-blue-pending` profile 尚未登记，环境激活不能视为通过。

## 资源与量化

入口通过 `project.yaml` 的资源声明或 `QWEN35_RESOURCES` 读取本地 JSON；私有路径、库和权重不提交。已有 `set_ascendc_env.sh` 环境时，先 source 一次并保留它设置的路径，无需重复配置 AscendC。下例从已有 `DDK_PATH` 取得根目录；未使用该脚本时才显式填写根目录。OHOS SDK 和 HIAI 资源可留到交叉编译时配置。模板可使用这些变量：

```bash
export QWEN35_TARGET_WEIGHTS="${QWEN35_TARGET_WEIGHTS:-/your/checkpoints/Qwen3.5-35B-A3B}"
export QWEN35_DRAFT_WEIGHTS="${QWEN35_DRAFT_WEIGHTS:-/your/checkpoints/Qwen3.5-35B-A3B-DFlash}"
export CPRO_DDK_ROOT="$(realpath "${DDK_PATH:?先加载 DDK 环境脚本}")"
export CPRO_DDK_ID=your-exact-ddk-version

"$MODEL_PYTHON" -m qwen35_moe_dflash.cli preflight --output "$AI_RUN_DIR/preflight.json"
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli audit --output "$AI_RUN_DIR/checkpoint-audit.json"
```

DOPT 默认取 `ddk_root/tools/tools_dopt/dopt_pytorch_py3`，也可在资源 JSON 设置 `dopt_root`。校准直接调用 DOPT 模块接口，避免其训练脚本的模型加载假设；先把融合专家权重暴露为独立 Linear，并检查覆盖率。完整校准/参考仍需容纳浮点模型的主机内存或显存，不能在 18G 可用内存的板机上运行这个主机流程。

自行准备有代表性的校准 JSON：`{"token_ids": [[...], [...]]}`。token 序列来自锁定的目标 tokenizer；不能把少量随机 token 当作正式校准数据。单条长度不能超过 S；保留独立的质量评测集。

```bash
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli quantize --device cuda:0 \
  --calibration "$AI_RUN_DIR/calibration.json" --sequence-length 128 --group-size 128 \
  --output "$AI_RUN_DIR/calibrated"
```

产物包括配置、矩阵计划、校准 token、vendor 组件哈希和 `target-simulation.pth`。这是**仿真检查点，不是压缩部署模型**。校准或推理出现非有限数值会失败；不能通过放宽 token 检查掩盖。官方接口背景见 [插件式量化](https://developer.huawei.com/consumer/cn/doc/doccenter-capabilities/cannkit-plug-in-quantification) 与 [LLM 量化流程](https://developer.huawei.com/consumer/cn/doc/doccenter-capabilities/cannkit-large-language-one-stop)。

## 参考与 AIR / OMC

```bash
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli prompt --text '请解释什么是推测解码。' \
  --output "$AI_RUN_DIR/prompt.json"
PROMPT_IDS=$("$MODEL_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["prompt_token_ids_csv"])' "$AI_RUN_DIR/prompt.json")
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli reference --device cuda:0 --precision fp16 \
  --quantized-checkpoint "$AI_RUN_DIR/calibrated" --sequence-length 128 \
  --prompt-token-ids "$PROMPT_IDS" --max-new-tokens 32 --output "$AI_RUN_DIR/reference.json"
```

量化参考使用实际 DOPT 仿真算子，普通区域按 FP16 图执行。Cython 层通过主机专用桥接调用，不能把该参考图用作 AIR。省略 `--quantized-checkpoint` 可检查未量化 FP16；`--precision upstream-bf16` 保留原始上游参考。量化后的普通/DFlash 必须严格一致；量化与原始模型的质量评测另做，不能要求二者 token 天然一致。

在匹配的导出环境先检查小图，然后尝试完整量化图：

```bash
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli export-tiny-air --sequence-length 8 \
  --output "$AI_RUN_DIR/tiny-air"
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli compile-omc --air "$AI_RUN_DIR/tiny-air" \
  --output "$AI_RUN_DIR/tiny-omc"
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli export-air --sequence-length 128 \
  --quantized-checkpoint "$AI_RUN_DIR/calibrated" --output "$AI_RUN_DIR/air-s128"
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli compile-omc --air "$AI_RUN_DIR/air-s128" \
  --output "$AI_RUN_DIR/omc-s128"
```

蓝区实际包暴露 `get_air_model`，但本机缺少其 `NpuQLinear` / `torch_npu.npu_quant_matmul_v4` 扩展，**尚未验证它能导出本模型**。导出检查原生量化 MatMul 数量、整数打包字节数和普通区域 FP16；若内部布局不同，应核对接口后适配，不能跳过检查。浮点诊断导出必须显式使用 `--float-baseline`，它不适合当前板机内存。

OMG 参数为 `--framework=1 --model=input.air --output=target --target=omc --platform=kirinx90 --use_origin_format=true --weight_data_type=FP16`。最后一项只控制原始 FP32 权重转换，不代替量化。转换在 run 内隔离目录中使用短相对路径。保留 AIR 的全部外部权重文件及原导出路径；manifest 锁定所有 payload，不能只复制 `.air`。

蓝区工具准备：`python3 "$REPO_DIR/targets/cpro/scripts/prepare_blue_ddk.py" --output "$AI_RUN_DIR/blue-ddk"`。脚本核对固定官方包 SHA256，在 run 内解压，不修改共享 DDK。ONNX 仅保留为额外的 CPU 数学测试，不参与生产转换。

## HIAI 构建、执行和对比

此时再配置 OHOS SDK 和内部链接资源；已配置过则复用现有值。

```bash
export OHOS_NATIVE_SDK=/your/ohos/native
export HIAI_INCLUDE_DIR=/your/internal/hiai/include
# 填写实际 HarmonyOS ARM64 库，以冒号分隔；以下仅为路径占位。
export HIAI_LIBRARIES=/your/internal/lib/libhiai.so:/your/internal/lib/libhiai_ir.so
"$MODEL_PYTHON" "$REPO_DIR/targets/cpro/scripts/build_hiai.py" --output "$AI_RUN_DIR/hiai-build"
# 只有公开头文件时可加 --headers-only；不会生成可运行程序。
```

将二进制、匹配的库和完整 OMC 目录部署到用户的 X90 板机，先运行 `./qwen35_dflash_hiai --probe`。内部库无需传出。普通和 DFlash 都使用完整词表 Top1，持久张量，HIAI 同步 Process；目标特征通过 host memcpy 传给草稿。

```bash
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli board-command \
  --omc "$AI_RUN_DIR/omc-s128" --prompt "$AI_RUN_DIR/prompt.json" \
  --board-model-dir /your/board/omc-s128 --board-output /your/board/report.json \
  --runtime-version 'probe返回的版本' --firmware-id '实际固件版本' \
  --max-new-tokens 32 --mode benchmark --output "$AI_RUN_DIR/board-command.json"
```

板端执行 JSON 中的 `shell`。默认 `--mode generate` 单次生成；benchmark 各做 3 次 warmup、10 次交替测量，包含拷贝与 host 调度，排除加载和 tokenizer。报告取回后：

```bash
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli compare --report "$AI_RUN_DIR/board-report.json" \
  --reference "$AI_RUN_DIR/reference.json" --omc "$AI_RUN_DIR/omc-s128" \
  --output "$AI_RUN_DIR/comparison.json"
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli decode --report "$AI_RUN_DIR/board-report.json" \
  --output "$AI_RUN_DIR/decoded.json"
```

比较会检查权重、校准、精度、ABI、OMC 哈希、输入和全部十轮 token/停止原因。模拟 HIAI 不能冒充板端结果。

## 本地检查

```bash
"$MODEL_PYTHON" -m pytest "$REPO_DIR/tests" -p no:cacheprovider --basetemp "$AI_RUN_DIR/pytest"
# 可选：真实 DOPT 集成测试使用蓝区二进制；不是 mock，也不是板端执行。
export QWEN35_TEST_DOPT_ROOT="$CPRO_DDK_ROOT/tools/tools_dopt/dopt_pytorch_py3"
"$MODEL_PYTHON" -m pytest "$REPO_DIR/tests/test_quantization.py" -p no:cacheprovider \
  --basetemp "$AI_RUN_DIR/pytest-dopt"
cmake -S "$REPO_DIR/targets/cpro/runtime" -B "$AI_RUN_DIR/native-tests" -DBUILD_HOST_TESTS=ON
cmake --build "$AI_RUN_DIR/native-tests" -j4
ctest --test-dir "$AI_RUN_DIR/native-tests" --output-on-failure
```

反馈 `preflight.json`、失败阶段的错误/日志和板端 probe/报告即可；权重、内部头文件和库保持在用户环境。

## 板端算子探测

模型 OMC 能编译但在板端加载失败时，可先逐项检查小图的编译、加载、执行和数值结果。复用已配置的 `CANN_ROOT`、`CPRO_DDK_ROOT`、`HIAI_BUILD_DIR`、`SN` 和 `BIP`；其中 `HIAI_BUILD_DIR` 是此前成功运行 `build_hiai.py` 的输出目录，内含 `command.json` 及原链接库记录。主机 Python 需要 NumPy，GE 导出和 OMG 使用现有工具链。脚本从构建记录复用 OHOS SDK、头文件及匹配的 HIAI 库，自动构建独立的 `qwen35_hiai_op_probe`，无需额外的 lib 参数。

```bash
python3 "$REPO_DIR/targets/cpro/scripts/probe_hdc_ops.py" \
  --output "$AI_RUN_DIR/op-probe-$(date +%Y%m%d-%H%M%S)"
```

默认套件包含 47 个小图，覆盖当前模型中的重点算子和变体，包括 MatMul、BatchMatMul、激活、类型转换、索引、广播，以及四维/五维输入差异。MatMul 包含输出宽度为 1、2、8 的常量/动态右输入，BatchMatMul 包含卷积降解所用的 `[48,8,4] × [48,4,1]`。每个案例生成独立 AIR/OMC 和输入，按 `hdc -t "$SN" -s "$BIP"` 上传后实际调用 HIAI Load 与 Process，并取回输出和日志。单个案例编译或执行失败会记录后继续测试其他案例；全部案例编译失败时返回非零退出码。

输出目录包含 `report.json`、`report.txt` 和 `results.tar.gz`。报告分别记录各阶段结果；浮点数学案例使用 FP16 参考对比，默认 `rtol=0.01`、`atol=0.002`，纯复制/重排和整数结果要求精确一致。小图成功只能证明该输入形状、类型和工具链下的案例通过，不能据此宣称整个算子支持所有形状，或完整模型全部由 NPU 执行。算子执行成功与设备分区证据分别记录。

可按需要添加以下选项：

- `--prepare-only`：仅生成案例和参考输入输出，不需要编译器或板机。
- `--compile-only`：生成并编译 AIR/OMC，跳过板端执行和原生 runner 构建。
- `--list-cases`：打印全部案例名称，不需要编译器或板机。
- `--case NAME`：仅运行指定案例，可重复传入；名称可先通过 `--list-cases` 查看。
- `--keep-board`：取回结果后仍保留本次创建的板端目录。

默认在结果成功取回并校验后删除本次脚本创建的板端目录；传输、校验等基础设施错误时保留现场。已有板端目录及其他任务文件不清理。

### NPU 小图编译成功但全部加载失败

若案例的 `hilog.txt` 出现 `Model exist graph op with NPUCL, but this cl not existed.`，表示模型需要 NPUCL，而当前 HIAI 进程未注册该计算后端。新版报告会在原生 Load 失败且对应 PID 日志含此消息时标记 `NPUCL_NOT_REGISTERED`。这时优先核对配套的板端运行库、插件及注册配置；不能据此认定 MatMul 不支持 NPU，也不能通过降低精度解决后端缺失。

2026-10-23 用户回传的 47 项结果中，45 项编译成功：35 项含 NPU 分区的案例全部 Load 失败，10 项 CPU-only 案例全部 Load 成功，其中 7 项通过数值检查。另 3 项 Gather/TopK/Cumsum 在 IO 描述检查时停止，需要核对实际 shape/format。新版 runner 会输出全部输入输出的描述和具体不匹配项，不会按元素数量相同而跳过布局校验。完整观察及限制见 [用户探测记录](../specs/operator-support-probe-user-report.json)。

无需重新编译即可先看已回传的日志：

```bash
cat "$OP_PROBE_RUN/board/matmul_m8_b_dynamic/hilog.txt"
cat "$OP_PROBE_RUN/board/add_rank4/hilog.txt"
```

若确认 NPUCL 未注册，可用以下只读命令核对 DDK 与板端候选库。库名和 ELF 架构仅用于定位，不能证明该库与当前固件兼容；此前缺失 `libai_fmk_dnnacl.so` 的日志也不证明只补这一个文件就足够。

```bash
find "$CPRO_DDK_ROOT/ddk" -maxdepth 6 \
  \( -name 'libai*.so*' -o -name 'libhiai*.so*' -o -name '*npu*.so*' -o -name '*dnnacl*.so*' \) \
  -exec file {} +

hdc -t "$SN" -s "$BIP" shell 'for d in /system/lib64 /vendor/lib64 /system/vendor/lib64 /chip_prod/lib64 /sys_prod/lib64; do
  if [ -d "$d" ]; then
    find "$d" -maxdepth 4 \( -name "libai*.so*" -o -name "libhiai*.so*" -o -name "*npu*.so*" -o -name "*dnnacl*.so*" \)
  fi
done'
```

配套后端恢复后，先在新输出目录只复验 `--case matmul_m8_b_dynamic --case add_rank4`，通过 Load、Process 和数值检查后再扩大测试。编译分区与实际执行位置仍分别记录。

### 复用原产物对照 DDK 与板端 HCL 后端

若板端已有 `/vendor/lib64/chipsetsdk/libai_fmk_hcl_model_runtime.so` 等 NPU 运行库，但探针进程仍报 NPUCL 未注册，可检查自动上传的 DDK HCL 库是否影响了后端选择。已有 `OP_PROBE_RUN` 无需重编译：

```bash
export RUNTIME_AB_RUN="${OP_PROBE_RUN}-runtime-ab-$(date +%Y%m%d-%H%M%S)"
python3 "$REPO_DIR/targets/cpro/scripts/probe_hdc_runtime.py" \
  --probe-run "$OP_PROBE_RUN" \
  --output "$RUNTIME_AB_RUN"
```

脚本仍使用 `SN`、`BIP`、`AI_RUN_DIR`，从原探测记录读取 runner 和库，不需要 lib 参数或编译环境。复用同一 runner、OMC、输入和参考，校验原哈希；选取 `matmul_m8_b_dynamic`、`add_rank4` 及 CPU 对照 `add_rank5`。两个独立部署分别是：

- `ddk_baseline`：原探针的完整上传库集合及搜索顺序。
- `candidate_without_ddk_hcl`：仅不上传 `libhiai_hcl_model_runtime.so` 与 `libhiai_hcl_model_runtime_impl.so`，在进程局部搜索路径中加入已存在的 `/vendor/lib64/chipsetsdk`。它是待验证的后端选择实验。

执行前直接解析 ELF 动态依赖表；若保留库直接依赖被排除库或存在相关 SONAME 冲突，候选测试会标记阻塞。脚本不执行目标二进制来分析依赖，也不依赖主机 `readelf` 的 DWARF 支持。检查只能覆盖 ELF 显式依赖，动态加载及 namespace 限制要由本次运行日志确认。

输出 `runtime-comparison.txt`、`runtime-comparison.json` 和可反馈的 `results.tar.gz`，各组保留独立报告、PID 日志及数值对比。默认取回校验后清理本次创建的目录，`--keep-board` 可保留；原探测目录、原库文件和板端系统库均不修改。即使候选组成功，也需确认后端选择和实际执行位置，再决定是否调整正式部署策略。

若候选组已通过 Load，却只报 `OMC tensor shape/format differs from IO TSV`，原探针可能早于详细 IO 诊断版本。可在新目录加 `--refresh-runner`：

```bash
export RUNTIME_IO_RUN="${OP_PROBE_RUN}-runtime-io-$(date +%Y%m%d-%H%M%S)"
python3 "$REPO_DIR/targets/cpro/scripts/probe_hdc_runtime.py" \
  --probe-run "$OP_PROBE_RUN" \
  --output "$RUNTIME_IO_RUN" \
  --refresh-runner
cat "$RUNTIME_IO_RUN/runtime-comparison.txt"
cat "$RUNTIME_IO_RUN/arms/candidate_without_ddk_hcl/board/matmul_m8_b_dynamic/stderr"
```

此模式需要原构建记录中的 OHOS SDK、HIAI 头文件和 CMake 可用，只构建一次当前诊断 runner，两组共用新二进制；原 OMC、输入、参考和库集合继续按原哈希复用。它不会重新导出 AIR 或编译 OMC。新旧 runner、当前源码、SDK/头文件及链接库的身份写入构建记录，报告注明已刷新 runner。新版会在比较前输出全部实际形状和格式，并指出具体不匹配项；dtype 无法由该旧版 HIAI 查询接口直接读取，仍由原产物合约约束。保持严格校验，不能仅凭元素总数相同跳过格式或布局差异。

已针对板端实测的 ND 尾部补维增加兼容：逻辑形状 `[8,16]` 与运行时 `[8,16,1,1]` 可以对应。仅接受原 rank 为 1～3、运行时恰好 rank 4、原维度前缀完全相同且新增尾维全部为 `1` 的情况，格式必须仍为 ND。探针记录该对应关系，用运行时返回的描述创建张量，输入输出字节及数值门槛不变；原样匹配的其他 rank 继续支持。前部或中间补维、非单例尾维、维度置换及非 ND 格式仍会拒绝。更新后用上面的 `--refresh-runner` 命令和新输出目录复测即可；该规则当前仅用于算子探针，完整模型 runner 尚未据此调整。
