# Qwen3.5-35B-A3B DFlash / CPro X90

部署链路为 **AIR → OMG → OMC → HarmonyOS HIAI**。新增直接 GE 构图的 target/draft 双图路径，保留原有 PyTorch/TorchAir 模型与导出入口。板端 C++ 使用 `hiai::AiModelMngerClient` 同步接口。精度要求是 **权重和激活可以量化，其余浮点计算 FP16**。

**当前操作入口：[Target / Draft 直接 GE 构图 → AIR](direct-model-air.md)。** 用户已反馈 Add 小图 AIR→OMC 成功；下一步在相同环境转换含 GDN/MoE/注意力的双图 tiny。新导出端只需 NumPy 与主机 GE，不加载 PyTorch/TorchNPU/TorchAir。完整结构定义已实现，实际模型 AIR/OMC 和上板仍待验证。新路径暂为 FP16 基线，压缩整数权重 GE 映射尚未实现。

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
