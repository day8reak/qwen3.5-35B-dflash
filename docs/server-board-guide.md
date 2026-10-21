# 服务器生成与 CPro/X90 板端运行指南

**量化、AIR 导出、OMG 转 OMC、HIAI 程序交叉编译，全部在服务器完成。板端只执行编译好的 HarmonyOS ARM64 程序，通过 HIAI 加载两个 OMC。** 当前文本编码和结果解码也在服务器完成，板端程序接收 token IDs、输出结果 JSON。

本文按现有 CLI 和 C++ 接口整理操作顺序，不表示完整模型已经跑通。目前交付的是待联调源码：完整 35B 校准、AIR/OMC、内部 HIAI 链接、X90 执行均待验证。详见 [validation.md](validation.md)。

## 1. 两边分别做什么

| 操作 | 执行位置 | 输入 | 输出 | 是否送到板端 |
|---|---|---|---|---|
| 权重审计、内存规划 | 服务器 | 目标/配套草稿权重与配置 | 审计、量化计划 JSON | 否 |
| DOPT 量化校准 | 服务器 | 原始权重、校准 token | `calibrated/` 仿真检查点 | 否 |
| prompt 编码、参考推理 | 服务器 | tokenizer、文本、校准结果 | `prompt.json`、`reference.json` | token IDs 进入运行脚本；JSON 留服务器 |
| PyTorch → AIR | 服务器导出环境 | 原始权重、校准结果 | `air-s128/`，含外部权重 | 否 |
| AIR → OMG → OMC | 服务器 | AIR、DDK、kirinx90 插件 | 完整 `omc-s128/` | **是，整个目录** |
| C++ 交叉编译和链接 | 服务器 | 源码、OHOS SDK、内部 HIAI 头文件/库 | `qwen35_dflash_hiai` | **是** |
| HIAI 版本探测 | 板端 | 可执行文件、配套运行库 | `runtime-probe.json` | 取回服务器生成命令 |
| 生成运行命令 | 服务器 | OMC manifest、prompt、板端版本/路径 | `run-generate.sh`、`run-benchmark.sh` | **是** |
| 模型推理、普通/DFlash 测量 | 板端 | OMC、可执行文件、运行脚本 | 生成报告、benchmark 报告、日志 | 取回服务器 |
| token 对比、解码为文本 | 服务器 | 板端报告、参考、tokenizer | `comparison.json`、`decoded.json` | 否 |

服务器的 CUDA 校准环境和 AIR 导出环境可以分开；后者必须具备匹配的 TorchAir、torch_npu、CANN/DDK 及 DOPT 量化扩展。已有 `torch 2.11.0+cu130` 本身不能证明 AIR 可导出。**AIR 导出和 OMG 转换应在同一导出路径完成**，外部权重可能记录绝对路径；当前脚本会拒绝移动后的 AIR bundle。

原始权重必须能在内部服务器读取。如果现在只有板机上有权重，需要先在内网复制或挂载给服务器；不需要传给开发者。完整校准/参考仍会加载浮点模型，不能按 INT4 成品大小估算服务器内存或显存。

## 2. 服务器：准备源码、环境和路径

以下为独立 clone 场景。`/your/...` 均须替换为内部真实路径；所有输出放在仓库外。若在 AI model workspace 中操作，使用 `ws context`、`ws session start`、`ws env` 分配路径和解释器，不覆盖已分配的 `AI_RUN_DIR`。

**已有 AscendC 环境时直接复用。** 在已激活的 `custom_pypto` 环境中，首次进入或新开 shell 时执行一次下列脚本；当前 shell 已执行成功则跳过：

```bash
source /your/internal/ddk/tools/tools_ascendc/set_ascendc_env.sh
```

脚本已设置的 `TOOLCHAIN_HOME`、`DDK_PATH`、`HIAI_VERSION`、`PATH`、`LD_LIBRARY_PATH`、`PYTHONPATH` 无需再手工列一遍。下面仅补工程所需变量：`CPRO_DDK_ROOT` 复用 `DDK_PATH`，项目 Python 路径追加到已有路径前面。先 source DDK，再补项目路径，避免后续 source 将项目路径覆盖。未使用该脚本的环境才需要显式填写 `CPRO_DDK_ROOT`。


```bash
git clone git@github.com:day8reak/qwen3.5-35B-dflash.git
cd qwen3.5-35B-dflash
# 已有 clone 时，在该仓库执行 git pull --ff-only。
export REPO_DIR="$PWD"
export MODEL_PYTHON="$(command -v python3)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO_DIR/model:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export AI_RUN_DIR=/your/run/storage/qwen35-cpro-001
mkdir -p "$AI_RUN_DIR"
cd "$AI_RUN_DIR"

export QWEN35_TARGET_WEIGHTS=/your/checkpoints/Qwen3.5-35B-A3B
export QWEN35_DRAFT_WEIGHTS=/your/checkpoints/Qwen3.5-35B-A3B-DFlash
export CPRO_DDK_ROOT="$(realpath "${DDK_PATH:?先加载 DDK 环境脚本}")"
export CPRO_DDK_ID=your-exact-ddk-version
```

这些变量由仓库的 `specs/resources.json` 模板读取。需要自定义 `dopt_root` 或库列表时，复制该模板到本次输出目录，编辑后设置 `QWEN35_RESOURCES` 指向它。不要把内部路径或库提交 Git。复用当前 Python 环境和已安装组件；仅缺失时再补项目依赖。OHOS SDK 与 HIAI 头文件/链接库到第 5 节交叉编译时配置，前期无需填写。未填写时 `preflight` 对这些资源显示缺失，不影响先检查模型导出相关依赖。

```bash
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli preflight \
  --output "$AI_RUN_DIR/preflight.json"
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli quant-plan \
  --available-gib 18 --sequence-length 128 --group-size 128 \
  --output "$AI_RUN_DIR/quant-plan.json"
```

`preflight` 是清单检查，成功写 JSON 不代表环境已齐备；检查其中 `air_export_ready`、`dopt_air_ready` 和各资源的缺失信息。`--available-gib 18` 只是当前规划输入，需按实际可用预算更新。当前候选的权重估算 **21.19 GiB**，还未计工作区，超过用户报告的约 18G free；因此尚不能按此候选安排完整模型上板常驻。

所有 CLI 输出路径均拒绝覆盖。失败重试时换新输出目录/文件名，保留原日志。

## 3. 服务器：先验证导出环境，再校准完整模型

激活匹配的导出环境后，重新选择解释器，再执行小图检查。每次切换环境都要更新 `MODEL_PYTHON`；仅激活环境不会改变之前保存的解释器绝对路径。保持 `REPO_DIR`、`PYTHONPATH`、`AI_RUN_DIR` 和资源变量指向本次任务：

```bash
export MODEL_PYTHON="$(command -v python3)"
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli export-tiny-air \
  --sequence-length 8 --output "$AI_RUN_DIR/tiny-air"
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli compile-omc \
  --air "$AI_RUN_DIR/tiny-air" --output "$AI_RUN_DIR/tiny-omc"
```

这里的 tiny 是随机小模型，只检查转换链路，不是 35B 成品，也不覆盖完整量化图。完整量化还需要 `NpuQLinear` / `torch_npu.npu_quant_matmul_v4` 等匹配组件；当前公开环境尚不具备。

完整权重就绪后，激活校准环境并执行：

```bash
export MODEL_PYTHON="$(command -v python3)"
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli audit \
  --output "$AI_RUN_DIR/checkpoint-audit.json"
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli prompt \
  --text '请解释什么是推测解码。' --output "$AI_RUN_DIR/prompt.json"
```

自行准备 `$AI_RUN_DIR/calibration.json`，格式为 `{"token_ids": [[...], [...]]}`。使用目标 tokenizer 生成的代表性样本，每条长度不超过 128，并保留独立质量评测集。不能用随机 token 或这一条演示 prompt 代替正式校准集。

```bash
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli quantize --device cuda:0 \
  --calibration "$AI_RUN_DIR/calibration.json" \
  --sequence-length 128 --group-size 128 --output "$AI_RUN_DIR/calibrated"

PROMPT_IDS=$("$MODEL_PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["prompt_token_ids_csv"])' \
  "$AI_RUN_DIR/prompt.json")
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli reference \
  --device cuda:0 --precision fp16 --quantized-checkpoint "$AI_RUN_DIR/calibrated" \
  --sequence-length 128 --prompt-token-ids "$PROMPT_IDS" --max-new-tokens 32 \
  --output "$AI_RUN_DIR/reference.json"
```

`cuda:0` 适用于有足够显存的校准服务器；CPU 可用 `--device cpu`，仍需足够主存。`calibrated/target-simulation.pth` 是 DOPT 仿真检查点，不能作为压缩模型交给 HIAI。候选允许权重和激活量化，其余浮点计算为 FP16；量化后的普通/DFlash token 必须一致，量化相对原模型的质量另行评估。

## 4. 服务器：生成 AIR，再生成 OMC

重新激活匹配的导出环境，保持资源指向同一套原始权重与校准结果：

```bash
export MODEL_PYTHON="$(command -v python3)"
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli export-air \
  --sequence-length 128 --quantized-checkpoint "$AI_RUN_DIR/calibrated" \
  --output "$AI_RUN_DIR/air-s128"
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli compile-omc \
  --air "$AI_RUN_DIR/air-s128" --output "$AI_RUN_DIR/omc-s128"
```

`compile-omc` 内部调用 DDK 的 OMG，核心参数为 `--framework=1 --target=omc --platform=kirinx90`。不需要在板端运行 OMG。成功产物结构如下，实际附属文件以转换结果为准：

```text
air-s128/                        # 留在服务器，并保留原导出路径
  manifest.json
  target/                        # AIR + 全部外部权重
  draft/                         # AIR + 全部外部权重
omc-s128/                        # 整个目录传板端
  manifest.json
  target/
    target.omc
    command.json
    omg.log
    ...                          # OMG 生成的其他附属文件
  draft/
    draft.omc
    command.json
    omg.log
    ...
```

仅 `omc-s128/manifest.json` 中 `status=COMPILED` 表示转换编排完成；仍需板端验证。目录里只有 `failure.json` 时不能当作成品部署。固定长度档位为 128，prompt 加请求生成长度必须满足 `prompt_tokens + max_new_tokens - 1 <= 128`；更长上下文需要重新导出相应档位。

## 5. 服务器：交叉编译板端 HIAI 程序

到这一步再补板端编译资源；若这些工程变量已经设置，也可跳过重复赋值。AscendC 脚本中的服务器工具库路径不等于 HarmonyOS ARM64 链接库路径。

```bash
export OHOS_NATIVE_SDK=/your/ohos/native
export HIAI_INCLUDE_DIR=/your/internal/hiai/include
# 按实际链接要求填写 HarmonyOS ARM64 库的绝对路径，以冒号分隔。
export HIAI_LIBRARIES=/your/internal/lib/actual-library.so
"$MODEL_PYTHON" "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$AI_RUN_DIR/hiai-build"
```

检查 `$AI_RUN_DIR/hiai-build/result.json` 为 `LINKED`，正式二进制为：

```text
$AI_RUN_DIR/hiai-build/build/qwen35_dflash_hiai
```

构建使用 OHOS ARM64、`c++_shared` 和指定内部 HIAI 库。只有公开头文件时，`--headers-only` 只能得到 `HEADERS_COMPILED`，不会生成可执行文件；主机测试的 `hiai_runner_fake` 也不能传板端冒充正式程序。

在装有 binutils 的服务器可检查 ELF 与直接依赖：

```bash
readelf -h "$AI_RUN_DIR/hiai-build/build/qwen35_dflash_hiai"
readelf -d "$AI_RUN_DIR/hiai-build/build/qwen35_dflash_hiai"
```

按 `NEEDED` 及其传递依赖准备配套运行库，包括板端未提供的 HIAI 依赖和适用的 `libc++_shared.so`。具体库名以内部链接结果为准。板端系统已提供的匹配库可以由系统加载；额外库放部署目录的 `lib/`，符号链接须能在板端解析，不能指向服务器独有的绝对路径。这里必须是 **HarmonyOS ARM64** 库，不能复制服务器 DDK 的 Linux 工具库替代。内部库始终留在你们环境内。

## 6. 服务器 → 板端：第一次传文件

服务器可按以下方式整理部署目录：

```bash
mkdir -p "$AI_RUN_DIR/deploy/bin" "$AI_RUN_DIR/deploy/lib" "$AI_RUN_DIR/deploy/models"
cp "$AI_RUN_DIR/hiai-build/build/qwen35_dflash_hiai" "$AI_RUN_DIR/deploy/bin/"
cp -a "$AI_RUN_DIR/omc-s128" "$AI_RUN_DIR/deploy/models/"
# 将上一步确定的额外运行库复制到 deploy/lib/，保留库名和必要链接。
```

通过你们现有的板机文件传输通道，将 `deploy/` **目录内容**送到板端部署根目录。下文示例为 `/data/local/tmp/qwen35-cpro`，需替换为该板机允许写入、执行并访问 HIAI/NPU 的真实路径。该路径是板端路径，与服务器的 `AI_RUN_DIR` 不同。

```text
/data/local/tmp/qwen35-cpro/
  bin/qwen35_dflash_hiai
  lib/                           # 所需额外运行库；系统已提供时可为空
  models/omc-s128/
    manifest.json
    target/target.omc             # 以及 target/ 下其他文件
    draft/draft.omc               # 以及 draft/ 下其他文件
```

**不要只拷贝两个 `.omc` 而丢掉附属文件。** 当前普通生成也加载 target 和 draft 两个模型。原始 safetensors、`calibrated/`、AIR、PyTorch、DOPT、TorchAir、OMG、OHOS 编译 SDK 均留服务器；当前原生 runner 不需要板端安装 Python 或 tokenizer。

## 7. 板端：探测 HIAI 运行版本

以下命令在**板端 shell**执行：

```sh
export BOARD_ROOT=/data/local/tmp/qwen35-cpro
export CASE_ID=case001
mkdir -p "$BOARD_ROOT/runs/$CASE_ID"
chmod +x "$BOARD_ROOT/bin/qwen35_dflash_hiai"
export LD_LIBRARY_PATH="$BOARD_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
set -C
"$BOARD_ROOT/bin/qwen35_dflash_hiai" --probe \
  > "$BOARD_ROOT/runs/$CASE_ID/runtime-probe.json" \
  2> "$BOARD_ROOT/runs/$CASE_ID/probe.stderr.log"
```

`set -C` 防止重定向覆盖已有日志；重试换新的 `CASE_ID`。probe 成功应返回退出码 0，并输出 `runtime_version` 与 `npu_supported: true`。它只检查运行库/NPU 可用性，不加载 35B OMC。

把 `runtime-probe.json` 取回服务器，保存为 `$AI_RUN_DIR/runtime-probe.json`。同时记录内部实际固件版本，下一步用作 `FIRMWARE_ID`。程序记录该标识，不会自动核验固件身份。

## 8. 服务器：生成本次推理脚本，再传板端

回到**服务器 shell**。以下板端路径、`CASE_ID` 必须与上一步一致：

```bash
export BOARD_ROOT=/data/local/tmp/qwen35-cpro
export CASE_ID=case001
export FIRMWARE_ID='填写实际板机固件版本'
RUNTIME_VERSION=$("$MODEL_PYTHON" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); assert p["npu_supported"] is True; print(p["runtime_version"])' \
  "$AI_RUN_DIR/runtime-probe.json")

for MODE in generate benchmark; do
  "$MODEL_PYTHON" -m qwen35_moe_dflash.cli board-command \
    --omc "$AI_RUN_DIR/omc-s128" --prompt "$AI_RUN_DIR/prompt.json" \
    --board-model-dir "$BOARD_ROOT/models/omc-s128" \
    --runner "$BOARD_ROOT/bin/qwen35_dflash_hiai" \
    --board-output "$BOARD_ROOT/runs/$CASE_ID/$MODE.json" \
    --runtime-version "$RUNTIME_VERSION" --firmware-id "$FIRMWARE_ID" \
    --max-new-tokens 32 --mode "$MODE" \
    --output "$AI_RUN_DIR/board-$MODE-command.json" || exit 1
done

"$MODEL_PYTHON" - <<'PY'
import json, os
from pathlib import Path
run = Path(os.environ["AI_RUN_DIR"])
for mode in ("generate", "benchmark"):
    command = json.loads((run / f"board-{mode}-command.json").read_text())["shell"]
    with (run / f"run-{mode}.sh").open("x") as output:
        output.write("#!/bin/sh\nset -eu\n" + command + "\n")
PY
```

`board-command` 根据 OMC manifest 自动填写模型 SHA256、DDK ID、词表/特征宽度、EOS 和 token IDs，保留路径引号；无需手工拼长命令。它本身生成 JSON，后面的 Python 将 `shell` 字段写成脚本。

将这两个文件传到板端 `$BOARD_ROOT/runs/$CASE_ID/`：

```text
$AI_RUN_DIR/run-generate.sh
$AI_RUN_DIR/run-benchmark.sh
```

更换 prompt 时，服务器重新执行 `prompt`、相同校准模型的 `reference` 和本节脚本生成，使用新的文件名/CASE_ID；只要模型、量化方案和长度档位不变，无需重新生成 OMC。

## 9. 板端：单次推理，然后测量普通/DFlash

重新打开板端 shell 时，先恢复变量和运行库路径，再运行：

```sh
export BOARD_ROOT=/data/local/tmp/qwen35-cpro
export CASE_ID=case001
export LD_LIBRARY_PATH="$BOARD_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
set -C

sh "$BOARD_ROOT/runs/$CASE_ID/run-generate.sh" \
  > "$BOARD_ROOT/runs/$CASE_ID/generate.stdout.log" \
  2> "$BOARD_ROOT/runs/$CASE_ID/generate.stderr.log"
```

成功后得到 `$BOARD_ROOT/runs/$CASE_ID/generate.json`，内含生成的 token IDs、停止原因和耗时。默认使用 DFlash；当前程序不是交互式文本聊天终端。若单次生成失败，先取回 stderr 排查，不继续测量。

单次成功后执行：

```sh
sh "$BOARD_ROOT/runs/$CASE_ID/run-benchmark.sh" \
  > "$BOARD_ROOT/runs/$CASE_ID/benchmark.stdout.log" \
  2> "$BOARD_ROOT/runs/$CASE_ID/benchmark.stderr.log"
```

得到 `benchmark.json`：先检查普通/DFlash 协议一致性，再各做 3 次预热和 10 次交替测量。耗时包含图调用、特征拷贝和 host 调度，不包含模型加载和 tokenizer。不能把这份测量当作完整应用端到端耗时，也未承诺当前全专家重算实现有加速。

板端程序拒绝覆盖已有报告或同名 `.tmp`。重跑时生成新的输出路径/脚本，并保留失败报告和日志。

## 10. 板端 → 服务器：取回报告、验证和解码

把板端 `runs/$CASE_ID/` 中的 `generate.json`、`benchmark.json` 和所有日志取回，放在 `$AI_RUN_DIR/board-results/$CASE_ID/`。在**服务器**执行：

```bash
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli compare \
  --report "$AI_RUN_DIR/board-results/$CASE_ID/benchmark.json" \
  --reference "$AI_RUN_DIR/reference.json" --omc "$AI_RUN_DIR/omc-s128" \
  --output "$AI_RUN_DIR/comparison.json"
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli decode \
  --report "$AI_RUN_DIR/board-results/$CASE_ID/generate.json" \
  --output "$AI_RUN_DIR/decoded.json"
```

`compare` 必须使用双模式的 **benchmark 报告**，不能传单次 `generate.json`。它校验模型/量化/精度/ABI/OMC 身份、prompt、十轮 token 和停止原因。`decode` 用服务器的目标 tokenizer 将 token IDs 还原成文本；单次与 benchmark 报告都支持。

遇到问题，反馈失败阶段、`preflight.json`、对应构建/OMG/板端日志和 probe/报告即可。完整权重、校准样本和内部库继续保留在内部环境。尚未获得完整 OMC 与真机报告前，源码检查通过不能视为部署完成。
