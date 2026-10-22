# 小模型 AIR → OMC → HIAI 板端验证

仓库已有 FP16 tiny 链路：使用真实 Qwen MoE / Gated DeltaNet 和 DFlash 结构，固定随机种子 17；目标 4 层、草稿 2 层、hidden 16、词表 64。它不需要下载完整权重、tokenizer 或做 DOPT 校准，适合先检查转换与板端接口；随机 token 不代表正常文本回答，也不能验证 35B 的质量、量化或性能。

**目前主机参考已验证，但真实 tiny AIR、OMC、正式 HIAI 链接及 X90 执行尚未验证。** 以下是待执行的联调流程。tiny 仍需要匹配的 TorchAir / torch_npu / CANN 导出环境；已有 CUDA PyTorch 或成功加载 AscendC 脚本，并不代表可以导 AIR。CPU 输入也不能保证所选 TorchAir 版本无需 Ascend 设备初始化。

环境缺库或 Python 依赖冲突时，先看 [AIR 环境排障](air-troubleshooting.md)。CLI 的 CPU/CUDA `demo` 和 `reference` 会在当前进程禁用 NPU 可选探测，可以保留已安装的 TorchNPU；导出命令在新进程中仍检查真实 NPU/CANN 依赖。该处理不修改厂商包或模型计算。

## 1. 服务器：生成参考、AIR 和 OMC

参考与 AIR 使用匹配的导出环境；OMG 使用已加载内部 DDK 的环境，两套库不兼容时分开终端执行。下面路径需要替换，输出目录必须在仓库外且本次未使用；若已分配 `AI_RUN_DIR`，保留它。每次切换环境后重新设置 `MODEL_PYTHON`，参考和导出使用同一套解释器及依赖。

```bash
cd /your/repos/qwen3.5-35B-dflash
export REPO_DIR="$PWD"
export MODEL_PYTHON="$(command -v python3)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO_DIR/model:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export AI_RUN_DIR="${AI_RUN_DIR:-/your/run/storage/qwen35-tiny-001}"
mkdir -p "$AI_RUN_DIR"
cd "$AI_RUN_DIR"
export CPRO_DDK_ROOT="$(realpath "${DDK_PATH:?先加载内部 DDK 环境}")"
export CPRO_DDK_ID=your-exact-ddk-version

"$MODEL_PYTHON" -m qwen35_moe_dflash.cli demo --device cpu --precision fp16 \
  --sequence-length 8 --prompt-token-ids 1,2,3 --max-new-tokens 4 \
  --output "$AI_RUN_DIR/tiny-reference.json"
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli export-tiny-air \
  --sequence-length 8 --output "$AI_RUN_DIR/tiny-air"
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli compile-omc \
  --air "$AI_RUN_DIR/tiny-air" --output "$AI_RUN_DIR/tiny-omc"

"$MODEL_PYTHON" - <<'PY'
import json, os
from pathlib import Path
run = Path(os.environ["AI_RUN_DIR"])
reference = json.loads((run / "tiny-reference.json").read_text())
with (run / "tiny-prompt.json").open("x") as output:
    json.dump({"prompt_token_ids": reference["prompt_token_ids"]}, output)
PY
```

如果 AIR 导出环境使用独立的 CANN 库，OMG 转换应在另一个终端加载内部 DDK，并恢复同一 `REPO_DIR`、`AI_RUN_DIR` 和项目 Python 路径，避免混用两套运行库。

逐条确认成功再继续；参考应为 `PASS`，AIR manifest 应为 `EXPORTED`，OMC manifest 应为 `COMPILED`。保留 AIR 的原导出路径及全部外部权重。tiny 不启用 EOS，参考和板端都以请求长度/上下文边界停止。这里 `3 + 4 - 1 <= 8`，不要沿用完整模型的默认生成长度 32。

两张图的接口与正式 runner 相同：target 输入 INT32 `[1,8]` IDs/mask，输出 INT32 `[1,8]` Top1 和 FP16 `[1,8,32]` 特征；draft 输入特征、IDs/mask，输出 INT32 `[1,3]`。宽度与词表由 manifest 自动传入，不要手写 35B 的参数。

## 2. 服务器：编译程序并传文件

按[服务器与板端指南第 5 节](server-board-guide.md)配置 OHOS Native SDK、内部 HIAI 头文件和 HarmonyOS ARM64 链接库，再执行：

```bash
"$MODEL_PYTHON" "$REPO_DIR/targets/cpro/scripts/build_hiai.py" \
  --output "$AI_RUN_DIR/hiai-build"
```

`result.json` 必须为 `LINKED`。通过现有传输通道，按下表放到板端部署根目录；示例 `/data/local/tmp/qwen35-tiny-001` 需要换成板机允许读写、执行及访问 HIAI 的路径。

| 服务器文件 | 板端相对部署根目录 |
|---|---|
| `hiai-build/build/qwen35_dflash_hiai` | `bin/qwen35_dflash_hiai` |
| 整个 `tiny-omc/`，包含 manifest 和全部附属文件 | `models/tiny-omc/` |
| 板端未提供的匹配运行库及其依赖 | `lib/` |

`--headers-only` 和主机测试的 `hiai_runner_fake` 都不能替代正式程序。AIR、Python 环境和 tiny 参考留在服务器。

## 3. 板端：探测 HIAI 版本

```sh
export BOARD_ROOT=/data/local/tmp/qwen35-tiny-001
mkdir -p "$BOARD_ROOT/lib"
chmod +x "$BOARD_ROOT/bin/qwen35_dflash_hiai"
export LD_LIBRARY_PATH="$BOARD_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
set -C
"$BOARD_ROOT/bin/qwen35_dflash_hiai" --probe \
  > "$BOARD_ROOT/runtime-probe.json" 2> "$BOARD_ROOT/probe.stderr.log"
```

退出码应为 0，JSON 中 `npu_supported` 为 `true`。将此 JSON 取回服务器，保存为 `$AI_RUN_DIR/runtime-probe.json`；同时记录实际板机固件版本。

## 4. 服务器生成命令，板端运行

回到服务器，保持第 1 节的变量；`BOARD_ROOT` 必须与板端一致：

```bash
export BOARD_ROOT=/data/local/tmp/qwen35-tiny-001
export FIRMWARE_ID='填写实际板机固件版本'
RUNTIME_VERSION=$("$MODEL_PYTHON" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); assert p["npu_supported"] is True; print(p["runtime_version"])' \
  "$AI_RUN_DIR/runtime-probe.json")
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli board-command \
  --omc "$AI_RUN_DIR/tiny-omc" --prompt "$AI_RUN_DIR/tiny-prompt.json" \
  --board-model-dir "$BOARD_ROOT/models/tiny-omc" \
  --runner "$BOARD_ROOT/bin/qwen35_dflash_hiai" \
  --board-output "$BOARD_ROOT/benchmark.json" \
  --runtime-version "$RUNTIME_VERSION" --firmware-id "$FIRMWARE_ID" \
  --mode benchmark --max-new-tokens 4 --output "$AI_RUN_DIR/tiny-command.json"
"$MODEL_PYTHON" - <<'PY'
import json, os
from pathlib import Path
run = Path(os.environ["AI_RUN_DIR"])
command = json.loads((run / "tiny-command.json").read_text())["shell"]
with (run / "run-tiny.sh").open("x") as output:
    output.write("#!/bin/sh\nset -eu\n" + command + "\n")
PY
```

将 `run-tiny.sh` 传到板端 `$BOARD_ROOT/`。在第 3 节的板端 shell 中执行；新开 shell 时先恢复 `BOARD_ROOT`、`LD_LIBRARY_PATH` 和 `set -C`：

```sh
sh "$BOARD_ROOT/run-tiny.sh" \
  > "$BOARD_ROOT/benchmark.stdout.log" 2> "$BOARD_ROOT/benchmark.stderr.log"
```

benchmark 先检查普通/DFlash 一致性，再各做 3 次预热和 10 次测量。取回 `benchmark.json` 为服务器 `$AI_RUN_DIR/tiny-benchmark.json`，并保留 stdout/stderr；失败时先检查日志，不继续比较。

## 5. 服务器：比较 token 和停止原因

```bash
"$MODEL_PYTHON" -m qwen35_moe_dflash.cli compare \
  --report "$AI_RUN_DIR/tiny-benchmark.json" --reference "$AI_RUN_DIR/tiny-reference.json" \
  --omc "$AI_RUN_DIR/tiny-omc" --output "$AI_RUN_DIR/tiny-comparison.json"
```

`PASS` 表示本次真实 HIAI 报告的普通/DFlash token 和停止原因均符合参考，并通过模型身份等检查。这里不运行 `prompt`/`decode`，它们面向完整目标模型的 tokenizer。所有产物拒绝覆盖；重试使用新的服务器输出路径和板端报告/脚本路径，保留失败日志。tiny 通过后再推进完整量化模型，不能据此宣称 35B 已通过。
